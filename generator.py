import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pdb
import math
import torch.nn.init as init


class Generator(nn.Module):

    def __init__(self, embedding_dim, hidden_dim, vocab_size, max_seq_len, gpu=False, oracle_init=False):
        super(Generator, self).__init__()
        self.hidden_dim = hidden_dim
        self.embedding_dim = embedding_dim
        self.max_seq_len = max_seq_len
        self.vocab_size = vocab_size
        self.gpu = gpu

        self.embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.gru = nn.GRU(embedding_dim, hidden_dim)
        self.gru2out = nn.Linear(hidden_dim, vocab_size)

        # initialise oracle network with N(0,1)
        # otherwise variance of initialisation is very small => high NLL for data sampled from the same model
        if oracle_init:
            for p in self.parameters():
                init.normal(p, 0, 1)

    def init_hidden(self, batch_size=1):
        h = autograd.Variable(torch.zeros(1, batch_size, self.hidden_dim))

        if self.gpu:
            return h.cuda()
        else:
            return h

    def forward(self, inp, hidden):
        """
        Embeds input and applies GRU one token at a time (seq_len = 1)
        """
        # input dim                                             # batch_size
        emb = self.embeddings(inp)                              # batch_size x embedding_dim
        emb = emb.view(1, -1, self.embedding_dim)               # 1 x batch_size x embedding_dim
        out, hidden = self.gru(emb, hidden)                     # 1 x batch_size x hidden_dim (out)
        out = self.gru2out(out.view(-1, self.hidden_dim))       # batch_size x vocab_size
        out = F.log_softmax(out)
        return out, hidden

    def sample(self, num_samples, start_letter=0):
        """
        Samples the network and returns num_samples samples of length max_seq_len.

        Outputs: samples, hidden
            - samples: num_samples x max_seq_length (a sampled sequence in each row)
        """

        samples = torch.zeros(num_samples, self.max_seq_len).type(torch.LongTensor)

        h = self.init_hidden(num_samples)
        inp = autograd.Variable(torch.LongTensor([start_letter]*num_samples))

        if self.gpu:
            samples = samples.cuda()
            inp = inp.cuda()

        for i in range(self.max_seq_len):
            out, h = self.forward(inp, h)               # out: num_samples x vocab_size
            out = torch.multinomial(torch.exp(out), 1)  # num_samples x 1 (sampling from each row)
            samples[:, i] = out.data

            inp = out.view(-1)

        return samples

    # NEW FUNCTION
    def sample_rollout_init(self, num_samples):
        """
        Initializes the sequence.
        """
        
        samples = torch.zeros(num_samples, 1).type(torch.LongTensor)

        h = self.init_hidden(num_samples)
        inp = autograd.Variable(torch.LongTensor([0]*num_samples))

        if self.gpu:
            samples = samples.cuda()
            inp = inp.cuda()

        out, h = self.forward(inp, h)               # out: num_samples x vocab_size
        out = torch.multinomial(torch.exp(out), 1)  # num_samples x 1 (sampling from each row)
        samples[:, 0] = out.data

        inp = out.view(-1)

        return samples        

    # NEW FUNCTION
    def sample_rollout(self, num_samples, previous_seq, position):
        """
        Adds one token to the sequence being generated
        """
        samples = torch.zeros(num_samples, position+1).type(torch.LongTensor)
        samples[:, 0:position] = previous_seq

        h = self.init_hidden(num_samples)
        inp = autograd.Variable(torch.LongTensor([0]*num_samples))

        if self.gpu:
            samples = samples.cuda()
            inp = inp.cuda()

        out, h = self.forward(inp, h)               # out: num_samples x vocab_size
        out = torch.multinomial(torch.exp(out), 1)  # num_samples x 1 (sampling from each row)
        samples[:, -1] = out.data

        inp = out.view(-1)

        return samples

    # NEW FUNCTION
    def mc(self, samples, position, start_letter = 0, gpu=False):
        """
        Finishes the current sequence being generated with a Monte Carlo sampling from the Generator.
        """

        batch_size, current_seq_len = samples.size()

        inp = torch.zeros(batch_size, self.max_seq_len)
        target = torch.zeros(batch_size, self.max_seq_len)
        target[:, 0:current_seq_len] = samples
        inp[:, 0] = start_letter
        inp[:, 1:current_seq_len+1] = target[:, :current_seq_len]

        inp = autograd.Variable(inp).type(torch.LongTensor)
        target = autograd.Variable(target).type(torch.LongTensor)

        if self.gpu:
            inp = inp.cuda()
            target = target.cuda()

        h = self.init_hidden(batch_size)

        for i in range(current_seq_len, self.max_seq_len):
            out, h = self.forward(inp[:,i],h)
            out = torch.multinomial(torch.exp(out), 1)
            target[:, i] = out.data
            if i < self.max_seq_len-1:
                inp[:, i+1] = out.data

        return inp, target

    def batchNLLLoss(self, inp, target):
        """
        Returns the NLL Loss for predicting target sequence.

        Inputs: inp, target
            - inp: batch_size x seq_len
            - target: batch_size x seq_len

            inp should be target with <s> (start letter) prepended
        """

        loss_fn = nn.NLLLoss()
        batch_size, seq_len = inp.size()
        inp = inp.permute(1, 0)           # seq_len x batch_size
        target = target.permute(1, 0)     # seq_len x batch_size
        h = self.init_hidden(batch_size)

        loss = 0
        for i in range(seq_len):
            out, h = self.forward(inp[i], h)
            loss += loss_fn(out, target[i])

        return loss     # per batch

    def batchPGLoss(self, inp, target, reward):
        """
        Returns a pseudo-loss that gives corresponding policy gradients (on calling .backward()).
        Inspired by the example in http://karpathy.github.io/2016/05/31/rl/

        Inputs: inp, target
            - inp: batch_size x seq_len
            - target: batch_size x seq_len
            - reward: batch_size (discriminator reward for each sentence, applied to each token of the corresponding
                      sentence)

            inp should be target with <s> (start letter) prepended
        """

        batch_size, seq_len = inp.size()
        inp = inp.permute(1, 0)          # seq_len x batch_size
        target = target.permute(1, 0)    # seq_len x batch_size
        h = self.init_hidden(batch_size)

        loss = 0
        for i in range(seq_len):
            out, h = self.forward(inp[i], h)
            # TODO: should h be detached from graph (.detach())?
            for j in range(batch_size):
                loss += -out[j][target.data[i][j]]*reward[j]     # log(P(y_t|Y_1:Y_{t-1})) * Q

        return loss/batch_size

