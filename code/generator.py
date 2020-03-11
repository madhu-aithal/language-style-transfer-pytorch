import numpy as np
import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F

class Generator(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, dropout_p):
        super(Generator, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.dropout_p = dropout_p

        # self.embedding = nn.Embedding(output_size, hidden_size)
        self.gru = nn.GRU(self.input_size, self.hidden_size, dropout = self.dropout_p)
        self.out = nn.Linear(hidden_size, output_size)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, input, hidden):
        # input = input.view(1,1,-1)
        # input = torch.unsqueeze(input, 0)
        # print("input: ", input.size())
        # print("hidden: ", hidden.size())
        output, hidden = self.gru(input, hidden)
        # assert not False in (hidden==hidden)
        # assert not False in (output==output)
        output = self.out(output[0])
        # assert not False in (output==output)
        # output = self.softmax(self.out(output[0]))
        return output, hidden

    def initHidden(self, device):
        return torch.zeros(1, 1, self.hidden_size, device=device)