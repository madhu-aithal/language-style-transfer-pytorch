import numpy as np
import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F

class Generator(nn.Module):
    def __init__(self, hidden_size, output_size, dropout_p):
        super(Generator, self).__init__()
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.dropout_p = dropout_p

        self.gru = nn.GRU(hidden_size, hidden_size, dropout = self.dropout_p)
        self.out = nn.Linear(hidden_size, output_size)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, input, hidden):
        input = input.view(1,1,-1)
        output, hidden = self.gru(input, hidden)
        output = self.softmax(self.out(output[0]))
        return output, hidden

    def initHidden(self, device):
        return torch.zeros(1, 1, self.hidden_size, device=device)