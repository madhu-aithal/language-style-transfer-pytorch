import numpy as np
import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F

class Encoder(nn.Module):
    def __init__(self, input_size, embedding_size, hidden_size, dropout_p):
        super(Encoder, self).__init__()
        self.hidden_size = hidden_size
        self.input_size = input_size
        self.embedding_size = embedding_size
        self.dropout_p = dropout_p
        self.embedding = nn.Embedding(input_size, embedding_size)
        self.gru = nn.GRU(embedding_size, hidden_size, dropout=self.dropout_p)

    def forward(self, input, hidden):     
        input = torch.unsqueeze(input, 0)         
        # input = input.view(1,input.shape[0], input.shape[1])
        output, hidden = self.gru(input, hidden)
        return output, hidden

    def initHidden(self, device):
        return torch.zeros(1, 1, self.hidden_size, device=device)