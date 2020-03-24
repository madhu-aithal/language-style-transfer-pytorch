import numpy as np
import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F

class Discriminator(nn.Module):
    def __init__(self, args, device, embedding_size, output_size, dropout_p=0.5):
        super(Discriminator, self).__init__()
        self.embedding_size = embedding_size
        self.output_size = output_size
        self.dropout_p = dropout_p
        self.device = device
        self.filter_sizes = [int(x) for x in args.filter_sizes.split(',')]
        self.n_filters = args.n_filters
        self.convs = []
        for size in self.filter_sizes:
            self.convs.append(nn.Conv2d(1,self.n_filters,(size,self.embedding_size)))

        in_layer_fc = len(self.filter_sizes)*self.n_filters
        self.fc1 = nn.Linear(in_layer_fc, self.output_size)
        self.dropout = nn.Dropout(p = dropout_p)

        self.sigmoid = nn.Sigmoid()

    def forward(self, input):
        sentence_length = input.shape[2]
        h = []
        for conv in self.convs:
            conv = conv.to(self.device)
            temp = F.leaky_relu(conv(input))
            h.append(temp)
        
        h_new = []
        for val, size in zip(h, self.filter_sizes):
            h_new.append(F.max_pool2d(val, ((sentence_length-size+1), 1)))        
        
        for idx,val in enumerate(h_new):
            h_new[idx] = val.view(-1, self.num_flat_features(val))
        h1 = torch.cat(h_new, dim=1)        
        h2 = self.dropout(h1)
        h3 = self.fc1(h2)
        y = self.sigmoid(h3)
        if False in (y==y):
            print()
        return y

    def num_flat_features(self, x):
        size = x.size()[1:]
        num_features = 1
        for s in size:
            num_features *= s
        return num_features