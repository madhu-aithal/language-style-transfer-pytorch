import numpy as np
import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F

class Discriminator(nn.Module):
    def __init__(self, embedding_size, output_size, dropout_p=0.5):
        super(Discriminator, self).__init__()
        self.embedding_size = embedding_size
        self.output_size = output_size
        self.dropout_p = dropout_p
        self.conv1 = nn.Conv2d(1,100,(3,self.embedding_size))
        self.conv2 = nn.Conv2d(1,100,(4,self.embedding_size))
        self.conv3 = nn.Conv2d(1,100,(5, self.embedding_size))
        in_layer_fc = 100+100+100
        self.fc1 = nn.Linear(in_layer_fc, 10)
        self.fc2 = nn.Linear(10, 2)
        self.dropout = nn.Dropout(p = dropout_p)

        self.softmax = nn.Softmax(dim=1)

    def forward(self, input):
        sentence_length = input.shape[2]
        h1 = F.relu(self.conv1(input))
        h2 = F.relu(self.conv2(input))
        h3 = F.relu(self.conv3(input))

        h1 = F.max_pool2d(h1, ((sentence_length-3+1), 1))
        h2 = F.max_pool2d(h2, ((sentence_length-4+1), 1))
        h3 = F.max_pool2d(h3, ((sentence_length-5+1), 1))
        
        h1 = h1.view(-1, self.num_flat_features(h1))
        h2 = h2.view(-1, self.num_flat_features(h2))
        h3 = h3.view(-1, self.num_flat_features(h3))
        h = torch.cat((h1,h2,h3), dim=1)
        h = self.dropout(h)
        h = self.fc1(h)
        h = F.relu(h)
        h = self.fc2(h)
        h = F.relu(h)
        y = self.softmax(h)

        return y

    def num_flat_features(self, x):
        size = x.size()[1:]
        num_features = 1
        for s in size:
            num_features *= s
        return num_features