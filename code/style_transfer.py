import os
from vocab import Vocabulary, build_vocab
from file_io import load_sent, write_sent
from options import load_arguments
import sys
import time
# import ipdb
import random
import _pickle as pickle

from utils import *
# from nn import *

import numpy as np
import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def timeSince(since, percent):
    now = time.time()
    s = now - since
    es = s / (percent)
    rs = es - s
    return '%s (- %s)' % (asMinutes(s), asMinutes(rs))

class Encoder(nn.Module):
    def __init__(self, input_size, hidden_size, dropout_p):
        super(Encoder, self).__init__()
        self.hidden_size = hidden_size
        self.input_size = input_size
        self.dropout_p = dropout_p
        # print("Encoder: ", input_size)
        self.embedding = nn.Embedding(input_size, hidden_size)
        self.gru = nn.GRU(hidden_size, hidden_size, dropout=self.dropout_p)

    def forward(self, input, hidden):

        # embedded_input = self.embedding(input)  
              
        input = input.view(1,1,-1)
        output, hidden = self.gru(input, hidden)
        return output, hidden

    def initHidden(self):
        return torch.zeros(1, 1, self.hidden_size, device=device)

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
        # output = self.embedding(input).view(1, 1, -1)
        # output = F.relu(output)
        input = input.view(1,1,-1)
        output, hidden = self.gru(input, hidden)
        output = self.softmax(self.out(output[0]))
        return output, hidden

    def initHidden(self):
        return torch.zeros(1, 1, self.hidden_size, device=device)

class Model():
    def __init__(self, input_size_enc, hidden_size_enc, 
    hidden_size_gen, output_size_gen, dropout_p):

        self.input_size_enc = input_size_enc
        self.hidden_size_enc = hidden_size_enc
        self.hidden_size_gen = hidden_size_gen
        self.output_size_gen = output_size_gen
        self.dropout_p = dropout_p
        self.generator1 = Generator(self.hidden_size_gen, self.output_size_gen, self.dropout_p)
        # self.generator2 = Generator(self.hidden_size_gen, self.output_size_gen, self.dropout_p)
        # print("Model: ", self.input_size_enc)
        self.encoder = Encoder(self.input_size_enc, self.hidden_size_enc, self.dropout_p)
        self.softmax = torch.nn.LogSoftmax(self.output_size_gen)
        self.EOS_token = 2
        self.GO_token = 1
        self.k = 5
        self.gamma = 0.001

    # def forward(self, input):
    #     encoder_hidden = self.encoder.initHidden()
    #     generator1_hidden = self.generator1.initHidden()
    #     generator2_hidden = self.generator2.initHidden()

        
            # self.encoder(input, encoder_hidden)

    def train_util(self, input_data, target, enc_optim, gen1_optim, criterion):
        # , max_length=MAX_LENGTH)

        enc = self.encoder
        gen1 = self.generator1
        # gen2 = self.generator2

        encoder_hidden = enc.initHidden()

        enc_optim.zero_grad()
        gen1_optim.zero_grad()

        input_length = input_data.size(0)
        # print("input_size: ", enc.input_size)
        # print("input_data: ", input_data)
        input_tensor = enc.embedding(input_data)

        # print("input_tensor")
        encoder_outputs = torch.zeros(input_length, enc.hidden_size, device=device)
        gen1_hid_states = []
        # gen2_hid_states = []

        for ei in range(input_length):
            # print("input index: ", ei)
            encoder_output, encoder_hidden = enc(
                input_tensor[ei].view(1,-1), encoder_hidden)
            encoder_outputs[ei] = encoder_output        

        gen1_input = torch.tensor([[self.GO_token]], device=device)
        
        # gen2_input = torch.tensor(torch.cat([(target+1)%2,encoder_outputs[-1]]), device=device)

        gen1_hidden = encoder_hidden
        # gen2_hidden = torch.cat([(target+1)%2,encoder_outputs])

        gen1_output = torch.zeros(self.output_size_gen)
        # gen2_output = torch.zeros(self.output_size_gen)
        loss = 0
        for i in range(input_length):
            gen1_input = enc.embedding(gen1_input)
            gen1_output, gen1_hidden = gen1(
                gen1_input, gen1_hidden)
            # target_tensor = nn.functional.one_hot(input_data[i],num_classes=gen1_output.shape[1])
            # target_tensor = target_tensor.view(1,-1)
            # print("gen1_output: ", gen1_output)
            # print("target: ", input_data[i].view(1))
            loss += criterion(gen1_output, input_data[i].view(1))

            gen1_hid_states.append(gen1_hidden)
            topv, topi = gen1_output.topk(1)
            gen1_input = topi.squeeze().detach()
            # gen1_input = input_tensor[i]  # Teacher forcing
            # gen1_input = self.softmax(gen1_output)
            if torch.argmax(gen1_output) == self.EOS_token:
                break

        # while np.argmax(gen2_output) != self.EOS_token:
        # for i in range(input_length):
        #     gen2_output, gen2_hidden = gen2(
        #         gen2_input, gen2_hidden, encoder_outputs)
        #     gen2_hid_states.append(gen2_hidden)
        #     topv, topi = gen2_output.topk(1)
        #     gen2_input = topi.squeeze().detach()  # detach from history as 
        #     gen2_input = self.softmax(gen2_input/self.gamma)

        #     loss += criterion(gen2_output, input_data[i])

        #     if np.argmax(gen2_output) == self.EOS_token:
        #         break
        #     # if decoder_input.item() == EOS_token:
        #     #     break

        loss.backward()

        enc_optim.step()
        gen1_optim.step()
        # gen2_optim.step()

        # return loss.item() / target_length
        return encoder_outputs[-1], gen1_hid_states, loss.item() / input_length

    def train(self, training_data, print_every=1000, plot_every=100, learning_rate=0.01):
        start = time.time()
        plot_losses = []
        print_loss_total = 0  # Reset every print_every
        plot_loss_total = 0  # Reset every plot_every

        enc_optim = optim.SGD(self.encoder.parameters(), lr=learning_rate)
        gen1_optim = optim.SGD(self.generator1.parameters(), lr=learning_rate)
        # gen2_optim = optim.SGD(self.generator2.parameters(), lr=learning_rate)

        criterion = nn.NLLLoss()
        indices = np.arange(training_data.shape[0])
        
        for i in range(len(indices)):
            input_data = training_data[i]        
            # target_tensor = training_pair[1]

            latent_z, gen1_hid_states, loss = self.train_util(input_data, 1, enc_optim, gen1_optim, criterion)
            print_loss_total += loss
            # plot_loss_total += loss
            # print("loss: ", loss)
            # print("Total loss: ", print_loss_total)
            # if i % print_every == 0:
            #     print_loss_avg = print_loss_total / print_every
            #     print_loss_total = 0
            #     print('%s (%d %d%%) %.4f' % (timeSince(start, iter / n_iters),
            #                                  iter, iter / n_iters * 100, print_loss_avg))

            # if iter % plot_every == 0:
            #     plot_loss_avg = plot_loss_total / plot_every
            #     plot_losses.append(plot_loss_avg)
            #     plot_loss_total = 0

        # showPlot(plot_losses)
        return print_loss_total

def get_model(args, vocab):
    dim_hidden = args.dim_y+args.dim_z    
    print("vocab size: ", vocab.size)
    model = Model(vocab.size, dim_hidden, 
    dim_hidden, vocab.size, args.dropout_keep_prob)
    return model

if __name__ == '__main__':
    args = load_arguments()
    # args.train = "../data/yelp/sentiment.train"
    # args.dev = "../data/yelp/sentiment.dev"
    # args.output = "../tmp/sentiment.dev"
    # args.vocab = "../tmp/yelp.vocab"
    # args.model = "./tmp/model"
    #####   data preparation   #####
    if args.train:
        train0 = load_sent(args.train + '.0', args.max_train_size)
        train1 = load_sent(args.train + '.1', args.max_train_size)
        # print((train0+train1)[len(train0)])
        # print(train1[0])
        print('#sents of training file 0:', len(train0))
        print('#sents of training file 1:', len(train1))

        if not os.path.isfile(args.vocab):
            build_vocab(train0 + train1, args.vocab)
    # print("args.embedding: ", args.embedding)
    # print("args.dim_emb: ", args.dim_emb)
    vocab = Vocabulary(args.vocab, args.embedding, args.dim_emb)
    # print('vocabulary size:', vocab.size)

    # if args.dev:
    #     dev0 = load_sent(args.dev + '.0')
    #     dev1 = load_sent(args.dev + '.1')

    # if args.test:
    #     test0 = load_sent(args.test + '.0')
    #     test1 = load_sent(args.test + '.1')
    if args.train:
        model = get_model(args, vocab)
        
        losses_epochs = []
        
        no_of_epochs = 100
        for iter in range(no_of_epochs):
            random.shuffle(train0)
            random.shuffle(train1)
            batches, _, _ = get_batches(train0, train1, vocab.word2id,
            args.batch_size, noisy=True)
            random.shuffle(batches)
            print("Epoch: ", iter)
            for batch in batches:
                # print(len(batch["enc_inputs"]))
                batch_input = batch["enc_inputs"]
                batch_input = torch.tensor(np.array(batch_input))
                # print("batch_input.shape: ", batch_input.shape)
                loss = model.train(batch_input)
                print("Loss: ", loss)
            print("---------\n")
                

    # print(train0)