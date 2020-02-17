import logging
import os
from vocab import Vocabulary, build_vocab
from file_io import load_sent, write_sent
from options import load_arguments
import sys
import time
import math
import random
from datetime import datetime
import _pickle as pickle
from utils import *

import numpy as np
import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F

from encoder import Encoder
from generator import Generator

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def initLogging(log_dir, no_of_epochs):
    filename = str(datetime.now().strftime('app'+str(no_of_epochs)+'_%H_%M_%d_%m_%Y.log'))
    path = os.path.join(log_dir, filename)
    logging.basicConfig(filename=path, filemode='w', format='%(name)s - %(levelname)s - %(message)s')
    logger=logging.getLogger() 
    logger.setLevel(logging.DEBUG)
    return logger 

def asMinutes(s):
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)

def timeSince(since, percent):
    now = time.time()
    s = now - since
    es = s / (percent)
    rs = es - s
    return '%s (- %s)' % (asMinutes(s), asMinutes(rs))


class Model():
    def __init__(self, input_size_enc, hidden_size_enc, 
    hidden_size_gen, output_size_gen, dropout_p):

        self.input_size_enc = input_size_enc
        self.hidden_size_enc = hidden_size_enc
        self.hidden_size_gen = hidden_size_gen
        self.output_size_gen = output_size_gen
        self.dropout_p = dropout_p
        self.generator1 = Generator(self.hidden_size_gen, self.output_size_gen, self.dropout_p).to(device)
        self.encoder = Encoder(self.input_size_enc, self.hidden_size_enc, self.dropout_p).to(device)
        self.softmax = torch.nn.LogSoftmax(self.output_size_gen)
        self.EOS_token = 2
        self.GO_token = 1
        self.k = 5
        self.gamma = 0.001

    def train_util(self, input_data, target, enc_optim, gen1_optim, criterion):

        if torch.cuda.is_available():
            input_data = input_data.cuda()
        enc = self.encoder
        gen1 = self.generator1

        encoder_hidden = enc.initHidden(device)

        enc_optim.zero_grad()
        gen1_optim.zero_grad()

        input_length = input_data.size(0)

        input_tensor = enc.embedding(input_data)
        if torch.cuda.is_available():
            input_tensor = input_tensor.cuda()

        encoder_outputs = torch.zeros(input_length, enc.hidden_size, device=device)
        gen1_hid_states = []

        for ei in range(input_length):
            encoder_output, encoder_hidden = enc(
                input_tensor[ei].view(1,-1), encoder_hidden)
            encoder_outputs[ei] = encoder_output        

        gen1_input = torch.tensor([[self.GO_token]], device=device)
        
        gen1_hidden = encoder_hidden

        gen1_output = torch.zeros(self.output_size_gen, device=device)

        loss = 0

        for i in range(input_length):
            gen1_input = enc.embedding(gen1_input)
            gen1_output, gen1_hidden = gen1(
                gen1_input, gen1_hidden)

            loss += criterion(gen1_output, input_data[i].view(1))

            gen1_hid_states.append(gen1_hidden)
            topv, topi = gen1_output.topk(1)
            gen1_input = topi.squeeze().detach()

            if torch.argmax(gen1_output) == self.EOS_token:
                print("EOS token generated early")
                logger.info("EOS token generated early")
                break

        loss.backward()

        enc_optim.step()
        gen1_optim.step()
        return encoder_outputs[-1], gen1_hid_states, loss.item() / input_length

    def train(self, training_data, learning_rate=0.01):
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
            
        return print_loss_total

def get_model(args, vocab):
    dim_hidden = args.dim_y+args.dim_z    
    print("vocab size: ", vocab.size)
    logger.info("vocab size: "+str(vocab.size))
    model = Model(vocab.size, dim_hidden, 
    dim_hidden, vocab.size, args.dropout_keep_prob)
    return model

if __name__ == '__main__':
    args = load_arguments()

    no_of_epochs = args.max_epochs
    
    logger = initLogging(args.log_dir, no_of_epochs)

    #####   data preparation   #####
    if args.train:
        train0 = load_sent(args.train + '.0', args.max_train_size)
        train1 = load_sent(args.train + '.1', args.max_train_size)

        print('#sents of training file 0:', len(train0))
        print('#sents of training file 1:', len(train1))

        logger.info('#sents of training file 0: ' + str(len(train0)))
        logger.info('#sents of training file 1: ' + str(len(train1)))

        if not os.path.isfile(args.vocab):
            build_vocab(train0 + train1, args.vocab)

    vocab = Vocabulary(args.vocab, args.embedding, args.dim_emb)

    if args.train:
        model = get_model(args, vocab)
        losses_epochs = []
        
        for iter in range(no_of_epochs):
            random.shuffle(train0)
            random.shuffle(train1)
            batches, _, _ = get_batches(train0, train1, vocab.word2id,
            args.batch_size, noisy=True)
            random.shuffle(batches)
            print("Epoch: ", iter)
            logger.info("Epoch: "+str(iter))
            total_loss = 0
            for batch in batches:
                batch_input = batch["enc_inputs"]
                batch_input = torch.tensor(np.array(batch_input), device=device)
                loss = model.train(batch_input)
                total_loss += loss
            print("Loss: ", total_loss)
            print("---------\n")
            logger.info("Loss: " + str(total_loss))
            logger.info("---------\n")