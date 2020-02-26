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
from discriminator import Discriminator


class Model(nn.Module):
    def __init__(self, input_size_enc, embedding_size_enc, hidden_size_enc, 
    hidden_size_gen, output_size_gen, dropout_p, device, logger, vocab):
        super(Model, self).__init__()
        self.input_size_enc = input_size_enc
        self.embedding_size_enc = embedding_size_enc
        self.hidden_size_enc = hidden_size_enc
        self.hidden_size_gen = hidden_size_gen
        self.output_size_gen = output_size_gen
        self.dropout_p = dropout_p
        self.device = device
        self.dropout = nn.Dropout(0.5)

        self.generator1 = Generator(self.embedding_size_enc, self.hidden_size_gen, self.output_size_gen, self.dropout_p).to(self.device)
        # self.generator2 = Generator(self.embedding_size_enc, self.hidden_size_gen, self.output_size_gen, self.dropout_p)
        self.encoder = Encoder(self.input_size_enc, self.embedding_size_enc, self.hidden_size_enc, self.dropout_p).to(self.device)
        self.discriminator1 = Discriminator(self.hidden_size_gen, 2)
        self.discriminator2 = Discriminator(self.hidden_size_gen, 2)

        self.softmax = torch.nn.LogSoftmax(self.output_size_gen)
        self.EOS_token = 2
        self.GO_token = 1
        self.k = 5
        self.gamma = 0.001
        self.logger = logger
        self.vocab = vocab
        

    def get_latent_reps(self, input_data, enc_optim):

        if torch.cuda.is_available():
            input_data = input_data.to(device=self.device)
        
        #enc_optim.zero_grad()

        input_data = torch.t(input_data)
        input_length = input_data.shape[0]
        batch_size = input_data.shape[1]

        encoder_hidden = self.encoder.initHidden(device=self.device)
        encoder_hidden = encoder_hidden.repeat(1,batch_size,1)

        input_tensor = self.encoder.embedding(input_data)
        input_tensor = self.dropout(input_tensor)

        if torch.cuda.is_available():
            input_tensor = input_tensor.to(device=self.device)

        encoder_outputs = torch.zeros(input_length, 1, batch_size, self.encoder.hidden_size, device=self.device)

        for ei in range(input_length):
            encoder_output, encoder_hidden = self.encoder(
                input_tensor[ei,:,:], encoder_hidden)
            encoder_outputs[ei] = encoder_hidden 

        return encoder_outputs[-1]

    def predict(self, input_data):
        if torch.cuda.is_available():
            input_data = input_data.to(device=self.device)

        input_length = input_data.shape[0]
        input_tensor = self.encoder.embedding(input_data)
        batch_size = input_data.shape[1]
        encoder_hidden = self.encoder.initHidden(device=self.device)
        encoder_hidden = encoder_hidden.repeat(1,batch_size,1)

        encoder_outputs = torch.zeros(input_length, 1, batch_size, self.encoder.hidden_size, device=self.device)

        for ei in range(input_length):
            encoder_output, encoder_hidden = self.encoder(
                input_tensor[ei,:,:], encoder_hidden)
            encoder_outputs[ei] = encoder_hidden 

        latent_z = encoder_outputs[-1,:,:]
        
        gen_input = torch.tensor([self.GO_token], device=self.device)
        gen_hidden = latent_z

        outputs = []
        gen_output = torch.zeros(self.output_size_gen, device=self.device)
        count = 0
        while torch.argmax(gen_output) != self.EOS_token:
            count += 1
            gen_input = gen_input.unsqueeze(0)
            gen_input = gen_input.unsqueeze(2)
            gen_input = self.encoder.embedding(gen_input).squeeze(2)
            gen_output, gen_hidden = self.generator1(
                gen_input, gen_hidden)
            gen_input = torch.argmax(gen_output, dim=1)
            outputs.append(self.vocab.id2word[gen_input])
            # print(self.vocab.id2word[gen_input])
            if count > input_length:
                break
        
        return outputs
        

    # Generate X using [y,z] where y is style attribute and z is the latent content obtained from the encoder
    def generate_x(self, hidden_vec, gen, true_outputs, criterion, paddings, teacher_forcing=False):
        gen_hid_states = []
        batch_size = hidden_vec.shape[1]
        gen_input = torch.tensor([self.GO_token], device=self.device)
        gen_input = gen_input.repeat(batch_size)
        gen_hidden = hidden_vec

        gen_output = torch.zeros(self.output_size_gen, device=self.device)
        input_length = true_outputs.shape[1]
        loss = 0
        losses = torch.zeros(input_length, batch_size)
        # pred_matrix = 
        for i in range(input_length):
            gen_input = gen_input.unsqueeze(0)
            gen_input = gen_input.unsqueeze(2)
            gen_input = self.encoder.embedding(gen_input).squeeze(2)
            gen_input = self.dropout(gen_input)
            gen_output, gen_hidden = gen(
                gen_input, gen_hidden)

            # if self.training == True:
            loss = criterion(gen_output, true_outputs[:,i].view(-1))
            losses[i] = loss
            gen_hid_states.append(gen_hidden)

            # if self.training == True:
            if teacher_forcing == True:
                # print("true output: ", true_outputs.size())
                gen_input = true_outputs[:,i]
            else:
                gen_input = torch.argmax(gen_output, dim=1)
            # else:
            #     gen_input = torch.argmax(gen_output, dim=1)
                # topv, topi = gen_output.topk(1)
                # gen_input = topi.squeeze().detach()

            # if torch.argmax(gen_output) == self.EOS_token:
            #     # print("EOS token generated early - generator1")
            #     # self.logger.info("EOS token generated early - generator1")
            #     break
        # loss = criterion(, ignore_index=mask)


        padding_tensor = torch.zeros(input_length, batch_size)
        for idx, val in enumerate(paddings):
            padding_tensor[0:val+1, idx] = 1

        avg_loss = 0
        # if self.training == True:
        temp = torch.sum(torch.mul(padding_tensor, losses), dim=0).unsqueeze(0)
        temp2 = torch.tensor(paddings).unsqueeze(0)
        avg_loss = torch.mean(temp/temp2)
        # loss = loss/input_length
        return gen_hid_states, avg_loss 

    def train_one_batch(self, training_data, paddings, learning_rate=0.01):
        # batch_loss_total = 0  
        self.train()
        # self.val()
        # with torch.no_grad():
        enc_optim = optim.AdamW(self.parameters(), lr=learning_rate)
        # gen1_optim = optim.AdamW(self.generator1.parameters(), lr=learning_rate)
        # gen2_optim = optim.AdamW(self.generator2.parameters(), lr=learning_rate)
        # discrim1_optim = optim.AdamW(self.discriminator1.parameters(), lr=learning_rate)
        # discrim2_optim = optim.AdamW(self.discriminator2.parameters(), lr=learning_rate)

        enc_optim.zero_grad()
        # gen1_optim.zero_grad()
        # gen2_optim.zero_grad()
        # discrim1_optim.zero_grad()
        # discrim2_optim.zero_grad()
        
        # losses = torch.zeros((training_data.shape[0],1))

        criterion = nn.CrossEntropyLoss(reduce=False)
        indices = np.arange(training_data.shape[0])
        latent_z = self.get_latent_reps(training_data, enc_optim)
        # latent_z = torch.unsqueeze(latent_z, 0)
        # hidden_states_original, loss_original = self.generate_x(latent_z, self.generator1, training_data, criterion, teacher_forcing=True)
        # hidden_states_translated, loss_translated = self.generate_x(latent_z, self.generator2, training_data, criterion, teacher_forcing=False)
        hidden_states_translated, loss_translated = self.generate_x(latent_z, self.generator1, training_data, criterion, paddings, teacher_forcing=True)


        avg_loss = 0
        # avg_loss = torch.mean(losses)
        avg_loss = torch.mean(loss_translated)
        avg_loss.backward()

        enc_optim.step()
        #gen1_optim.step()

        return avg_loss

# Not using train_max_epochs() anymore. It is defined in autoencoder_main.py
    def train_max_epochs(self, args, train0, train1, vocab, no_of_epochs, save_model_path):
        losses_epochs = []
        
        for epoch in range(no_of_epochs):
            random.shuffle(train0)
            random.shuffle(train1)
            batches0, batches1, _, _ = get_batches(train0, train1, vocab.word2id,
            args.batch_size, noisy=True)

            random.shuffle(batches0)
            random.shuffle(batches1)
            print("Epoch: ", epoch)
            self.logger.info("Epoch: "+str(epoch))
            avg_loss = 0
            running_loss = 0
            i = 1

            for batch0, batch1 in zip(batches0, batches1):
               
                batch0_input = batch0["enc_inputs"]
                batch0_input = torch.tensor(batch0_input, device=self.device)
                # batch0_input = torch.nn.utils.rnn.pack_padded_sequence(batch0_input, batch0["lengths"])
                # batch0_input = torch.cat((batch0_input, torch.zeros([batch0_input.shape[0],1],dtype=torch.long, device=self.device)), 1)

                batch1_input = batch1["enc_inputs"]                
                batch1_input = torch.tensor(batch1_input, device=self.device)
                # batch1_input = torch.nn.utils.rnn.pack_padded_sequence(batch1_input, batch0["lengths"])
                # batch1_input = torch.cat((batch1_input, torch.ones([batch1_input.shape[0],1],dtype=torch.long,device=self.device)), 1)
                
                loss0 = self.train_one_batch(batch0_input, batch0["lengths"], learning_rate=args.learning_rate)
                loss1 = self.train_one_batch(batch1_input, batch1["lengths"], learning_rate=args.learning_rate)
                
                avg_loss += (loss0+loss1)/2
                running_loss += avg_loss
                # avg_loss.backward()
                
                # enc_optim.step()
                # gen1_optim.step()

                # if i % 20 == 19: 
                #     # ...log the running loss
                #     writer.add_scalar('training loss',
                #                     running_loss / 1000,
                #                     epoch * (len(batch0)+len(batch1)) + i)

                #     # ...log a Matplotlib Figure showing the model's predictions on a
                #     # random mini-batch
                #     # writer.add_figure('predictions vs. actuals',
                #     #                 plot_classes_preds(net, inputs, labels),
                #     #                 global_step=epoch * len(trainloader) + i)
                #     running_loss = 0.0
                
                if i%5 == 0:
                    torch.save(model, save_model_path)

                i+=1
            print("Avg Loss: ", avg_loss)
            print("---------\n")
            self.logger.info("Avg Loss: " + str(avg_loss))
            self.logger.info("---------\n")