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

class Model():
    def __init__(self, input_size_enc, hidden_size_enc, 
    hidden_size_gen, output_size_gen, dropout_p, device, logger, lambda_val=1):

        self.input_size_enc = input_size_enc
        self.hidden_size_enc = hidden_size_enc
        self.hidden_size_gen = hidden_size_gen
        self.output_size_gen = output_size_gen
        self.dropout_p = dropout_p
        self.device = device

        self.generator = Generator(self.hidden_size_gen-1, self.hidden_size_gen, self.output_size_gen, self.dropout_p).to(self.device)
        # self.generator2 = Generator(self.hidden_size_gen, self.output_size_gen, self.dropout_p)
        self.encoder = Encoder(self.input_size_enc, self.hidden_size_enc, self.dropout_p).to(self.device)
        self.discriminator1 = Discriminator(self.hidden_size_gen, 1).to(self.device)
        self.discriminator2 = Discriminator(self.hidden_size_gen, 1).to(self.device)

        self.softmax = torch.nn.LogSoftmax(dim=0)
        self.EOS_token = 2
        self.GO_token = 1
        self.k = 5
        self.gamma = 0.001
        self.logger = logger
        self.lambda_val = lambda_val
        

    def get_latent_reps(self, input_data):

        if torch.cuda.is_available():
            input_data = input_data.to(device=self.device)
        
        # enc_optim.zero_grad()

        input_data = torch.t(input_data)
        input_length = input_data.shape[0]
        batch_size = input_data.shape[1]

        encoder_hidden = self.encoder.initHidden(device=self.device)
        encoder_hidden = encoder_hidden.repeat(1,batch_size,1)

        input_tensor = self.encoder.embedding(input_data)

        if torch.cuda.is_available():
            input_tensor = input_tensor.to(device=self.device)

        encoder_outputs = torch.zeros(input_length, 1, batch_size, self.encoder.hidden_size, device=self.device)

        for ei in range(input_length):
            encoder_output, encoder_hidden = self.encoder(
                input_tensor[ei,:,:], encoder_hidden)
            encoder_outputs[ei] = encoder_hidden 

        return encoder_outputs[-1]

    # Generate X using [y,z] where y is style attribute and z is the latent content obtained from the encoder
    def generate_x(self, hidden_vec, true_outputs, criterion, paddings, teacher_forcing=False):
        gen = self.generator
        
        batch_size = hidden_vec.shape[1]
        gen_input = torch.tensor([self.GO_token], device=self.device)
        gen_input = gen_input.repeat(batch_size)
        gen_hidden = hidden_vec

        gen_output = torch.zeros(self.output_size_gen, device=self.device)
        input_length = true_outputs.shape[1]
        loss = 0
        losses = torch.zeros(input_length, batch_size, device=self.device)
        gen_hid_states = torch.zeros(input_length, gen_hidden.shape[0], gen_hidden.shape[1], gen_hidden.shape[2], device=self.device)

        gen_input = gen_input.unsqueeze(0)
        gen_input = gen_input.unsqueeze(2)
        gen_input = self.encoder.embedding(gen_input).squeeze(2)

        for i in range(input_length):
            
            gen_output, gen_hidden = gen(
                gen_input, gen_hidden)
            # loss += criterion(gen_output, true_outputs[:,i])
            loss = criterion(gen_output, true_outputs[:,i])
            losses[i] = loss
            gen_hid_states[i] = gen_hidden

            if teacher_forcing == True:
                gen_input = true_outputs[:,i]
                gen_input = gen_input.unsqueeze(0)
                gen_input = gen_input.unsqueeze(2)
                gen_input = self.encoder.embedding(gen_input).squeeze(2)
                # print()
            else:
                # gen_input = torch.argmax(gen_output, dim=1)
                gen_input = self.softmax(gen_output/self.gamma)
                gen_input = gen_input.unsqueeze(0)
                # gen_input = gen_input.unsqueeze(2)
                # print(self.encoder.embedding.weight)
                gen_input = torch.matmul(gen_input, self.encoder.embedding.weight)
                # gen_input = self.encoder.embedding(gen_input).squeeze(2)
                # print()


                # topv, topi = gen_output.topk(1)
                # gen_input = topi.squeeze().detach()
            
            # if torch.argmax(gen_output) == self.EOS_token:
            #     # print("EOS token generated early - generator1")
            #     # self.logger.info("EOS token generated early - generator1")
            #     break
        padding_tensor = torch.zeros(input_length, batch_size, device=self.device)
        for idx, val in enumerate(paddings):
            padding_tensor[0:val+1, idx] = 1

        avg_loss = 0
        # if self.training == True:
        avg_loss = torch.mean(torch.mul(padding_tensor, losses))

        return gen_hid_states, avg_loss

    def train_one_batch(self, training_data, paddings, learning_rate, sentiment):
        # batch_loss_total = 0  
        
        # losses = torch.zeros((training_data.shape[0],1))

        criterion = nn.NLLLoss()
        indices = np.arange(training_data.shape[0])
        latent_z = self.get_latent_reps(training_data)
        latent_z_original = []
        latent_z_translated = []
        if sentiment == 1:
            latent_z_original = torch.cat((latent_z, torch.zeros([1,latent_z.shape[1],1],
                            dtype=torch.float,device=self.device)), 2)
            latent_z_translated = torch.cat((latent_z, torch.ones([1,latent_z.shape[1],1],
                            dtype=torch.float,device=self.device)), 2)
        else:
            latent_z_original = torch.cat((latent_z, torch.ones([1,latent_z.shape[1],1],
                            dtype=torch.float,device=self.device)), 2)
            latent_z_translated = torch.cat((latent_z, torch.zeros([1,latent_z.shape[1],1],
                            dtype=torch.float,device=self.device)), 2)
        # latent_z = torch.unsqueeze(latent_z, 0)
        hidden_states_original, loss_original = self.generate_x(latent_z_original, training_data, criterion, paddings, teacher_forcing=True)
        hidden_states_translated, loss_translated = self.generate_x(latent_z_translated, training_data, criterion, paddings, teacher_forcing=False)
        # hidden_states_translated, loss_translated = self.generate_x(latent_z, self.generator1, training_data, criterion, teacher_forcing=False)

        # avg_loss = 0
        # avg_loss = torch.mean(losses)
        avg_loss = torch.mean(loss_translated)
        # avg_loss.backward()

        # enc_optim.step()
        # gen1_optim.step()

        # return avg_loss
        return hidden_states_original, hidden_states_translated, avg_loss

    def train_max_epochs(self, args, train0, train1, vocab, no_of_epochs):
        losses_epochs = []
        enc_optim = optim.SGD(self.encoder.parameters(), lr=args.learning_rate)
        gen_optim = optim.SGD(self.generator.parameters(), lr=args.learning_rate)
        discrim1_optim = optim.SGD(self.discriminator1.parameters(), lr=args.learning_rate)
        discrim2_optim = optim.SGD(self.discriminator2.parameters(), lr=args.learning_rate)

        for iter in range(no_of_epochs):

            random.shuffle(train0)
            random.shuffle(train1)
            batches0, batches1, _, _ = get_batches(train0, train1, vocab.word2id,
            args.batch_size, noisy=True)

            random.shuffle(batches0)
            random.shuffle(batches1)
            print("Epoch: ", iter)
            self.logger.info("Epoch: "+str(iter))
            avg_loss = 0

            losses_enc_gen = []
            losses_adv1 = []
            losses_adv2 = []
            rec_losses = []

            for batch0, batch1 in zip(batches0, batches1):

                enc_optim.zero_grad()
                gen_optim.zero_grad()
                discrim1_optim.zero_grad()
                discrim2_optim.zero_grad()

                batch0_input = batch0["enc_inputs"]
                batch0_input = torch.tensor(batch0_input, device=self.device)
                batch0_input = torch.cat((batch0_input, torch.zeros([batch0_input.shape[0],1],dtype=torch.long, device=self.device)), 1)

                batch1_input = batch1["enc_inputs"]                
                batch1_input = torch.tensor(batch1_input, device=self.device)
                batch1_input = torch.cat((batch1_input, torch.ones([batch1_input.shape[0],1],dtype=torch.long,device=self.device)), 1)
                
                h1, h1_tilde, loss0 = self.train_one_batch(batch0_input, batch0["lengths"], args.learning_rate, 1)
                h2, h2_tilde, loss1 = self.train_one_batch(batch1_input, batch1["lengths"], args.learning_rate, 2)
                
                adv1_output = self.discriminator1(h1)
                adv1_output_tilde = self.discriminator1(h2_tilde)

                adv2_output = self.discriminator1(h2)
                adv2_output_tilde = self.discriminator1(h1_tilde) 

                loss_adv1 = -torch.mean(torch.log(adv1_output))-torch.mean(torch.log(1-adv1_output_tilde))                           
                loss_adv2 = -torch.mean(torch.log(adv2_output))-torch.mean(torch.log(1-adv2_output_tilde))               
     
                losses_adv1.append(loss_adv1)
                losses_adv2.append(loss_adv2)

                loss_reconstruction = (loss0+loss1)/2
                rec_losses.append(loss_reconstruction)

                loss_enc_gen = loss_reconstruction - self.lambda_val*(loss_adv1+loss_adv2)
                losses_enc_gen.append(loss_enc_gen)

                loss_enc_gen.backward(retain_graph=True)
                loss_adv1.backward(retain_graph=True)
                loss_adv2.backward()

                enc_optim.step()
                gen_optim.step()

                discrim1_optim.step()
                discrim2_optim.step()
            print("Avg Reconstruction Loss: ", torch.mean(torch.tensor(rec_losses)))
            print("Avg Loss of Encoder-Generator: ", torch.mean(torch.tensor(losses_enc_gen)))
            print("Avg Loss of D1: ", torch.mean(torch.tensor(losses_adv1)))
            print("Avg Loss of D2: ", torch.mean(torch.tensor(losses_adv2)))
            print("---------\n")
            # self.logger.info("Avg Loss of Encoder-Generator: ", torch.mean(torch.tensor(losses_enc_gen)))
            # self.logger.info("Avg Loss of D1: ", torch.mean(torch.tensor(losses_adv1)))
            # self.logger.info("Avg Loss of D2: ", torch.mean(torch.tensor(losses_adv2)))
            # self.logger.info("---------\n")
            # self.logger.info("Avg Loss of Encoder-Generator: ", torch.mean(losses_enc_gen))
            # self.logger.info("Avg Loss of D1: ", torch.mean(losses_adv1))
            # self.logger.info("Avg Loss of D2: ", torch.mean(losses_adv2))
            # self.logger.info("---------\n")