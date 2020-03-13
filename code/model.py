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

DISCRIMINATOR_PATH = "models/model_0.0005_60K_discriminator_1_epochs"
AUTOENCODER_PATH = "models/model_0.0005_60K_autoencoder_50_epochs"

class Model(nn.Module):
    def __init__(self, args, input_size_enc, embedding_size_enc, hidden_size_enc, 
    hidden_size_gen, output_size_gen, dropout_p, device, logger, vocab, lambda_val=1):
        super(Model, self).__init__()
        self.input_size_enc = input_size_enc
        self.hidden_size_enc = hidden_size_enc
        self.embedding_size_enc = embedding_size_enc
        self.hidden_size_gen = hidden_size_gen
        self.output_size_gen = output_size_gen
        self.dropout_p = dropout_p
        self.device = device
        self.vocab = vocab
        self.dropout = nn.Dropout(self.dropout_p)
        self.args = args

        # Loading the components of the pretrained model and assigning it to the current model
        autoencoder_model = torch.load(AUTOENCODER_PATH)
        discriminator_model = torch.load(DISCRIMINATOR_PATH)
        self.generator = autoencoder_model.generator.to(self.device)
        self.encoder = autoencoder_model.encoder.to(self.device)
        self.discriminator1 = discriminator_model.discriminator1.to(self.device)
        self.discriminator2 = discriminator_model.discriminator2.to(self.device)

        # self.generator = Generator(self.embedding_size_enc, self.hidden_size_gen, self.output_size_gen, self.dropout_p).to(self.device)
        # self.encoder = Encoder(self.input_size_enc, self.embedding_size_enc, self.hidden_size_enc, self.dropout_p).to(self.device)
        # self.discriminator1 = Discriminator(args, self.device, self.hidden_size_gen, 1).to(self.device)
        # self.discriminator2 = Discriminator(args, self.device, self.hidden_size_gen, 1).to(self.device)

        self.softmax = torch.nn.Softmax(dim=1)
        self.EOS_token = 2
        self.GO_token = 1
        self.k = 5
        self.gamma = 0.001
        self.logger = logger
        self.lambda_val = lambda_val
        self.beta1, self.beta2 = 0.5, 0.999
        self.grad_clip = 30.0
        

    # Encoder
    # Takes x and finds the latent representation z
    def get_latent_reps(self, input_data):

        if torch.cuda.is_available():
            input_data = input_data.to(device=self.device)
        
        input_length = input_data.shape[0]
        batch_size = input_data.shape[1]

        encoder_hidden = self.encoder.initHidden(device=self.device)
        encoder_hidden = encoder_hidden.repeat(1,batch_size,1)

        input_tensor = self.encoder.embedding(input_data)
        input_tensor = self.dropout(input_tensor)

        if torch.cuda.is_available():
            input_tensor = input_tensor.to(device=self.device)

        for ei in range(input_length):
            encoder_output, encoder_hidden = self.encoder(
                input_tensor[ei,:,:], encoder_hidden)

        return encoder_hidden

    def gumbel_softmax(self, logits, eps=1e-20):
        U = torch.rand_like(logits)
        G = -torch.log(-torch.log(U + eps) + eps)
        result = self.softmax((logits + G) / self.gamma)
        return result

    # Greedy search prediction used for testing some sample inputs
    def predict_greedy_search(self, input_data, target_sentiment):
        latent_z = self.get_latent_reps(input_data)
        input_length = input_data.size()[0]
        if target_sentiment == 1:
            latent_z = torch.cat((latent_z, torch.ones([1,latent_z.shape[1],self.args.dim_y],
                            dtype=torch.float,device=self.device)), dim=2)
        else:
            latent_z = torch.cat((latent_z, torch.zeros([1,latent_z.shape[1],self.args.dim_y],
                            dtype=torch.float,device=self.device)), dim=2)
        
        gen_input = torch.tensor([self.GO_token], device=self.device)
        gen_hidden = latent_z

        outputs = []
        gen_output = torch.zeros(self.output_size_gen, device=self.device)
        
        count = 0
        while torch.argmax(gen_output) != self.EOS_token:
            if count >= input_length*2:
                break            
            gen_input = gen_input.unsqueeze(0)
            gen_input = gen_input.unsqueeze(2)
            gen_input = self.encoder.embedding(gen_input).squeeze(2)
            gen_input = self.dropout(gen_input)

            gen_output, gen_hidden = self.generator(
                gen_input, gen_hidden)
            gen_input = torch.argmax(gen_output, dim=1)

            outputs.append(self.vocab.id2word[gen_input])
            count += 1

        return outputs

    # Beam search prediction used for testing some sample inputs
    def predict_beam_search(self, input_data, target_sentiment, k):
        latent_z = self.get_latent_reps(input_data)
        input_length = input_data.size()[0]
        if target_sentiment == 1:
            latent_z = torch.cat((latent_z, torch.ones([1,latent_z.shape[1],self.args.dim_y],
                            dtype=torch.float,device=self.device)), dim=2)
        else:
            latent_z = torch.cat((latent_z, torch.zeros([1,latent_z.shape[1],self.args.dim_y],
                            dtype=torch.float,device=self.device)), dim=2)
        
        gen_input = torch.tensor([self.GO_token], device=self.device)
        gen_hidden = latent_z
        outputs = []
        gen_output = torch.zeros(self.output_size_gen, device=self.device)
        result = []
        count = 0
        softmax = torch.nn.Softmax(dim=1)
        while torch.argmax(gen_output) != self.EOS_token:
            if len(result) == k or count >= input_length*2:
                break        
            if count == 0:
                gen_input = gen_input.unsqueeze(0)
                gen_input = gen_input.unsqueeze(2)
                gen_input = self.encoder.embedding(gen_input).squeeze(2)
                gen_input = self.dropout(gen_input)

                gen_output, gen_hidden = self.generator(
                    gen_input, gen_hidden)
                gen_output = softmax(gen_output)
                topv, topi = gen_output.topk(k)
                for i in range(topv.size()[1]):                
                    outputs.append({
                        "sequence": [topi[:,i].item()],
                        "score": np.log(topv[:,i].item())
                    })
                    if topi[:,i].item() == self.EOS_token:
                        result.append(outputs[-1])
                        del outputs[-1]
                
            else:
                outputs_old = outputs.copy()
                outputs = []
                for val in outputs_old:
                    gen_input = torch.tensor(val["sequence"][-1], device=self.device).view(1,1,1)
                    gen_input = self.encoder.embedding(gen_input).squeeze(2)
                    gen_input = self.dropout(gen_input)

                    gen_output, gen_hidden = self.generator(
                        gen_input, gen_hidden)
                    
                    topv, topi = gen_output.topk(k)

                    for i in range(topv.size()[1]):                          
                        outputs.append({
                            "sequence": val["sequence"] + [topi[:,i].item()],
                            "score": np.log(val["score"])+np.log(topv[:,i].item())
                        })
                        if topi[:,i].item() == self.EOS_token:
                            result.append(outputs[-1])
                            del outputs[-1]
                            
                outputs = sorted(outputs, key = lambda i: i['score'], reverse = True)
                outputs = outputs[:k]
            count += 1
        result = sorted(result, key = lambda i: i['score'], reverse = True)
        
        for output in result:
            output["sentence"]=[self.vocab.id2word[val] for val in output["sequence"]]
        return result


    # Generate X using [y,z] where y is style attribute and z is the latent content obtained from the encoder
    def generate_x(self, hidden_vec, true_outputs, criterion, teacher_forcing=False):
        gen = self.generator
        
        batch_size = hidden_vec.shape[1]
        gen_input = torch.tensor([self.GO_token], device=self.device)
        gen_input = gen_input.repeat(batch_size)
        gen_hidden = hidden_vec

        gen_output = torch.zeros(self.output_size_gen, device=self.device)
        input_length = true_outputs.shape[0]
        loss = 0
        losses = torch.zeros(input_length, batch_size, device=self.device)
        gen_hid_states = torch.zeros(input_length, gen_hidden.shape[0], gen_hidden.shape[1], gen_hidden.shape[2], device=self.device)

        gen_input = gen_input.unsqueeze(0)
        gen_input = gen_input.unsqueeze(2)
        gen_input_new = self.encoder.embedding(gen_input).squeeze(2)
        gen_input = gen_input_new
        if False in (gen_input_new==gen_input_new):
            print() 
        # self.logger.info("true_outputs: "+str(true_outputs))
        for i in range(input_length):            
            gen_output, gen_hidden = gen(
                gen_input, gen_hidden)            
            # assert not False in (gen_output==gen_output)
            # if False in (gen_output==gen_output):
            #     print()
            gen_output = gen_output+1e-8
            # self.logger.info("i: "+str(i))
            # self.logger.info("gen output "+str(gen_output[0]))
            
            gen_hid_states[i] = gen_hidden

            if teacher_forcing == True:
                loss = criterion(gen_output, true_outputs[i,:])
                losses[i] = loss
                gen_input = true_outputs[i,:]
                gen_input = gen_input.unsqueeze(0)
                gen_input = gen_input.unsqueeze(2)
                gen_input = self.encoder.embedding(gen_input).squeeze(2)
            else:                
                gen_input = self.gumbel_softmax(gen_output)
                gen_input = gen_input.unsqueeze(0)
                gen_input = torch.matmul(gen_input, self.encoder.embedding.weight)
        
        avg_loss = 0
        avg_loss = torch.mean(losses)
        if False in (losses==losses):
            print() 
        # self.logger.info("avg loss: "+str(avg_loss))
        return gen_hid_states, avg_loss

    # Train one batch of positive or negative samples
    # Returns h1 and h1~ or h2 and h2~
    def train_one_batch(self, training_data, target, sentiment):
       
        criterion = nn.CrossEntropyLoss(ignore_index=self.vocab.word2id['<pad>'], reduction='mean', size_average=True)
        latent_z = self.get_latent_reps(training_data)
        latent_z_original = []
        latent_z_translated = []
        if sentiment == 0:
            latent_z_original = torch.cat((latent_z, torch.zeros([1,latent_z.shape[1],self.args.dim_y],
                            dtype=torch.float,device=self.device)), 2)
            latent_z_translated = torch.cat((latent_z, torch.ones([1,latent_z.shape[1],self.args.dim_y],
                            dtype=torch.float,device=self.device)), 2)
        else:
            latent_z_original = torch.cat((latent_z, torch.ones([1,latent_z.shape[1],self.args.dim_y],
                            dtype=torch.float,device=self.device)), 2)
            latent_z_translated = torch.cat((latent_z, torch.zeros([1,latent_z.shape[1],self.args.dim_y],
                            dtype=torch.float,device=self.device)), 2)
        hidden_states_original, loss_original = self.generate_x(latent_z_original, target, criterion, teacher_forcing=True)
        hidden_states_translated, _ = self.generate_x(latent_z_translated, target, criterion, teacher_forcing=False)

        avg_loss = torch.mean(loss_original)

        return hidden_states_original, hidden_states_translated, avg_loss


    # Trains one set of positive and negative samples batch
    def train_util(self, batch0, batch1, epoch, writer):
        correct_count = 0
        batch0_input = batch0["enc_inputs"]
        batch0_input = torch.tensor(batch0_input, device=self.device)
        batch0_input = batch0_input.t()
        target_batch0 = torch.tensor(batch0["targets"], device=self.device).t()
        # batch0_input = torch.cat((batch0_input, torch.zeros([batch0_input.shape[0],1],dtype=torch.long, device=self.device)), dim=1)

        batch1_input = batch1["enc_inputs"]                
        batch1_input = torch.tensor(batch1_input, device=self.device)
        batch1_input = batch1_input.t()
        target_batch1 = torch.tensor(batch1["targets"], device=self.device).t()
        # batch1_input = torch.cat((batch1_input, torch.ones([batch1_input.shape[0],1],dtype=torch.long,device=self.device)), dim=1)
        
        # 0 represents that the sample/batch has negative sentiment
        # 1 represents that the sample/batch has positive sentiment
        h1, h1_tilde, loss0 = self.train_one_batch(batch0_input, target_batch0, 0)
        h2, h2_tilde, loss1 = self.train_one_batch(batch1_input, target_batch1, 1)
        
        # Permuting so that the input for discriminator is in the 
        # format (Batch,Channels,H,W) (changed from (H,Channels,Batch,W))
        adv1_output = self.discriminator1(h1.permute(2,1,0,3))        
        adv1_output_tilde = self.discriminator1(h2_tilde.permute(2,1,0,3))

        adv2_output = self.discriminator2(h2.permute(2,1,0,3))
        adv2_output_tilde = self.discriminator2(h1_tilde.permute(2,1,0,3)) 

        correct_count += torch.sum((adv1_output>0.5) == True)
        correct_count += torch.sum((adv2_output>0.5) == True)
        correct_count += torch.sum((adv1_output_tilde<0.5) == True)
        correct_count += torch.sum((adv2_output_tilde<0.5) == True)
        correct_count = correct_count.item()
        correct_count /= (adv1_output.size()[0]+adv1_output_tilde.size()[0]+adv2_output.size()[0]+adv2_output_tilde.size()[0])

        writer.add_scalars("Adv_Outputs", {
            "adv1_output": adv1_output[0].item(),
            "adv1_output_tilde": adv1_output_tilde[0].item(),
            "adv2_output": adv2_output[0].item(),
            "adv2_output_tilde": adv2_output_tilde[0].item()
        }, epoch)

        loss_adv1 = -torch.mean(torch.log(adv1_output))-torch.mean(torch.log(1-adv1_output_tilde))                           
        loss_adv2 = -torch.mean(torch.log(adv2_output))-torch.mean(torch.log(1-adv2_output_tilde))    

        loss_reconstruction = (loss0+loss1)/2
        loss_enc_gen = loss_reconstruction - self.lambda_val*(loss_adv1+loss_adv2)

        return loss_adv1, loss_adv2, loss_enc_gen, loss_reconstruction, correct_count

    def train_max_epochs(self, args, train0, train1, dev0, dev1, vocab, no_of_epochs, writer, save_epochs_flag=False, 
            save_epochs=20, save_batch_flag=False, save_batch=5):
        self.train()

        enc_optim = optim.AdamW(self.encoder.parameters(), lr=args.learning_rate, betas=(self.beta1, self.beta2))
        gen_optim = optim.AdamW(self.generator.parameters(), lr=args.learning_rate, betas=(self.beta1, self.beta2))
        discrim1_optim = optim.AdamW(self.discriminator1.parameters(), lr=args.learning_rate, betas=(self.beta1, self.beta2))
        discrim2_optim = optim.AdamW(self.discriminator2.parameters(), lr=args.learning_rate, betas=(self.beta1, self.beta2))
        save_model_path = os.path.join(args.save_model_path, get_filename(args, "model"))

        flag = True
        pretrain_flag = False
        autoencoder_train_flag = False

        # prev_rec_avgloss = math.inf
        # prev_disc_acc = 0
        for epoch in range(no_of_epochs):

            random.shuffle(train0)
            random.shuffle(train1)
            batches0, batches1, _1, _2 = get_batches(train0, train1, vocab.word2id,
            args.batch_size, noisy=True)
            
            random.shuffle(batches0)
            random.shuffle(batches1)
            print("Epoch: ", epoch)
            self.logger.info("Epoch: "+str(epoch))

            losses_enc_gen = []
            losses_adv1 = []
            losses_adv2 = []
            rec_losses = []

            losses_enc_gen_dev = []
            losses_adv1_dev = []
            losses_adv2_dev = []
            rec_losses_dev = []
            i = 0
            flag = True
            disc_tot_accuracy = 0
            for batch0, batch1 in zip(batches0, batches1):
                i += 1
                # print(i)
                enc_optim.zero_grad()
                gen_optim.zero_grad()
                discrim1_optim.zero_grad()
                discrim2_optim.zero_grad()

                loss_adv1, loss_adv2, loss_enc_gen, loss_reconstruction, disc_batch_acc = self.train_util(batch0, batch1, epoch, writer)
                disc_tot_accuracy += disc_batch_acc

                if pretrain_flag == True:
                    if autoencoder_train_flag == True:
                        # Train only autoencoder part
                        # Doing backprop on loss_reconstruction (not loss_enc_gen) because I think we just need to 
                        # train the autoencoder to reconstruct the input sentence, as part of the pretraining. 
                        # As a result, the autoencoder will be good at its task of reconstructing the input sentence
                        loss_reconstruction.backward()
                        torch.nn.utils.clip_grad_value_(self.encoder.parameters(), self.grad_clip)
                        torch.nn.utils.clip_grad_value_(self.generator.parameters(), self.grad_clip)
                        enc_optim.step()
                        gen_optim.step()
                    else:
                        # Train only discriminators
                        loss_adv1.backward(retain_graph=True)
                        loss_adv2.backward()
                        torch.nn.utils.clip_grad_value_(self.discriminator1.parameters(), self.grad_clip)
                        torch.nn.utils.clip_grad_value_(self.discriminator2.parameters(), self.grad_clip)
                        discrim1_optim.step()
                        discrim2_optim.step()
                else:                        
                    if flag == True:
                        loss_enc_gen.backward()
                        torch.nn.utils.clip_grad_value_(self.encoder.parameters(), self.grad_clip)
                        torch.nn.utils.clip_grad_value_(self.generator.parameters(), self.grad_clip)
                        enc_optim.step()
                        gen_optim.step()
                    else:
                        loss_adv1.backward(retain_graph=True)
                        loss_adv2.backward()
                        torch.nn.utils.clip_grad_value_(self.discriminator1.parameters(), self.grad_clip)
                        torch.nn.utils.clip_grad_value_(self.discriminator2.parameters(), self.grad_clip)
                        discrim1_optim.step()
                        discrim2_optim.step()
                    flag = not flag

                losses_enc_gen.append(loss_enc_gen.detach())
                losses_adv1.append(loss_adv1.detach())
                losses_adv2.append(loss_adv2.detach())
                rec_losses.append(loss_reconstruction.detach())

            if self.args.dev:
                
                batches0, batches1, _1, _2 = get_batches(dev0, dev1, vocab.word2id,
                    args.batch_size, noisy=True)

                random.shuffle(batches0)
                random.shuffle(batches1)

                for batch0, batch1 in zip(batches0, batches1):

                    loss_adv1, loss_adv2, loss_enc_gen, loss_reconstruction, disc_batch_dev_acc = self.train_util(batch0, batch1)

                    losses_adv1_dev.append(loss_adv1)
                    losses_adv2_dev.append(loss_adv2)

                    losses_enc_gen_dev.append(loss_enc_gen)
                    rec_losses_dev.append(loss_reconstruction)

            if save_epochs_flag == True and epoch%save_epochs == save_epochs-1:
                torch.save(self, save_model_path)

            disc_tot_accuracy = 1.0*disc_tot_accuracy/i

            print("Avg Reconstruction Loss: ", torch.mean(torch.tensor(rec_losses)))
            print("Avg Loss of Encoder-Generator: ", torch.mean(torch.tensor(losses_enc_gen)))
            print("Avg Loss of D1: ", torch.mean(torch.tensor(losses_adv1)))
            print("Avg Loss of D2: ", torch.mean(torch.tensor(losses_adv2)))

            self.logger.info("Avg Reconstruction Loss: " + str(torch.mean(torch.tensor(rec_losses))))
            self.logger.info("Avg Loss of Encoder-Generator: " + str(torch.mean(torch.tensor(losses_enc_gen))))
            self.logger.info("Avg Loss of D1: " + str(torch.mean(torch.tensor(losses_adv1))))
            self.logger.info("Avg Loss of D2: " + str(torch.mean(torch.tensor(losses_adv2))))
            self.logger.info("Discriminator accuracy: " + str(disc_tot_accuracy))

            writer.add_scalar("discriminator_acc", disc_tot_accuracy, epoch)
            writer.add_scalars('All_losses', {
                'recon-loss': torch.mean(torch.tensor(rec_losses)),
                'enc-gen': torch.mean(torch.tensor(losses_enc_gen)),
                'D1': torch.mean(torch.tensor(losses_adv1)),
                'D2': torch.mean(torch.tensor(losses_adv2))
            }, epoch)

            if pretrain_flag == True:
                if torch.mean(torch.tensor(rec_losses)) < 0.1:
                    break

                if disc_tot_accuracy >= 0.9:
                    break

            if self.args.dev:
                print("\nDev loss")
                print("Avg Reconstruction Loss: ", torch.mean(torch.tensor(rec_losses_dev)))
                print("Avg Loss of Encoder-Generator: ", torch.mean(torch.tensor(losses_enc_gen_dev)))
                print("Avg Loss of D1: ", torch.mean(torch.tensor(losses_adv1_dev)))
                print("Avg Loss of D2: ", torch.mean(torch.tensor(losses_adv2_dev)))

                self.logger.info("\nDev loss")
                self.logger.info("Avg Reconstruction Loss: " + str(torch.mean(torch.tensor(rec_losses_dev))))
                self.logger.info("Avg Loss of Encoder-Generator: " + str(torch.mean(torch.tensor(losses_enc_gen_dev))))
                self.logger.info("Avg Loss of D1: " + str(torch.mean(torch.tensor(losses_adv1_dev))))
                self.logger.info("Avg Loss of D2: " + str(torch.mean(torch.tensor(losses_adv2_dev))))

                writer.add_scalars('All_losses_dev', {
                    'recon-loss': torch.mean(torch.tensor(rec_losses)),
                    'enc-gen': torch.mean(torch.tensor(losses_enc_gen_dev)),
                    'D1': torch.mean(torch.tensor(losses_adv1_dev)),
                    'D2': torch.mean(torch.tensor(losses_adv2_dev))
                }, epoch)
            
            print("---------\n")
            self.logger.info("---------\n")
             
        torch.save(self, save_model_path)