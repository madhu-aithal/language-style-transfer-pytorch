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

# from model import Model
from autoencoder import Model
# from torch.utils.tensorboard import SummaryWriter

# writer = SummaryWriter('runs/cross-alignment')

device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

def get_filename(args):
    filename = str(datetime.now().strftime('model'+str(args.max_epochs)+'_%H_%M_%d_%m_%Y'))
    path = os.path.join(args.save_model_path, filename)
    return path

def init_logging(args):
    filename = str(datetime.now().strftime('app'+str(args.max_epochs)+'_%H_%M_%d_%m_%Y.log'))
    path = os.path.join(args.log_dir, filename)
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


def get_model(args, vocab):
    dim_hidden = args.dim_y+args.dim_z    
    print("vocab size: ", vocab.size)
    logger.info("vocab size: "+str(vocab.size))
    model = Model(vocab.size, dim_hidden, 
    dim_hidden, vocab.size, args.dropout_keep_prob, device, logger, vocab)
    return model

if __name__ == '__main__':
    args = load_arguments()
    print(args)
    no_of_epochs = args.max_epochs
    save_model_path = get_filename(args)
    
    logger = init_logging(args)

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
        
        for epoch in range(args.max_epochs):
            random.shuffle(train0)
            random.shuffle(train1)
            batches0, batches1, _, _ = get_batches(train0, train1, vocab.word2id,
            args.batch_size, noisy=True)

            random.shuffle(batches0)
            random.shuffle(batches1)
            print("Epoch: ", epoch)
            logger.info("Epoch: "+str(epoch))
            avg_loss = 0
            running_loss = 0
            i = 1

            for batch0, batch1 in zip(batches0, batches1):

                batch0_input = batch0["enc_inputs"]
                batch0_input = torch.tensor(batch0_input, device=device)
                
                batch1_input = batch1["enc_inputs"]                
                batch1_input = torch.tensor(batch1_input, device=device)
                
                loss0 = model.train_one_batch(batch0_input, batch0["lengths"], learning_rate=args.learning_rate)
                loss1 = model.train_one_batch(batch1_input, batch1["lengths"], learning_rate=args.learning_rate)
                
                avg_loss += (loss0+loss1)/2
                running_loss += avg_loss
                
                if i%5 == 0:
                    torch.save(model, save_model_path)

                i+=1
            print("Avg Loss: ", avg_loss)
            print("---------\n")
            logger.info("Avg Loss: " + str(avg_loss))
            logger.info("---------\n")

        test_input = ["this place was very good !"]
        test_input = [val.split() for val in test_input]

        test_input_processed = []
        for list_val in test_input:
            temp_list = []
            for val in list_val:
                temp_list.append(model.vocab.word2id[val])
            test_input_processed.append(temp_list)
        print(test_input_processed)
        print(test_input)
        test_input_tensor = torch.tensor(test_input_processed)
        print(model.predict(test_input_tensor.t()))
        # writer.add_graph(model)
        # writer.close()
        # model.train_max_epochs(args, train0, train1, vocab, no_of_epochs, save_model_path)
        
        # torch.save(model, save_model_path)

        