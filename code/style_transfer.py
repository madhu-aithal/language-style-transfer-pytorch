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
from utils import *

import numpy as np
import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F

from model import Model
from torch.utils.tensorboard import SummaryWriter


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


def get_model(args, vocab, logger):

    device = torch.device("cuda:"+str(args.cuda_device) if torch.cuda.is_available() else "cpu")
    print("vocab size: ", vocab.size)

    logger.info("vocab size: "+str(vocab.size))
    model = Model(args, vocab.size, args.dim_emb, args.dim_z, 
    args.dim_z+args.dim_y, vocab.size, args.dropout_keep_prob, device, logger, vocab)
    return model

if __name__ == '__main__':
    args = load_arguments()
    logger = init_logging(args)
    
    print("args: ", args)
    logger.info("args: "+str(args))
    no_of_epochs = args.max_epochs

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
    dev0 = []
    dev1 = []
    if args.dev:
        dev0 = load_sent(args.dev + '.0')
        dev1 = load_sent(args.dev + '.1')

    if args.train:
        summ_filename = 'runs/cross-alignment/'+get_filename(args, "summary")
        writer = SummaryWriter(summ_filename)

        model = get_model(args, vocab, logger)
        model.train_max_epochs(args, train0, train1, dev0, dev1, vocab, no_of_epochs, writer)

                
        test_input = ["the staff is friendly ."]
        test_input = [val.split() for val in test_input]

        test_input_processed = []
        for list_val in test_input:
            temp_list = []
            for val in list_val:
                temp_list.append(model.vocab.word2id[val])
            test_input_processed.append(temp_list)
        print(test_input_processed)
        print(test_input)
        logger.info("Test input: "+str(test_input))
        logger.info("Test input vector: "+str(test_input_processed))
        with torch.no_grad():
            model.eval()
            test_input_tensor = torch.tensor(test_input_processed).t()
            # output = model.predict_autoencoder(test_input_tensor)
            output = model.predict(test_input_tensor, 1)
            print(output)
            logger.info("Reconstructed sentence: "+str(output))

        