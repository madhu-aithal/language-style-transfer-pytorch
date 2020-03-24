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
    args.dim_z+args.dim_y, vocab.size, args.dropout_keep_prob, device, logger, vocab, 
    lambda_val=1)
    return model

def run_model(args):
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
        model.train_max_epochs(args, train0, train1, dev0, dev1, vocab, no_of_epochs, writer, 
        save_epochs_flag=True, save_epochs=2)
        
if __name__ == '__main__':
    args = load_arguments()
    # batch_sizes = [64, 256, 512]
    # for batch_size in batch_sizes:
    #     print(f"batch size: {batch_size}")
    #     args.batch_size = batch_size
    run_model(args)