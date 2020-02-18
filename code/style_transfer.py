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

from model import Model

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
    model = Model(vocab.size+1, dim_hidden, 
    dim_hidden+1, vocab.size, args.dropout_keep_prob, device, logger)
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
        model.train_max_epochs(args, train0, train1, vocab, no_of_epochs)
        torch.save(model, save_model_path)

        