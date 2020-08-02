import logging
import os
from vocab import Vocabulary, build_vocab
from file_io import load_sent, write_sent, load_sent_csvgz
from options import load_arguments
import sys
import time
import math
import random
from datetime import datetime
import utils
import numpy as np
import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
from pathlib import Path
# import sentencepiece as spm
from model import Model
from torch.utils.tensorboard import SummaryWriter
import re

class _CustomDataParallel(nn.Module):
    def __init__(self, model, device_ids):
        super(_CustomDataParallel, self).__init__()
        self.model = nn.DataParallel(model, device_ids=device_ids).cuda()
        print(type(self.model))

    # def forward(self, *input):
    #     return self.model(*input)

    def __getattr__(self, name):
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.model.module, name)

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
    # if torch.cuda.device_count() > 1:
    #     print("Let's use", torch.cuda.device_count(), "GPUs!")
    #     # dim = 0 [30, xxx] -> [10, ...], [10, ...], [10, ...] on 3 GPUs
    #     model = _CustomDataParallel(model, device_ids=[1,2,3])

    return model

def run_model(args):
    time = datetime.now().timestamp()  
    
    #####   data preparation   #####
    if args.train:
        
        logger, saves_dir = utils.init_logging(args, time)
        
        print("args: ", args)
        logger.info("args: "+str(args))
        no_of_epochs = args.max_epochs
        train0 = load_sent(args.train + '.0', args.max_train_size, args.max_seq_length, args.sentence_flag)
        train1 = load_sent(args.train + '.1', args.max_train_size, args.max_seq_length, args.sentence_flag)
        
        print('#sents of training file 0:', len(train0))
        print('#sents of training file 1:', len(train1))

        logger.info('#sents of training file 0: ' + str(len(train0)))
        logger.info('#sents of training file 1: ' + str(len(train1)))

        # build vocab for every run
        if not os.path.isfile(args.vocab):
            build_vocab(train0 + train1, args.vocab)

    vocab = Vocabulary(args.vocab, args.embedding, args.dim_emb)
    
    dev0 = []
    dev1 = []
    
    if args.dev:
        dev0 = load_sent(args.dev + '.0', -1, args.max_seq_length, args.sentence_flag)
        dev1 = load_sent(args.dev + '.1', -1, args.max_seq_length,  args.sentence_flag)
    
    if args.predict:
        if args.model_path:
            # logger.info("Predicting a sample input\n---------------------\n")
            device = torch.device("cuda:"+str(args.cuda_device) if torch.cuda.is_available() else "cpu")
            model = torch.load(args.model_path, map_location=device)
            model.training = False
            output = utils.predict(model, args.predict, args.target_sentiment, args.beam)
            print(f"Input given: {args.predict} \nTarget sentiment: {args.target_sentiment} \nTranslated output: {output}")
            # logger.info(f"Input given: {args.predict} \nTarget sentiment: {args.target_sentiment} \nTranslated output: {output}")
    if args.test:
        logger, saves_dir = utils.init_logging(args, time)
        
        print("args: ", args)
        logger.info("args: "+str(args))
        device = torch.device("cuda:"+str(args.cuda_device) if torch.cuda.is_available() else "cpu")
        file0 = open(args.test+".0", "r")
        file1 = open(args.test+".1", "r")
        saves_path = os.path.join(args.saves_path, utils.get_filename(args, time, ""))
        Path(saves_path).mkdir(parents=True, exist_ok=True)
        out_file_0 = open(os.path.join(saves_path, "test_outputs_neg_to_pos"), "w")
        out_file_1 = open(os.path.join(saves_path, "test_outputs_pos_to_neg"), "w")
        model = torch.load(args.model_path, map_location=device)
        model.training = False
        
        for line in file0:
            line = line.strip("\n")
            output = utils.predict(model, line, 1, args.beam)
            out_file_0.write(output+"\n")
                
        for line in file1:
            line = line.strip("\n")
            output = utils.predict(model, line, 0, args.beam)
            out_file_1.write(output+"\n")
                
    if args.train:
        summ_filename = 'runs/cross-alignment/'+utils.get_filename(args, time, "summary")
        writer = SummaryWriter(summ_filename)

        model = get_model(args, vocab, logger)
        model.train_max_epochs(saves_dir, args, train0, train1, dev0, dev1, vocab, no_of_epochs, writer, time,
            save_epochs_flag=True)
        
if __name__ == '__main__':
    args = load_arguments()
    # batch_sizes = [64, 256, 512]
    # for batch_size in batch_sizes:
    #     print(f"batch size: {batch_size}")
    #     args.batch_size = batch_size
    run_model(args)