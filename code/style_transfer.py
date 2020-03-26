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
import utils
import numpy as np
import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
from pathlib import Path

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
    logger = utils.init_logging(args)
    
    print("args: ", args)
    logger.info("args: "+str(args))
    no_of_epochs = args.max_epochs

    if args.predict:
        if args.model_path:
            logger.info("Predicting a sample input\n---------------------\n")
            model = torch.load(args.model_path)
            model.training = False
            output = utils.predict(model, args.predict, args.target_sentiment, args.beam)
            print(f"Input given: {args.predict} \nTarget sentiment: {args.target_sentiment} \nTranslated output: {output}")
            logger.info(f"Input given: {args.predict} \nTarget sentiment: {args.target_sentiment} \nTranslated output: {output}")
        
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

    if args.test:
        test0 = load_sent(args.test + '.0')
        test1 = load_sent(args.test + '.1')
        if args.model_path:
            saves_path = os.path.join(args.saves_path, utils.get_filename(args, "model"))
            Path(saves_path).mkdir(parents=True, exist_ok=True)
            model = torch.load(args.model_path)
            model.training = False
            batches0, batches1, _, _ = utils.get_batches(test0, test1, model.vocab.word2id, model.args.batch_size)

            output_file_0 = open(os.path.join(saves_path, "test_outputs_neg_to_pos"), "w")
            output_file_1 = open(os.path.join(saves_path, "test_outputs_pos_to_neg"), "w")

            for batch0, batch1 in zip(batches0, batches1):
                batch0 = batch0["enc_inputs"]
                batch1 = batch1["enc_inputs"]
                test_outputs_0 = utils.predict_batch(model, batch0, sentiment=1, beam_size=args.beam, plain_format=True)
                test_outputs_1 = utils.predict_batch(model, batch1, sentiment=0, beam_size=args.beam, plain_format=True)
                output_file_0.write('\n'.join(test_outputs_0) + '\n')
                output_file_1.write('\n'.join(test_outputs_1) + '\n')
                
    if args.train:
        summ_filename = 'runs/cross-alignment/'+utils.get_filename(args, "summary")
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