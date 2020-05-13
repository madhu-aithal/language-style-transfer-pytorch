import numpy as np
import random
from datetime import datetime
import os
import logging
from pathlib import Path
import torch 
import torch.nn as nn


def strip_eos(sents):
    return [sent[:sent.index('<eos>')] if '<eos>' in sent else sent
        for sent in sents]

def feed_dictionary(model, batch, rho, gamma, dropout=1, learning_rate=None):
    feed_dict = {model.dropout: dropout,
                 model.learning_rate: learning_rate,
                 model.rho: rho,
                 model.gamma: gamma,
                 model.batch_len: batch['len'],
                 model.batch_size: batch['size'],
                 model.enc_inputs: batch['enc_inputs'],
                 model.dec_inputs: batch['dec_inputs'],
                 model.targets: batch['targets'],
                 model.weights: batch['weights'],
                 model.labels: batch['labels']}
    return feed_dict

def makeup(_x, n):
    x = []
    for i in range(n):
        x.append(_x[i % len(_x)])
    return x

def reorder(order, _x):
    x = range(len(_x))
    for i, a in zip(order, _x):
        x[i] = a
    return x

# noise model from paper "Unsupervised Machine Translation Using Monolingual Corpora Only"
def noise(x, unk, word_drop=0.0, k=3):
    n = len(x)
    for i in range(n):
        if random.random() < word_drop:
            x[i] = unk

    # slight shuffle such that |sigma[i]-i| <= k
    sigma = (np.arange(n) + (k+1) * np.random.rand(n)).argsort()
    return [x[sigma[i]] for i in range(n)]


def get_batch(x, y, word2id, sp, noisy=False, min_len=5):
    pad = word2id['<pad>']
    go = word2id['<go>']
    eos = word2id['<eos>']
    unk = word2id['<unk>']

    lengths = []
    rev_x, go_x, x_eos, weights = [], [], [], []
    max_len = max([len(sent) for sent in x])
    max_len = max(max_len, min_len)
    for sent in x:
        sent_id = [word2id[w] if w in word2id else unk for w in sent]
        l = len(sent)
        padding = [pad] * (max_len - l)
        _sent_id = noise(sent_id, unk) if noisy else sent_id
        lengths.append(l-1)
        # torch.nn.utils.rnn.pack_padded_sequence(batch_in, seq_lengths, batch_first=True)
        rev_x.append(padding + _sent_id[::-1])
        go_x.append([go] + sent_id + padding)
        x_eos.append(sent_id + [eos] + padding)
        weights.append([1.0] * (l+1) + [0.0] * (max_len-l))

    return {'enc_inputs': rev_x,
            'dec_inputs': go_x,
            'targets':    x_eos,
            'weights':    weights,
            'labels':     y,
            'size':       len(x),
            'len':        max_len+1,
            'lengths': lengths}

def get_batch_bpe(x, y, sp, noisy=False, min_len=5):    
    pad = sp.pad_id()
    go = sp.bos_id()
    eos = sp.eos_id()
    unk = sp.unk_id()

    lengths = []
    rev_x, go_x, x_eos, weights = [], [], [], []
    
    sent_ids = []
    for sent in x:
        sent_id = sp.EncodeAsIds(" ".join(sent))
        sent_ids.append(sent_id)

    max_len = max([len(sent) for sent in sent_ids])
    max_len = max(max_len, min_len)

    for sent_id in sent_ids:
        # sent_id = [word2id[w] if w in word2id else unk for w in sent]
        l = len(sent_id)
        padding = [pad] * (max_len - l)
        _sent_id = noise(sent_id, unk) if noisy else sent_id
        lengths.append(l-1)
        rev_x.append(padding + _sent_id[::-1])
        go_x.append([go] + sent_id + padding)
        x_eos.append(sent_id + [eos] + padding)
        weights.append([1.0] * (l+1) + [0.0] * (max_len-l))

    return {'enc_inputs': rev_x,
            'dec_inputs': go_x,
            'targets':    x_eos,
            'weights':    weights,
            'labels':     y,
            'size':       len(x),
            'len':        max_len+1,
            'lengths': lengths}

def get_batches(x0, x1, word2id, batch_size, sp, noisy=False):
    if len(x0) < len(x1):
        x0 = makeup(x0, len(x1))
    if len(x1) < len(x0):
        x1 = makeup(x1, len(x0))
    n = len(x0)

    order0 = range(n)
    z = sorted(zip(order0, x0), key=lambda i: len(i[1]))
    order0, x0 = zip(*z)

    order1 = range(n)
    z = sorted(zip(order1, x1), key=lambda i: len(i[1]))
    order1, x1 = zip(*z)

    batches0 = []
    batches1 = []
    s = 0
    while s < n:
        t = min(s + batch_size, n)
        batches0.append(get_batch(x0[s:t],
            [0]*(t-s), word2id, sp, noisy))
        s = t
    
    s = 0
    while s < n:
        t = min(s + batch_size, n)
        batches1.append(get_batch(x1[s:t],
            [1]*(t-s), word2id, sp, noisy))
        s = t

    return batches0, batches1, order0, order1

def get_batches_bpe(x0, x1, batch_size, sp, noisy=False):
    if len(x0) < len(x1):
        x0 = makeup(x0, len(x1))
    if len(x1) < len(x0):
        x1 = makeup(x1, len(x0))
    n = len(x0)

    order0 = range(n)
    z = sorted(zip(order0, x0), key=lambda i: len(i[1]))
    order0, x0 = zip(*z)

    order1 = range(n)
    z = sorted(zip(order1, x1), key=lambda i: len(i[1]))
    order1, x1 = zip(*z)

    batches0 = []
    batches1 = []
    s = 0
    while s < n:
        t = min(s + batch_size, n)
        batches0.append(get_batch_bpe(x0[s:t],
            [0]*(t-s), sp, noisy))
        s = t
    
    s = 0
    while s < n:
        t = min(s + batch_size, n)
        batches1.append(get_batch_bpe(x1[s:t],
            [1]*(t-s), sp, noisy))
        s = t

    return batches0, batches1, order0, order1

def get_batches_single(x, word2id, batch_size, noisy=False):  
    n = len(x)
    order = range(n)
    z = sorted(zip(order, x), key=lambda i: len(i[1]))
    order, x = zip(*z)

    batches = []
    s = 0
    while s < n:
        t = min(s + batch_size, n)
        batches.append(get_batch(x[s:t],
            [0]*(t-s), word2id, noisy))
        s = t
    
    return batches, order

def get_filename(args, time: int, util_name=""):   
    time = datetime.fromtimestamp(int(time))
    filename = str(time.strftime(str(args.learning_rate)+"_"+str(args.max_epochs)+'_%b-%d-%Y_%H-%M-%S'))
    if util_name != "":
        filename = util_name+"_"+filename
    return filename

def init_logging(args, time: int, modelname='cross-alignment'):    
    # Path(args.log_dir).mkdir(parents=True, exist_ok=True)
    Path(args.saves_path).mkdir(parents=True, exist_ok=True)
    save_log_path = os.path.join(args.saves_path, get_filename(args, time, "model"))
    Path(save_log_path).mkdir(parents=True, exist_ok=True)
    filename = get_filename(args, time)
    filename = str(datetime.now().strftime(filename+".log"))
    path = os.path.join(save_log_path, filename)
    logging.basicConfig(filename=path, filemode='w', format='%(name)s - %(levelname)s - %(message)s')
    logger=logging.getLogger() 
    logger.setLevel(logging.DEBUG)
    return logger 


def predict(model, sample_input, sentiment, sp, beam_size=1, plain_format=True):
    sample_input = [val for val in sample_input.split(" ")]
    sample_input_processed = []
    test_output = ""
    for val in sample_input:
        if sp.piece_to_id(val) == 3:
            sample_input_processed.append(sp.unk_id())
        else:
            sample_input_processed += sp.encode_as_ids(val)

    with torch.no_grad():
        model.eval()
        sample_input_tensor = torch.tensor(sample_input_processed, device=model.device).unsqueeze(1)     
        if beam_size == 1:
            output = model.predict_greedy_search(sample_input_tensor, sentiment, sp)
        else:
            output = model.predict_beam_search(sample_input_tensor, sentiment, beam_size, sp)
        test_output = " ".join(output)
    return test_output
         


def predict_batch(model, test_inputs, sentiment, beam_size=1, plain_format=True):
    test_outputs = []   
    for test_input in test_inputs:
        # test_input = [val for val in test_input.split(" ")]
        # test_input_processed = []
        # for val in test_input:
        #     if val not in model.vocab.word2id:
        #         test_input_processed.append(model.vocab.word2id['<unk>'])
        #     else:
        #         test_input_processed.append(model.vocab.word2id[val])

        with torch.no_grad():
            model.eval()
            test_input_tensor = torch.tensor(test_input, device=model.device).unsqueeze(1)     
            if beam_size == 1:
                output = model.predict_greedy_search(test_input_tensor, sentiment)
            else:
                output = model.predict_beam_search(test_input_tensor, sentiment, beam_size)
            test_outputs.append(" ".join(output))         
    return test_outputs