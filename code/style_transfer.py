import os
from vocab import Vocabulary, build_vocab
from file_io import load_sent, write_sent
from options import load_arguments


import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Encoder(nn.Module):
    def __init__(self, input_size, hidden_size, dropout_p):
        super(Encoder, self).__init__()
        self.hidden_size = hidden_size
        self.input_size = input_size
        self.dropout_p = dropout_p

        self.embedding = nn.Embedding(input_size, hidden_size)
        self.gru = nn.GRU(input_size, hidden_size, dropout=self.dropout_p)

    def forward(self, input, hidden):
        embedded_input = self.embedding(input)        
        output, hidden = self.gru(embedded_input, hidden)
        return output, hidden

    def initHidden(self):
        return torch.zeros(1, 1, self.hidden_size, device=device)

# class Decoder(nn.Module):
#     def __init__(self, hidden_size, output_size):
#         super(Decoder, self).__init__()
#         self.hidden_size = hidden_size
#         self.output_size = output_size

    


def get_model(args, vocab):
    model = Encoder(vocab.dim_emb, 200 + 500)

if __name__ == '__main__':
    args = load_arguments()

    #####   data preparation   #####
    if args.train:
        train0 = load_sent(args.train + '.0', args.max_train_size)
        train1 = load_sent(args.train + '.1', args.max_train_size)
        print((train0+train1)[len(train0)])
        # print(train1[0])
        print('#sents of training file 0:', len(train0))
        print('#sents of training file 1:', len(train1))

        if not os.path.isfile(args.vocab):
            build_vocab(train0 + train1, args.vocab)
    print("args.embedding: ", args.embedding)
    print("args.dim_emb: ", args.dim_emb)
    vocab = Vocabulary(args.vocab, args.embedding, args.dim_emb)
    print('vocabulary size:', vocab.size)

    if args.dev:
        dev0 = load_sent(args.dev + '.0')
        dev1 = load_sent(args.dev + '.1')

    if args.test:
        test0 = load_sent(args.test + '.0')
        test1 = load_sent(args.test + '.1')

    print(train0)