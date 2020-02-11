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

class Generator(nn.Module):
    def __init__(self, hidden_size, output_size, dropout_p):
        super(Generator, self).__init__()
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.dropout_p = dropout_p

        self.gru = nn.GRU(hidden_size, hidden_size, dropout = self.dropout_p)
        self.out = nn.Linear(hidden_size, output_size)

    def forward(self, input, hidden):
        # output = self.embedding(input).view(1, 1, -1)
        # output = F.relu(output)
        output, hidden = self.gru(input, hidden)
        output = self.softmax(self.out(output[0]))
        return output, hidden

    def initHidden(self):
        return torch.zeros(1, 1, self.hidden_size, device=device)

class Autoencoder(nn.Module):
    def __init__(self, input_size_enc, hidden_size_enc, 
    hidden_size_dec, output_size_dec, dropout_p):
        self.input_size_enc = input_size_enc
        self.hidden_size_enc = hidden_size_enc
        self.hidden_size_dec = hidden_size_dec
        self.output_size_dec = output_size_dec
        self.dropout_p = dropout_p
        self.generator1 = Generator(self.hidden_size_dec, self.output_size_dec, self.dropout_p)
        self.generator2 = Generator(self.hidden_size_dec, self.output_size_dec, self.dropout_p)
        self.encoder = Encoder(self.input_size_enc, self.hidden_size_enc, self.dropout_p)

    def forward(self, input):
        encoder_hidden = self.encoder.initHidden()
        generator1_hidden = self.generator1.initHidden()
        generator2_hidden = self.generator2.initHidden()

        # self.encoder(input, encoder_hidden)



def train(input_tensor, target, enc, gen1, gen2, enc_optim, gen1_optim, gen2_optim, criterion):
    # , max_length=MAX_LENGTH)
    encoder_hidden = encoder.initHidden()

    enc_optim.zero_grad()
    gen1_optim.zero_grad()

    input_length = input_tensor.size(0)
    # target_length = target_tensor.size(0)

    # encoder_outputs = torch.zeros(max_length, encoder.hidden_size, device=device)
    encoder_outputs = torch.zeros(enc.hidden_size, device=device)

    loss = 0

    for ei in range(input_length):
        encoder_output, encoder_hidden = encoder(
            input_tensor[ei], encoder_hidden)
        encoder_outputs[ei] = encoder_output[0, 0]

    # gen1_input = torch.tensor(torch.cat([1,encoder_outputs[-1]]), device=device)

    gen1_hidden = torch.cat([1,encoder_outputs[-1]])

    use_teacher_forcing = True if random.random() < teacher_forcing_ratio else False

    if use_teacher_forcing:
        # Teacher forcing: Feed the target as the next input
        for di in range(target_length):
            decoder_output, decoder_hidden, decoder_attention = decoder(
                decoder_input, decoder_hidden, encoder_outputs)
            loss += criterion(decoder_output, target_tensor[di])
            decoder_input = target_tensor[di]  # Teacher forcing

    else:
        # Without teacher forcing: use its own predictions as the next input
        for di in range(target_length):
            decoder_output, decoder_hidden, decoder_attention = decoder(
                decoder_input, decoder_hidden, encoder_outputs)
            topv, topi = decoder_output.topk(1)
            decoder_input = topi.squeeze().detach()  # detach from history as input

            loss += criterion(decoder_output, target_tensor[di])
            if decoder_input.item() == EOS_token:
                break

    loss.backward()

    encoder_optimizer.step()
    decoder_optimizer.step()

    return loss.item() / target_length

def trainIters(encoder, generator1, generator2, print_every=1000, plot_every=100, learning_rate=0.01):
    start = time.time()
    plot_losses = []
    print_loss_total = 0  # Reset every print_every
    plot_loss_total = 0  # Reset every plot_every

    encoder_optimizer = optim.SGD(encoder.parameters(), lr=learning_rate)
    decoder_optimizer = optim.SGD(decoder.parameters(), lr=learning_rate)

    criterion = nn.NLLLoss()
    indices = np.arange(training_pair.shape[0])

    for i in range(indices):
        input_tensor = training_data[i]        
        # target_tensor = training_pair[1]

        loss = train(input_tensor, encoder,
                     generator1, generator2, enc_optim, gen1_optim, gen2_optim, criterion)
        # print_loss_total += loss
        # plot_loss_total += loss

        # if iter % print_every == 0:
        #     print_loss_avg = print_loss_total / print_every
        #     print_loss_total = 0
        #     print('%s (%d %d%%) %.4f' % (timeSince(start, iter / n_iters),
        #                                  iter, iter / n_iters * 100, print_loss_avg))

        # if iter % plot_every == 0:
        #     plot_loss_avg = plot_loss_total / plot_every
        #     plot_losses.append(plot_loss_avg)
        #     plot_loss_total = 0

    # showPlot(plot_losses)

def get_model(args, vocab):
    dim_hidden = args.dim_y+args.dim_z
    encoder_generator = Autoencoder(vocab.dim_emb, dim_hidden, 
    dim_hidden, vocab.size, args.dropout_keep_prob)


if __name__ == '__main__':
    args = load_arguments()

    #####   data preparation   #####
    if args.train:
        train0 = load_sent(args.train + '.0', args.max_train_size)
        train1 = load_sent(args.train + '.1', args.max_train_size)
        # print((train0+train1)[len(train0)])
        # print(train1[0])
        print('#sents of training file 0:', len(train0))
        print('#sents of training file 1:', len(train1))

        if not os.path.isfile(args.vocab):
            build_vocab(train0 + train1, args.vocab)
    # print("args.embedding: ", args.embedding)
    # print("args.dim_emb: ", args.dim_emb)
    vocab = Vocabulary(args.vocab, args.embedding, args.dim_emb)
    # print('vocabulary size:', vocab.size)

    if args.dev:
        dev0 = load_sent(args.dev + '.0')
        dev1 = load_sent(args.dev + '.1')

    if args.test:
        test0 = load_sent(args.test + '.0')
        test1 = load_sent(args.test + '.1')

    # print(train0)