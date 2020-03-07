import torch 
import torch.nn as nn
from utils import *

PATH = "model_saves/model.cross-alignment.0.001.100.03-03-2020.14:53.epoch_100"

model = torch.load(PATH)
model.training = False


def predict_util(model, input_data, target, target_sentiment):
    # target = torch.tensor(target, device=model.device)
    # target = target.unsqueeze(0)
    input_length = input_data.size()[0]
    latent_z = model.get_latent_reps(input_data)

    criterion = nn.CrossEntropyLoss(ignore_index=model.vocab.word2id['<pad>'])

    # latent_z = hidden_states_original[-1,:,:]
    if target_sentiment == 1:
        latent_z = torch.cat((latent_z, torch.ones([1,latent_z.shape[1],model.args.dim_y],
                        dtype=torch.float,device=model.device)), dim=2)
    else:
        latent_z = torch.cat((latent_z, torch.zeros([1,latent_z.shape[1],model.args.dim_y],
                        dtype=torch.float,device=model.device)), dim=2)
    
    gen_input = torch.tensor([model.GO_token], device=model.device)
    gen_hidden = latent_z

    outputs = []
    gen_output = torch.zeros(model.output_size_gen, device=model.device)
    count = 0
    while torch.argmax(gen_output) != model.EOS_token:
        print(model.vocab.id2word[gen_input])
        gen_input = gen_input.unsqueeze(0)
        gen_input = gen_input.unsqueeze(2)
        gen_input = model.encoder.embedding(gen_input).squeeze(2)
        gen_input = model.dropout(gen_input)

        gen_output, gen_hidden = model.generator(
            gen_input, gen_hidden)
        gen_input = torch.argmax(gen_output, dim=1)
        # if count > 0:
        #     loss = criterion(gen_output, target[count-1,:])
        outputs.append(model.vocab.id2word[gen_input])

        count += 1
        # print(model.vocab.id2word[gen_input])
        # if count > input_length:
        #     break
    
    return outputs

def predict():
    test_input = ["the staff is friendly ."]
    # target = ["<eos>"]
    target = test_input
    test_input = [val.split() for val in test_input]
    target_input = [val.split() for val in target]

    test_input_processed = []
    target_input_processed = []
    for list_val1,list_val2 in zip(test_input, target_input):
        temp_list = []
        for val in list_val1:
            temp_list.append(model.vocab.word2id[val])
        test_input_processed.append(temp_list)

        temp_list = []
        for val in list_val2:
            temp_list.append(model.vocab.word2id[val])
        target_input_processed.append(temp_list)

    print(test_input_processed)
    print(target_input_processed)
    # dev0 = load_sent('/home/madhu/language-style-transfer-pytorch/data/yelp/sentimentshort.dev' + '.0')
    # dev1 = load_sent('/home/madhu/language-style-transfer-pytorch/data/yelp/sentimentshort.dev' + '.1')
    # batches0, batches1, _, _ = get_batches(dev0, dev1, vocab.word2id,
    #             args.batch_size, noisy=True)

    with torch.no_grad():
        model.eval()
        test_input_tensor = torch.tensor(test_input_processed, device=model.device).t()
        target_input_tensor = torch.tensor(target_input_processed, device=model.device).t()
        output = model.predict(test_input_tensor, 1)
        print(output)

predict()