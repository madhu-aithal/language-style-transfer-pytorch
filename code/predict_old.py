import torch 
import torch.nn as nn
from utils import *
import numpy as np

PATH = "model_saves/model_0.0005_70_12:18_03-10-2020"

model = torch.load(PATH)
model.training = False


def predict_beam_search(model, input_data, target, target_sentiment, k):
    # target = torch.tensor(target, device=model.device)
    # target = target.unsqueeze(0)
    # input_length = input_data.size()[0]
    latent_z = model.get_latent_reps(input_data)

    # criterion = nn.CrossEntropyLoss(ignore_index=model.vocab.word2id['<pad>'])

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
    result = []
    count = 0
    while torch.argmax(gen_output) != model.EOS_token:
        if len(result) == k:
            break        
        if count == 0:
            gen_input = gen_input.unsqueeze(0)
            gen_input = gen_input.unsqueeze(2)
            gen_input = model.encoder.embedding(gen_input).squeeze(2)
            gen_input = model.dropout(gen_input)

            gen_output, gen_hidden = model.generator(
                gen_input, gen_hidden)
            
            topv, topi = gen_output.topk(k)
            for i in range(topv.size()[1]):                
                outputs.append({
                    "sequence": [topi[:,i].item()],
                    "score": np.log(topv[:,i].item())
                })
                if topi[:,i].item() == model.EOS_token:
                    result.append(outputs[-1])
                    del outputs[-1]
            
        else:
            outputs_old = outputs.copy()
            outputs = []
            for val in outputs_old:
                gen_input = torch.tensor(val["sequence"][-1], device=model.device).view(1,1,1)
                gen_input = model.encoder.embedding(gen_input).squeeze(2)
                gen_input = model.dropout(gen_input)

                gen_output, gen_hidden = model.generator(
                    gen_input, gen_hidden)
                
                topv, topi = gen_output.topk(k)

                for i in range(topv.size()[1]):                          
                    outputs.append({
                        "sequence": val["sequence"] + [topi[:,i].item()],
                        "score": np.log(val["score"])+np.log(topv[:,i].item())
                    })
                    if topi[:,i].item() == model.EOS_token:
                        result.append(outputs[-1])
                        del outputs[-1]
                        
            outputs = sorted(outputs, key = lambda i: i['score'], reverse = True)
            outputs = outputs[:k]
        count += 1
    result = sorted(result, key = lambda i: i['score'], reverse = True)
    
    for output in result:
        output["sentence"]=[model.vocab.id2word[val] for val in output["sequence"]]
    return result

def predict(test_inputs, sentiment):    
    # target = ["<eos>"]
    for test_input in test_inputs:
    
        target = test_input
        test_input = [val for val in test_input.split(" ")]
        target_input = [val for val in target.split(" ")]

        test_input_processed = []
        target_input_processed = []

        for val, val2 in zip(test_input, target_input):
            if val not in model.vocab.word2id:
                test_input_processed.append(model.vocab.word2id['<unk>'])
            else:
                test_input_processed.append(model.vocab.word2id[val])

            if val2 not in model.vocab.word2id:
                target_input_processed.append(model.vocab.word2id['<unk>'])
            else:
                target_input_processed.append(model.vocab.word2id[val])

        target_input_processed.append(model.vocab.word2id['<eos>'])
        target_input_processed = target_input_processed[1:]
        print("Input: ", test_input)
        # print("Input vector: ", test_input_processed)
        # print(target_input_processed)
        # dev0 = load_sent('/home/madhu/language-style-transfer-pytorch/data/yelp/sentimentshort.dev' + '.0')
        # dev1 = load_sent('/home/madhu/language-style-transfer-pytorch/data/yelp/sentimentshort.dev' + '.1')
        # batches0, batches1, _, _ = get_batches(dev0, dev1, vocab.word2id,
        #             args.batch_size, noisy=True)
        
        with torch.no_grad():
            model.eval()
            test_input_tensor = torch.tensor(test_input_processed, device=model.device).unsqueeze(1)
            target_input_tensor = torch.tensor(target_input_processed, device=model.device).unsqueeze(1)
            # output = model.predict(test_input_tensor, sentiment)
            output = predict_beam_search(model, test_input_tensor, target_input_tensor, sentiment, 5)
            print("Reconstructed input:", output)
        
        print("--------------------")
        print()

if __name__ == "__main__":
    test_input_0 = ["i was not impressed at all ."]
    # # "it really feels like a total lack of effort , honesty and professionalism .",
    # "i was not impressed at all .",
    # "seriously though , their food is so bad .",
    # "service was just awful ."]

    # test_input_1 = ["the breakfast was the best and the women helping with the breakfast were amazing !",
    # "it 's a good place to hang out .",
    # "real nice place .",
    # "by far the best cake donuts in pittsburgh .",
    # "the sushi was surprisingly good ."]

    print("Negative sentences")
    predict(test_input_0, sentiment=0)
    print("-----------------")
    print("Positive sentences")
    # predict(test_input_1, sentiment=1)