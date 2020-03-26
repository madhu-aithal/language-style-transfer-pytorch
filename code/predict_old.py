import torch 
import torch.nn as nn
import utils
from nltk.translate.bleu_score import sentence_bleu

# PATH = "model_saves/model_0.0005_50_14:10_03-14-2020/" # Batch size = 64. Alt training from the beginning
# PATH = "model_saves/model_0.0005_50_12:42_03-14-2020/" # Batch size = 512. Alt training from the beginning
# PATH = "model_saves/model_0.0005_50_15:52_03-14-2020/"
# PATH = "model_saves/model_0.0005_20_12:24_03-15-2020/" #Autorncoder only
PATH = "model_saves/model_0.0005_50_01:57_03-17-2020/"
PREDICTIONS_OUTPUT_FILE = PATH+"output_predictions"
PREDICTIONS_INPUT_FILE = "/home/madhu/language-style-transfer-pytorch/data/yelp/sentiment.dev"

def predict_batch(model, test_inputs, sentiment, greedy_search=True, plain_format=True):
    test_outputs = []   
    # max_len = max([len(val) for val in test_inputs])
    # size = len(test_inputs)
    # test_inputs_tensor = torch.zeros([max_len, size])
    batches, _ = utils.get_batches_single(test_inputs, vocab.word2id,
                model.args.batch_size, noisy=True)
    for batch in batches:
        batch_input = torch.tensor(batch_input["enc_inputs"], device=self.device)
        output = model.predict_greedy_search(test_input_tensor, sentiment)
        

    for test_input in test_inputs:
        test_input = [val for val in test_input.split(" ")]
        test_input_processed = []
        for val in test_input:
            if val not in model.vocab.word2id:
                test_input_processed.append(model.vocab.word2id['<unk>'])
            else:
                test_input_processed.append(model.vocab.word2id[val])

    with torch.no_grad():
        model.eval()
        test_input_tensor = torch.tensor(test_input_processed, device=model.device).unsqueeze(1)     
        if greedy_search == True:
            output = model.predict_greedy_search(test_input_tensor, sentiment)
        else:
            output = model.predict_beam_search(test_input_tensor, sentiment, 10)
        test_outputs.append(" ".join(output))
            # if plain_format == True:
            #     print(output)
            # else:
            #     print("Input: ", test_input)
            #     print("Reconstructed input:", output)        
            #     score = sentence_bleu(test_inputs, " ".join(output[:-1]))
            #     blue_scores.append(score)
            #     print(f"BLEU score: {score}")
            #     print("--------------------")
            #     print()
    # blue_scores = torch.tensor(blue_scores)
    # print("avg BLEU score: ", torch.mean(blue_scores))
    return test_outputs


if __name__ == "__main__":
    file_neg = open(PREDICTIONS_INPUT_FILE+".0", "r")
    test_input_0 = file_neg.readlines()
    test_input_0 = [val.rstrip() for val in test_input_0]

    file_pos = open(PREDICTIONS_INPUT_FILE+".1", "r")
    test_input_1 = file_pos.readlines()
    test_input_1 = [val.rstrip() for val in test_input_1]
    # test_input_0 = [
    #     # "my goodness it was so gross .",
    #     "the steak was rough and bad .",
    #     "it really feels like a total lack of effort , honesty and professionalism .",
    #     "i was not impressed at all .",
    #     "seriously though , their food is so bad .",
    #     "service was just awful ."
    #     ]

    # test_input_1 = [
    #     # "i love the ladies here !",
    #     # "came here with my wife and her grandmother !",
    #     "the breakfast was the best and the women helping with the breakfast were amazing !",
    #     "it 's a good place to hang out .",
    #     "real nice place .",
    #     "by far the best cake donuts in pittsburgh .",
    #     "the sushi was surprisingly good ."
    #     ]

    # print("Negative sents")
    # for val in test_input_0:
    #     print(val)

    # print("--------------")
    # print("Positive sents")
    # for val in test_input_1:
    #     print(val)

    epoch_saves = [20]

    for epoch in epoch_saves:   
        output_file_0 = open(PREDICTIONS_OUTPUT_FILE+f"_{epoch}_epochs_neg_to_pos.txt", "w")
        output_file_1 = open(PREDICTIONS_OUTPUT_FILE+f"_{epoch}_epochs_pos_to_neg.txt", "w")
        print(f"\nModel trained with {epoch} epochs\n-------------")
        path = PATH+str(epoch)+"_epochs" 
        model = torch.load(path)
        model.training = False
        print("Negative to positive")
        # output_file_0.write("Negative to positive \n")
        test_outputs = predict_batch(model, test_input_0, sentiment=1, greedy_search=True, plain_format=True)
        output_file_0.write('\n'.join(test_outputs) + '\n')
        print("-----------------")
        print("Positive to negative")
        # output_file.write("Positive to negative \n")
        test_outputs = predict(model, test_input_1, sentiment=0, greedy_search=True, plain_format=True)
        output_file_1.write('\n'.join(test_outputs) + '\n')
        print("-----------------")