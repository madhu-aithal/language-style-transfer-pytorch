import torch 
import torch.nn as nn
from utils import *

PATH = "model_saves/model_0.0005_250_22:18_03-12-2020"

model = torch.load(PATH)
model.training = False

def predict(test_inputs, sentiment, greedy_search=True):    
    for test_input in test_inputs:
        test_input = [val for val in test_input.split(" ")]

        test_input_processed = []
        for val in test_input:
            if val not in model.vocab.word2id:
                test_input_processed.append(model.vocab.word2id['<unk>'])
            else:
                test_input_processed.append(model.vocab.word2id[val])

        print("Input: ", test_input)
        
        with torch.no_grad():
            model.eval()
            test_input_tensor = torch.tensor(test_input_processed, device=model.device).unsqueeze(1)     
            if greedy_search == True:
                output = model.predict_greedy_search(test_input_tensor, sentiment)
            else:
                output = model.predict_beam_search(test_input_tensor, sentiment, 10)
            print("Reconstructed input:", output)
        
        print("--------------------")
        print()

if __name__ == "__main__":
    test_input_0 = [
        # "my goodness it was so gross .",
        "the steak was rough and bad .",
        "it really feels like a total lack of effort , honesty and professionalism .",
        "i was not impressed at all .",
        "seriously though , their food is so bad .",
        "service was just awful ."
        ]

    test_input_1 = [
        # "i love the ladies here !",
        # "came here with my wife and her grandmother !",
        "the breakfast was the best and the women helping with the breakfast were amazing !",
        "it 's a good place to hang out .",
        "real nice place .",
        "by far the best cake donuts in pittsburgh .",
        "the sushi was surprisingly good ."
        ]

    print("Negative sentences")
    predict(test_input_0, sentiment=1)
    print("-----------------")
    print("Positive sentences")
    predict(test_input_1, sentiment=0)