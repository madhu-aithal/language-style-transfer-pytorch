import torch 

PATH = "model_saves/model300_18_45_24_02_2020"

# input = ["What is your name?"]

# input_tensor = torch.tensor(input)
# input_tensor = input_tensor.unsqueeze(0)

model = torch.load(PATH)
model.training = False
# z = model.generate
# print(model.)

test_input = ["this place was very good"]
test_input = [val.split() for val in test_input]

test_input_processed = []
for list_val in test_input:
    temp_list = []
    for val in list_val:
        temp_list.append(model.vocab.word2id[val])
    test_input_processed.append(temp_list)
print(test_input_processed)
print(test_input)
test_input_tensor = torch.tensor(test_input_processed)
print(model.predict(test_input_tensor.t()))