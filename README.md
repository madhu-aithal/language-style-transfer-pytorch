# language-style-transfer-pytorch
This is a PyTorch implementation of Language style transfer using cross-alignment technique



model_0.0005_50_12:05_03-16-2020 - discriminator only. dropout prob=0.5
model_0.0005_50_18:26_03-15-2020 - adv training after pretraining autoencoder (upto loss 0.2 and disc 80%)
model_0.0005_30_18:34_03-15-2020 - autoencoder training only - loss upto 0.1
model_0.0005_30_23:51_03-15-2020 - autoencoder only dropout prob = 0.2, upto loss 0.1
model_0.0005_30_09:50_03-16-2020 - disc. training only, dropout prob 0.2, accuracy > 80%

model_0.0005_50_01:00_03-17-2020 - adv training dropout prob=0.5, w/ pretrained models
model_0.0005_50_18:54_03-16-2020 - adv training dropout prob=0.2, w/ pretrained models
model_0.0005_50_01:57_03-17-2020 - adv training dropout prob=0.5, w/o pretrained models
model_0.0005_50_09:57_03-17-2020 - adv training dropout prob=0.5, w/ pretrained models (ae - 5 epochs)
model_0.0005_50_00:17_03-18-2020 - adv training dropout prob=0.2, w/o pretrained models

model_0.0005_1_10:42_03-18-2020 - autoencoder pretraining, one epoch only, dropout prob=0.2
model_0.0005_1_10:44_03-18-2020 - autoencoder pretraining, one epoch only, dropout prob=0.5
model_0.0005_50_11:35_03-18-2020 - autoencoder w/ pretraining, dropout prob=0.2, autencoder trained 1 epoch, discrminator trained for 1 epoch

model_0.0005_50_02:01_03-19-2020 - adv training w/o pretrained models, dropout_prob=0.5, batch_size=512
model_0.0005_50_02:04_03-19-2020 - adv training w/o pretrained models, dropout_prob=0.5, batch_size=256
