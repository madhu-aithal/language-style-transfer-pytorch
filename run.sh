#!/bin/bash
for BATCH_SIZE in 128 256 512
do 
    python code/style_transfer.py \
    --max_epochs 10 \
    --save_model_path /home/madhu/language-style-transfer-pytorch/model_saves \
    --log_dir /home/madhu/language-style-transfer-pytorch/logs \
    --train /home/madhu/language-style-transfer-pytorch/data/yelp/sentiment.train \
    --output /home/madhu/language-style-transfer-pytorch/tmp/sentiment.dev \
    --vocab /home/madhu/language-style-transfer-pytorch/tmp/yelp.vocab \
    --model /home/madhu/language-style-transfer-pytorch/tmp/model \
    --cuda_device 2 \
    --batch_size $BATCH_SIZE \
    --dev /home/madhu/language-style-transfer-pytorch/data/yelp/sentiment.dev
done