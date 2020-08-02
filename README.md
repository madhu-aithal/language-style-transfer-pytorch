# language-style-transfer-pytorch
This is a PyTorch implementation of Language style transfer using cross-alignment technique. Paper link [here](https://papers.nips.cc/paper/7259-style-transfer-from-non-parallel-text-by-cross-alignment.pdf). It contains code for encoding the text with two different methods. 
1. Bag of words (`/code_bow/`)
2. Byte pair encoding (`/code_bpe/`)

How to install
----
Make sure you have Python 3.7 or above and pip installed. Execute the below command to install the required packages
    ```
    pip install -r requirements.txt
    ```

How to run
---
Below are sample commands to train and test Style transfer and TextCNN models. Replace `code_bow` with `code_bpe` to use BPE encoding. Please note that positive and negative tweets/reviews files have same prefix. They end with `.0` and `.1` for negative and positive reviews/tweets respectively. Example `tweets.train.0` and `tweets.train.1` for negative and positive tweets respectively. For this example, you have to use the below command to train the Style transfer model
```
python code_bow/style_transfer.py --saves_path <saves_dir> --train "data/tweets/tweets.train" --max_epochs 20 --vocab ./tmp/twitter.vocab
```

1. **Training**
    ```
    python code_bow/style_transfer.py --saves_path <saves_dir> --train <train_files_path> --max_epochs 20 --vocab ./tmp/twitter.vocab
    ```

2. **Prediction**
    ```
    python code_bow/style_transfer.py --predict "the service was great ." --vocab ./tmp/twitter.vocab --model_path <saved_model_path>
    ```

3. **Testing**
    ```
    python code_bow/style_transfer.py --test <tweets_test_data_files_path> --model_path <saved_model_path> --vocab ./tmp/twitter.vocab 
    ```

4. **Testing the model outputs with a TextCNN model**
    ```
    python textcnn/main.py -test=<test_outputs_file_path> -snapshot=<saved_textcnn_model_dir>
    ```