import spacy
import numpy as np

nlp = spacy.load('en_core_web_md')

YELP_NEG_SENTS_PATH = "data/yelp/sentiment.train.0"
YELP_POS_SENTS_PATH = "data/yelp/sentiment.train.1"

IMDB_POS = "/data/madhu/imdb_dataset/processed_data/cross_alignment/reviews.train.1"
IMDB_NEG = "/data/madhu/imdb_dataset/processed_data/cross_alignment/reviews.train.0"

def mean_length_yelp():
    neg_token_lens = []
    pos_token_lens = []
    count = 0
    with open(YELP_NEG_SENTS_PATH, "r") as fin:
        for line in fin:
            count += 1
            line = line.strip("\n")
            doc = nlp(line)
            neg_token_lens.append(len(doc))
            if count >= 10000:
                break
    
    with open(YELP_POS_SENTS_PATH, "r") as fin:
        count = 0
        for line in fin:
            line = line.strip("\n")
            doc = nlp(line)
            pos_token_lens.append(len(doc))
            if count >= 10000:
                break
    mean_sent_len = (np.mean(neg_token_lens)+np.mean(pos_token_lens))/2
    print("Yelp - mean sent length: ", mean_sent_len)

def mean_length_imdb():
    neg_token_lens = []
    pos_token_lens = []
    count = 0
    with open(IMDB_NEG, "r") as fin:
        for line in fin:
            line = line.strip("\n")            
            doc = nlp(line)
            for sent in doc.sents:
                count += 1
                doc_sent = nlp(sent.string)
                neg_token_lens.append(len(doc_sent))            
                if count >= 10000:
                    break
            if count >= 10000:
                break

    count = 0
    with open(IMDB_POS, "r") as fin:
        for line in fin:
            line = line.strip("\n")            
            doc = nlp(line)
            for sent in doc.sents:
                count += 1
                doc_sent = nlp(sent.string)
                pos_token_lens.append(len(doc_sent))            
                if count >= 50000:
                    break
            if count >= 50000:
                break
    mean_sent_len = (np.mean(neg_token_lens)+np.mean(pos_token_lens))/2
    print("IMDB - mean sent length: ", mean_sent_len)

if __name__ == "__main__":
    mean_length_yelp()
    mean_length_imdb()