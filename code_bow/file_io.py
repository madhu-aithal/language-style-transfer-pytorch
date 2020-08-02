from nltk import word_tokenize, sent_tokenize
import csv
import gzip
import spacy
import random

def load_doc(path):
    data = []
    with open(path) as f:
        for line in f:
            sents = sent_tokenize(line)
            doc = [word_tokenize(sent) for sent in sents]
            data.append(doc)
    return data


nlp = spacy.load("en_core_web_md")
tokenizer = nlp.Defaults.create_tokenizer(nlp)

# def get_sents(reviews, min_no_of_tokens: int = 5):
#     all_sents = []    
#     for rev in reviews:   
#         doc = nlp(rev)        
#         for sent in doc.sents:
#             tokens = tokenizer(sent.string.strip())
#             if len(tokens) >= min_no_of_tokens:
#                 all_sents.append(sent.string.strip().strip("\n")) 

#     return all_sents

def process_sents(sent):   
    sent = sent.strip("\n")
    doc = nlp(sent)
    processed_sent = ""
    for token in doc:
        processed_sent += token.text.lower() + " "
    processed_sent = processed_sent.strip()
    return processed_sent

def load_sent(path, max_size, max_seq_length, sentence_flag):
    data = []
    with open(path) as f:
        reviews = f.read().splitlines()
        # reviews = process_sents(reviews)
        random.shuffle(reviews)
        for line in reviews:
            line = line.strip("\n")
                    
            if sentence_flag:
                doc = nlp(line)
                token_count = len(doc)
                if token_count <= max_seq_length:                    
                    for sent in doc.sents:                    
                        data.append(process_sents(sent.string).split())
                        if len(data) == max_size:
                            break
                if len(data) == max_size:
                    break
            else:
                token_count = len(line.split())
                if token_count <= max_seq_length:                
                    # data.append(process_sents(line))
                    data.append(line.lower().split())
                    if len(data) == max_size:
                        break

    print("data size: ", len(data))
    return data

def load_sent_csvgz(path, max_size=-1):
    data0 = []
    data1 = []
    with gzip.open(path, mode="rt") as f:
        csv_reader = csv.reader(f, delimiter=',')
        next(csv_reader, None)
        for row in csv_reader:  
            if len(data0) == max_size:
                break              
            data0.append(row[0].strip().split())
            data1.append(row[3].strip().split())
    return data0, data1


def load_vec(path):
    with open(path) as f:
        for line in f:
            p = line.split()
            p = [float(v) for v in p]
            x.append(p)
    return x

def write_doc(docs, sents, path):
    with open(path, 'w') as f:
        index = 0
        for doc in docs:
            for i in range(len(doc)):
                f.write(' '.join(sents[index]))
                f.write('\n' if i == len(doc)-1 else ' ')
                index += 1

def write_sent(sents, path):
    with open(path, 'w') as f:
        for sent in sents:
            f.write(' '.join(sent) + '\n')

def write_vec(vecs, path):
    with open(path, 'w') as f:
        for vec in vecs:
            for i, x in enumerate(vec):
                f.write('%.3f' % x)
                f.write('\n' if i == len(vec)-1 else ' ')
