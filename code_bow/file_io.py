from nltk import word_tokenize, sent_tokenize
import csv
import gzip
import spacy

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

def load_sent(path, max_size=-1, sentence_flag=True):
    data = []
    with open(path) as f:
        for line in f:
            line = line.strip("\n")
            if sentence_flag:
                doc = nlp(line)        
                for sent in doc.sents:                    
                    data.append(sent.string)
                    if len(data) == max_size:
                        break
                if len(data) == max_size:
                    break
            else:
                data.append(line.split())
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
