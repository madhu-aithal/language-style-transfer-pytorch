import sys
import math

if __name__=="__main__":

    translated_file = sys.argv[1]
    original_file = sys.argv[2]
    out_file = sys.argv[3]
    word_count_translated = {}
    word_count_original = {}

    f_trans = open(translated_file, "r")
    f_orig = open(original_file, "r")
    f_out = open(out_file, "w")
    f_out.write("Top 100 words\n")
    data = f_trans.readlines()
    for line in data:
        for word in line.split():
            word_count_translated[word] = word_count_translated.get(word, 0)+1
    
    data = f_orig.readlines()
    for line in data:
        for word in line.split():
            word_count_original[word] = word_count_original.get(word, 0)+1
    diff = set(word_count_translated) - set(word_count_original)

    word_count_translated_copy = word_count_translated.copy()
    for key in word_count_translated_copy.keys():
        if not (key in diff):
            del word_count_translated[key]
    
    del word_count_translated['<eos>']    

    total_count = sum(list(word_count_translated.values()))
    entropy = 0

    for key, val in word_count_translated.items():
        p = val/total_count*1.0
        entropy += p*math.log2(p)

    word_count_translated = {k: v for k, v in sorted(word_count_translated.items(), key=lambda item: item[1], reverse=True)}    
    count = 0
    
    for key, val in word_count_translated.items():
        f_out.write(str({
            key: val
        })+"\n")
        count += 1
        if count>=100:
            break
    # f_out.write("Total number of words: "+str(total_count)+"\n")
    f_out.write("Entropy: "+str(-entropy)+"\n")
    # print("Word count: ", str(word_count_translated))
    print(str(word_count_translated))    
    print("Entropy: ", -entropy)