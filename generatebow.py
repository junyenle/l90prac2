import os
from utils import open_file, get_data_paths
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from sklearn import svm
from nltk.tokenize import word_tokenize
import random
import time
import statistics
from collections import Counter

bigrams = True
unigrams = True
numfolds = 10
fold = 0
minfreqbigram = 7
minfrequnigram = 4
ofile = open("bowvector.txt", "w+")

# root = '/usr/groups/mphil/L90/data-tagged'
data_positive = 'data/POS'
data_negative = 'data/NEG'
punc = [',', "'", '.', '"', '-', '_', ';', ':', '/', '\\', '(', ')', '[', ']', '{', '}', '!', '`', '|', '?', '~']

def convertData(data_files, modelname, posorneg):
    model= Doc2Vec.load(modelname)
    toRet = []
    toRet2 = []
    for filename in data_files:
        raw_data = open_file(filename)
        raw_data = [x for x in raw_data if len(x) > 1]
        #print(raw_data)
        tokenized_data = word_tokenize(raw_data.lower())
        #print(tokenized_data)
        vector = model.infer_vector(tokenized_data)
        toRet.append(vector)
        toRet2.append(posorneg)
    return toRet, toRet2
    
train_pos, test_pos = get_data_paths(data_positive, numfolds, fold)  
train_neg, test_neg = get_data_paths(data_negative, numfolds, fold)

vocab = Counter()
words_to_delete = set()
ft = 0
for filename in train_pos + train_neg:
    print(filename)
    raw_data = open_file(filename)
    tokenized_data = word_tokenize(raw_data.lower())
    tokenized_data = [x for x in tokenized_data if len(x) > 1 and x[0] not in punc]
    if bigrams:
        for i, word in enumerate(tokenized_data):
            if i != 0:
                bigram = tokenized_data[i-1] + ' ' + word
                vocab[bigram] += 1
        ft = minfreqbigram
        #for word in vocab:
        #    if vocab[word] < ft:
        #        words_to_delete.add(word)
    if unigrams:
        #print(tokenized_data)
        for word in tokenized_data:
            vocab[word] += 1
        ft = minfrequnigram
        #for word in vocab:
        #    if vocab[word] < ft:
        #        words_to_delete.add(word)
for word in vocab:
    ft = 0
    length = len(word.split())
    if length == 1:
        ft = minfrequnigram
    elif length == 2:
        ft = minfreqbigram
    if vocab[word] < ft:
        words_to_delete.add(word)
for word in words_to_delete:
    del(vocab[word])
    
print(vocab)
print(len(vocab))

for word in vocab:
    ofile.write(word + "\n")

ofile.close()
