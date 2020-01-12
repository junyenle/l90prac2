import os
from utils import open_file, get_data_paths
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from sklearn import svm
from nltk.tokenize import word_tokenize
import random
import time
import statistics


POS = 1
NEG = 0

models = ["dbow_30e.model"] # set to "naive" for bowvector
numfolds = 10
# root = ""
# root = '/usr/groups/mphil/L90/data-tagged'
data_positive = data/POS'
data_negative = data/NEG'
outfile = open("svm_output.txt", 'w+')
bowvectorsource = set()

naivemodelfile = open("bowvector.txt", "r")
for rawline in naivemodelfile.readlines():
    word = rawline.strip()
    bowvectorsource.add(word)
naivemodelfile.close()

def convertData(data_files, modelname, posorneg):
    toRet = []
    toRet2 = []
    if modelname == "naive":
        for filename in data_files:
            vector = []
            raw_data = open_file(filename)
            tokenized_data = word_tokenize(raw_data.lower())
            length = 1 * len(tokenized_data) - 1
            for word in bowvectorsource:
                #presence
                if word in tokenized_data:
                    vector.append(1)
                else:
                    vector.append(0)
                #normalized frequency
                sum = 0
                #for tokword in tokenized_data:
                #    if word == tokword:
                #        sum += 1 / length
                #vector.append(sum)
            for i, word in enumerate(bowvectorsource):
                if word in tokenized_data:
                    vector.append(1)
                else:
                    vector.append(0)
                #for tokword in tokenized_data:
                #    if word == tokword:
                #        sum += 1 / length
            toRet.append(vector)
            toRet2.append(posorneg)  
    elif modelname[:6] == "concat":
        epoch = modelname[6:]
        model1 = Doc2Vec.load("dbow_" + epoch + "e.model")
        model2 = Doc2Vec.load("dm_" + epoch + "e.model")
        for filename in data_files:
            raw_data = open_file(filename)
            #print(raw_data)
            tokenized_data = word_tokenize(raw_data.lower())
            #print(tokenized_data)
            vector1 = model1.infer_vector(tokenized_data)
            vector2 = model2.infer_vector(tokenized_data)
            toRet.append(vector1 + vector2)
            toRet2.append(posorneg)
    else:
        model= Doc2Vec.load(modelname)
        for filename in data_files:
            raw_data = open_file(filename)
            #print(raw_data)
            tokenized_data = word_tokenize(raw_data.lower())
            #print(tokenized_data)
            vector = model.infer_vector(tokenized_data)
            toRet.append(vector)
            toRet2.append(posorneg)
    return toRet, toRet2

for MODELFILE in models:
    print(MODELFILE)
    accuracies = []
    outfile.write(MODELFILE + "\n")
    for fold in range(numfolds):
        # get data
        print("\nfold {}".format(fold))
        print("obtaining data")
        train_pos, test_pos = get_data_paths(data_positive, numfolds, fold)  
        train_neg, test_neg = get_data_paths(data_negative, numfolds, fold)

        # preprocess data
        print("preprocessing data")
        start = time.time()
        pos_features, pos_labels = convertData(train_pos, MODELFILE, POS)
        neg_features, neg_labels = convertData(train_neg, MODELFILE, NEG)
        features = pos_features + neg_features
        labels = pos_labels + neg_labels
        # random shuffle
        z = list(zip(features, labels))
        random.shuffle(z)
        features[:], labels[:] = zip(*z)
        #print(time.time() - start)

        # train model
        print("training model")
        #start = time.time()
        clf = svm.SVC(gamma='scale', degree = 3, kernel = 'poly', C=1)
        clf.fit(features, labels)
        #print(time.time() - start)

        # test model
        print("testing model")
        #start = time.time()
        pos_features, pos_labels = convertData(test_pos, MODELFILE, POS)
        neg_features, neg_labels = convertData(test_neg, MODELFILE, NEG)
        features = pos_features + neg_features
        oracle = pos_labels + neg_labels
        correct = 0
        incorrect = 0
        predictions = clf.predict(features)
        for i, prediction in enumerate(predictions):
            if prediction == oracle[i]:
                correct += 1
            else:
                incorrect += 1

        print(time.time() - start)
        # compute accuracy
        accuracy = float(correct) / (correct + incorrect)
        print(accuracy)
        accuracies.append(accuracy)
    
    mean = statistics.mean(accuracies)
    variance = statistics.variance(accuracies)
    print("\nmean is {}".format(mean))
    print("variance is {}".format(variance))
    for accuracy in accuracies:
        outfile.write("{}\n".format(accuracy))
    outfile.write("{}\n{}\n".format(mean, variance))
outfile.close()