import os
from utils import open_file
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from nltk.tokenize import word_tokenize

TESTING = False

path = "doc2vecdata"
allfiles = []
data = []

if TESTING:
    files = os.listdir(path + "/dummy")
    for filename in files:
        allfiles.append(path + "/dummy/" + filename)
else:    
    files = os.listdir(path + "/A")
    for filename in files:
        allfiles.append(path + "/A/" + filename)
        
    files = os.listdir(path + "/B")
    for filename in files:
        allfiles.append(path + "/B/" + filename)
        
    files = os.listdir(path + "/C")
    for filename in files:
        allfiles.append(path + "/C/" + filename)
    
for filename in allfiles:
    data.append(open_file(filename))
    
print("loaded data")
print(len(data))

tagged_data = [TaggedDocument(words=word_tokenize(_d.lower()), tags=[str(i)]) for i, _d in enumerate(data)]
print("tagged data")

epochs = [10, 20, 30, 40, 50]

for max_epochs in epochs:
    print("\nprocessing "  + str(max_epochs) + "-epoch models")
    model = Doc2Vec(size=100,
                    alpha=0.025, 
                    min_alpha=0.00025,
                    min_count=4,
                    dm =1)
    model.build_vocab(tagged_data)
    for epoch in range(max_epochs):
        print('iteration {0}'.format(epoch))
        model.train(tagged_data,
                    total_examples=model.corpus_count,
                    epochs=model.iter)
        # decrease the learning rate
        model.alpha -= 0.0002
        # fix the learning rate, no decay
        model.min_alpha = model.alpha
    model.save("dm_" + str(max_epochs) + "e.model")
    print("saved dm_" + str(max_epochs) + "e")

    model = Doc2Vec(size=100,
                    alpha=0.025, 
                    min_alpha=0.00025,
                    min_count=4,
                    dm =0)
    model.build_vocab(tagged_data)
    for epoch in range(max_epochs):
        print('iteration {0}'.format(epoch))
        model.train(tagged_data,
                    total_examples=model.corpus_count,
                    epochs=model.iter)
        # decrease the learning rate
        model.alpha -= 0.0002
        # fix the learning rate, no decay
        model.min_alpha = model.alpha
    model.save("dbow_" + str(max_epochs) + "e.model")
    print("saved dbow_" + str(max_epochs) + "e")


