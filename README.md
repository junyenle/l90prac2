INSTRUCTIONS:

1. extract data.7z and place data and doc2vecdata folders into the root (this) folder
      you can put it somewhere else but you will have to configure data paths in the scripts below

CORE FUNCTIONS
1. GENERATE THE PRESENCE VECTORS
   open generatebow.py
   configure settings on lines 11-22 as you wish
      be especially sure to check the data path on lines 20-21
   run python3 generatebow.py
2. TRAIN DOC2VEC
   open traindoc2vec.py
   check data path on line 8
   set epochs to train on line 38
      the program will train both DM and DBOW models for all epoch numbers placed in this array
   configure DM and DBOW if you wish at lines 42-46 and 60-64
   run python3 traindoc2vec.py
3. TRAIN AND TEST SVM
   open svm.py
   configure svm settings on line 112, if you wish
   set models to use for training svm on line 14
      a separate svm will be trained and tested for each of the models in the array (THE MODEL MUST EXIST! step 2 creates them)
         set model to "naive" to run presence vectors instead of doc2vec
   run python3 svm.py
      scores will be printed in the file specified on line 20

UTILITIES:
1. sign test
   put data in the arrays on lines 13-14
   run python3 signtest.py
2. permuation test
   set the null and hyp models on lines 6 and 9 (THESE MUST EXIST AND HAVE CORRESPONDING SCORES IN THE SVM OUTPUT FILE)
   run python3 permuationtest.py
   