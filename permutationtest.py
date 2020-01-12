import random
import statistics
numpermutations = 5000

# model names MUST match their svm_output.txt labels
nullmodel = "dbow_20e.model"
nulldata = []
nullmean = 0
hypmodel = "dbow_40e.model"
hypdata = []
hypmean = 0

datafile = open("svm_output.txt", "r")
#parse data file
counter = 0
mode = "NEITHER"
for lineraw in datafile.readlines():
    line = lineraw.strip()
    if counter % 13 == 0:
        print(line)
        if line == nullmodel:
            mode = "NULL"
            print("NULL")
        elif line == hypmodel:
            mode = "HYP"
            print("HYP")
        else:
            mode = "NEITHER"
    elif counter % 13 >= 1 and counter % 12 <= 9:
        if mode == "NULL":
            nulldata.append(float(line))
        elif mode == "HYP":
            hypdata.append(float(line))
    elif counter % 13 == 10:
        if mode == "NEITHER":
            continu = 1
        elif mode == "NULL":
            nullmean = float(line)
        elif mode == "HYP":
            hypmean = float(line)
    counter +=1
print(nulldata)
print(hypdata)

reald = hypmean - nullmean
fakeds = []

for i in range(numpermutations):
    nullcopy = nulldata
    hypcopy = hypdata
    for j, item in enumerate(nullcopy):
        if random.randint(0,1) == 1:
            nullcopy[j], hypcopy[j] = hypcopy[j], nullcopy[j]
    nullcopymean = statistics.mean(nullcopy)
    hypcopymean = statistics.mean(hypcopy)
    fakeds.append(hypcopymean - nullcopymean)
    
counter = 0
for d in fakeds:
    if d >= reald:
        counter += 1
        
p = (counter + 1) / (numpermutations + 1)
print("NULL: {}".format(nullmodel))
print("HYP: {}".format(hypmodel))
print(p)