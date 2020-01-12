import operator as op
import math
from functools import reduce

def ncr(n, r):
    r = min(r, n-r)
    numer = reduce(op.mul, range(n, n-r, -1), 1)
    denom = reduce(op.mul, range(1, r+1), 1)
    return numer / denom

#dataA = [.62, .63, .665, .685, .580, .655, .605, .665, .610, .650] # uni
dataA = [.805, .865, .82, .905, .825, .875, .845, .835, .855, .88] # smoothed
dataB = [0.645, 0.655, 0.66, 0.700, 0.585, 0.665, 0.665, 0.675, 0.65, 0.67] # unsmoothed

plus = 0
minus = 0
null = 0
for i in range(len(dataA)):
    if dataA[i] > dataB[i]:
        plus += 1
    elif dataA[i] < dataB[i]:
        minus += 1
    else:
        null += 0
        
N = 2 * math.ceil(null/2) + plus + minus
k = math.ceil(null/2) + min(plus, minus)
q = 0.5

sum = 0
for i in range(k + 1):
    sum += ncr(N, i) * (q ** i) * ((1 - q)**(N - i))
sum *= 2
print(sum)