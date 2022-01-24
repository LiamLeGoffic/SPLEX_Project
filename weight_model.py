import sys
import pandas as pd
import numpy as np
from collections import Counter
from src.weight_method import *

data = pd.read_csv('data/Pokemon.csv')
feature_int = [feature for feature in data.columns if type(data[feature][0])==np.int64 and feature not in ['Generation', '#']]
test=data[feature_int]
classes = list(data['Legendary'])
n=len(test.iloc[:,0])

for lim in np.arange(0.35, 0.45, 0.01):
    predictions = []
    for i in range(n):
        predictions.append(get_class(dict(test.iloc[i, :]), lim)) # 0.38 the best
    print(round(lim, 2))
    TP = len([1 for i in range(n) if classes[i]==predictions[i] and classes[i]])
    TN = len([1 for i in range(n) if classes[i]==predictions[i] and not classes[i]])
    FP = len([1 for i in range(n) if classes[i]!=predictions[i] and classes[i]])
    FN = len([1 for i in range(n) if classes[i]!=predictions[i] and not classes[i]])
    print('Accuracy :', (TP+TN)/n)
    print('Precision :', TP/(TP+FP), TN/(TN+FN), '\n')