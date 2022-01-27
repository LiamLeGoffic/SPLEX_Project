import numpy as np
import pandas as pd
from IPython.display import clear_output

# get the predicted predictions thanks to the correlation of each feature with the classe (on the training set)
def special_method(data, corr, constant, features):
    pred = []
    for i in range(len(data.iloc[:,0])):
        n = np.dot(np.transpose(corr), data[features].iloc[i, :])
        if n-constant>=0:
            pred.append(1)
        else:
            pred.append(0)
    return pred

# compare the predictions with the true classes
def test_sm(pred, Y):
    TP, FP, TN, FN = 0, 0, 0, 0
    Y = list(Y)
    for i in range(len(pred)):
        if pred[i]==Y[i]:
            if pred[i]==1:
                TP+=1
            else:
                TN+=1
        else:
            if pred[i]==1:
                FP+=1
            else:
                FN+=1
    acc = (TP+TN)/len(Y)
    pre = TP/max(1, (TP+FP))
    rec = TP/(TP+FN)
    f_score = 2*pre*rec/max(1, (pre+rec))
    return acc, pre, rec, f_score

# k-fold validation of the special method that is repeted N times with a different shuffle each time (because of the data size) 
# and get the average accuracy, the average precision, the average recall and the average f-score
def K_fold_special_method(data, features, target, k, N, constant):
    Accuracy, Precision, Recall, F_score = [], [], [], []
    for it in range(N):
        clear_output()
        print(it+1, '/', N)
        data = data.sample(frac=1)
        X = data[features]
        Y = data[target]
        n = len(X)
        for i in range(k):
            start = int(i*n/k)
            end = int((i+1)*n/k)
            X_test, X_train = X.iloc[start:end, :], pd.concat([X.iloc[:start, :], X.iloc[end:, :]], axis=0)
            Y_test, Y_train = Y.iloc[start:end], pd.concat([Y.iloc[:start], Y.iloc[end:]], axis=0)
            corr = pd.concat([X_train, Y_train], axis=1).corr()[target][features]
            pred = special_method(X_test, corr, constant, features)
            acc, pre, rec, f_score = test_sp(pred, Y_test)
            Accuracy.append(acc)
            Precision.append(pre)
            Recall.append(rec)
            F_score.append(f_score)
    return np.mean(Accuracy), np.mean(Precision), np.mean(Recall), np.mean(F_score)