import numpy as np
from IPython.display import clear_output

# from https://medium.com/@DataStevenson/logistic-regression-and-pokemon-945e954d84a3
# numerize a column in binary values
def updateTypeColumn(dataframe, columnName, columnValue):
    for index, row in dataframe.iterrows():
        if row.Legendary == columnValue:
            dataframe.loc[index, columnName] = 1
        else:
            dataframe.loc[index, columnName] = 0

# from https://medium.com/@DataStevenson/logistic-regression-and-pokemon-945e954d84a3
def sigmoid(z):
    return 1/(1 + np.exp(-z))

# from https://medium.com/@DataStevenson/logistic-regression-and-pokemon-945e954d84a3
def gradientDescent(x, y, m, theta, alpha, iterations=1500):
    for iteration in range(iterations):
        for j in range(len(theta)):
            gradient = 0
            for i in range(m):
                z = np.dot(
                    np.transpose(theta),
                    x[i]
                )
                gradient += (sigmoid(z) - y[i]) * x[i][j]
            theta[j] = theta[j] - ((alpha/m) * gradient)
    return theta

# from https://medium.com/@DataStevenson/logistic-regression-and-pokemon-945e954d84a3
# get the accuracy and the error on the testing set
def test(x, y, m, theta):
    correct = 0
    for i in range(m):
        z = np.dot(
                np.transpose(theta),
                x[i]
            )
        predicted_value = sigmoid(z)
        if predicted_value >= 0.5 and y[i] == 1:
            correct += 1
        elif predicted_value < 0.5 and y[i] == 0:
            correct += 1
    return correct/m, (1 - (correct/m))

# K-fold validation with N different shuffles of the dataset and get the average accuracy, the average error and the 
# different average weights of each feature (Theta)
def K_fold_logistic_regression(data, features, target, k, N=1, iterations=1500):
    Accuracy = []
    Error = []
    Thetas = []
    for it in range(N):
        if N!=1:
            clear_output()
            print(it+1, '/', N)
        data = data.sample(frac=1)
        X = [[row[1][feature] for feature in features] for row in data.iterrows()]
        Y = [row[1][target] for row in data.iterrows()]
        n = len(X)
        for i in range(k):
            start = int(i*n/k)
            end = int((i+1)*n/k)
            X_test, X_train = X[start:end], X[:start]+X[end:]
            Y_test, Y_train = Y[start:end], Y[:start]+Y[end:]
            theta = np.random.uniform(size=len(X_train[0]))
            theta = gradientDescent(X_train, Y_train, len(X_train[0]), theta, 0.001, iterations=iterations)
            accuracy_rate, error_rate = test(X_test, Y_test, len(Y_test), theta)
            if N==1:
                print('k folding n°', i+1)
                print('SIZE :\ntrain ->', len(X_train), '/ validation ->', len(X_test))
                print('NUMBER OF LEGENDARIES :\ntrain ->', sum(Y_train), '/ validation ->', sum(Y_test))
                print(theta)
            else:
                Thetas.append(theta)
            Accuracy.append(accuracy_rate)
            Error.append(error_rate)
        if N==1:
            return Accuracy, Error
    return np.mean(Accuracy), np.mean(Error), np.mean(Thetas, axis=0)