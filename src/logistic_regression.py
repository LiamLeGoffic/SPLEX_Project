import numpy as np

def updateTypeColumn(dataframe, columnName, columnValue):
    for index, row in dataframe.iterrows():
        if row.Legendary == columnValue:
            dataframe.loc[index, columnName] = 1
        else:
            dataframe.loc[index, columnName] = 0

def sigmoid(z):
    return 1/(1 + np.exp(-z))

def costFunction(x, y, m, theta):
    loss = 0
    for i in range(m):
        z = np.dot(
            np.transpose(theta),
            x[i]
        )
        loss += y[i] * np.log(sigmoid(z)) + (1 - y[i]) * np.log(1 - sigmoid(z))
    return -(1/m) * loss

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
        #print('Current Error is:', costFunction(x, y, m, theta))
    return theta

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

def K_fold_logistic_regression(data, features, target, k):
    Accuracy = []
    Error = []
    data = data.sample(frac=1)
    X = [[row[1][feature] for feature in features] for row in data.iterrows()]
    Y = [row[1][target] for row in data.iterrows()]
    n = len(X)
    for i in range(k):
        print('k folding n°', i+1)
        start = int(i*n/k)
        end = int((i+1)*n/k)
        X_test, X_train = X[start:end], X[:start]+X[end:]
        Y_test, Y_train = Y[start:end], Y[:start]+Y[end:]
        print('SIZE :\ntrain ->', len(X_train), '/ validation ->', len(X_test))
        print('NUMBER OF LEGENDARIES :\ntrain ->', sum(Y_train), '/ validation ->', sum(Y_test))
        theta = np.random.uniform(size=len(X_train[0]))
        theta = gradientDescent(X_train, Y_train, len(X_train[0]), theta, 0.001)
        accuracy_rate, error_rate = test(X_test, Y_test, len(Y_test), theta)
        #print(theta)
        Accuracy.append(accuracy_rate)
        Error.append(error_rate)
    return Accuracy, Error