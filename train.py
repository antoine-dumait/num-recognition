import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

def init_random_params():
    W1 = np.random.rand(10, 784) - 0.5 #weight
    b1 = np.random.rand(10, 1) - 0.5 #bias
    W2 = np.random.rand(10, 10) - 0.5 #weight
    b2 = np.random.rand(10, 1) - 0.5 #bias
    return W1, b1, W2, b2

def ReLu(arr):
    return np.maximum(0, arr)

def ReLU_deriv(arr):
    return arr > 0

def softmax(arr):
    return np.exp(arr) / sum(np.exp(arr))

def forward_prop(W1, b1, W2, b2, X):
    Z1 = W1.dot(X) + b1
    A1 = ReLu(Z1)
    Z2 = W2.dot(A1) + b2
    A2 = softmax(Z2)
    return Z1, A1, Z2, A2

    #returns arr, 1 at index and 0 everywhere else 
    #ex one_hot(Y)->[0,0,0,1,0,0,0,0,0,0] with y.size = 10
# c faux
def one_hot(Y, m):
    one_hot_Y = np.zeros((Y.size, m)) #créer 
    one_hot_Y[np.arange(Y.size), Y] = 1 
    one_hot_Y = one_hot_Y.T
    return one_hot_Y

def backward_prop(Z1, A1, Z2, A2, W1, W2, X, Y):
    one_hot_Y = one_hot(Y, 10)
    dZ2 = A2 - one_hot_Y
    dW2 = 1 / m * dZ2.dot(A1.T)
    db2 = 1 / m * np.sum(dZ2)
    dZ1 = W2.T.dot(dZ2) * ReLU_deriv(Z1)
    dW1 = 1 / m * dZ1.dot(X.T)
    db1 = 1 / m * np.sum(dZ1)
    return dW1, db1, dW2, db2

def update_params(W1, b1, W2, b2, dW1, db1, dW2, db2, alpha):
    W1  = W1 - alpha * dW1
    b1  = b1 - alpha * db1
    W2  = W2 - alpha * dW2
    b2  = b2 - alpha * db2
    return W1, b1, W2, b2

def get_predictions(A2):
    return np.argmax(A2, 0)

def get_accuracy(predictions, Y):
    print(predictions, Y)
    return np.sum(predictions == Y) / Y.size

def gradient_descent(X, Y, alpha, iterations):
    W1, b1, W2, b2 = init_random_params()
    for i in range(0,iterations+1):
        Z1, A1, Z2, A2 = forward_prop(W1, b1, W2, b2, X)
        dW1, db1, dW2, db2 = backward_prop(Z1, A1, Z2, A2, W1, W2, X, Y)
        W1, b1, W2, b2 = update_params(W1, b1, W2, b2, dW1, db1, dW2, db2, alpha)
        if(i % 50 == 0):
            print("Iteration: ", i)
            predictions = get_predictions(A2)
            print(get_accuracy(predictions, Y))
    return W1, b1, W2, b2
if __name__ == "__main__":
    data = pd.read_csv('data.csv')
    # print(data.head())
    data = np.array(data)
    m, n = data.shape
    print(m,n)
    np.random.shuffle(data)
    test_data = data[0:1000].transpose()
    labels_test = test_data[0]
    values_test = test_data[1:n]
    values_test = values_test / 255.
    values_test = values_test.round(0)
    train_data = data[1000:m].transpose()
    labels_train = train_data[0]
    values_train = train_data[1:n]
    values_train = values_train / 255.
    values_train = values_train.round(0)
    print(values_train[0][20000:21000])
    print(values_train.shape)
    W1, b1, W2, b2 = gradient_descent(values_train, labels_train, 0.5, 4000)
    np.savetxt("weight_1", W1)
    np.savetxt("bias_1", b1)
    np.savetxt("weight_2", W2)
    np.savetxt("bias_2", b2)

