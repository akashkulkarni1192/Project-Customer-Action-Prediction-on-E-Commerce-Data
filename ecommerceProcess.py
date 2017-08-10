import pandas as pd
import numpy as np

def get_data():
    data = pd.read_csv("ecommerce_data.csv")
    data_matrix = data.as_matrix()
    X = data_matrix[:, :-1]
    Y = data_matrix[:, -1]
    Y_encoded = one_hot_encoding(Y)
    return X, Y, Y_encoded


def one_hot_encoding(Y):
    N = len(Y)
    classes = len(set(Y))
    Y_encoding = np.zeros((N, classes))
    for i in range(N):
        Y_encoding[i, int(Y[i])] = 1
    return Y_encoding

def regularize(X):
    regX = (X - X.mean()) / X.std()
    return regX




