import pandas as pd
import numpy as np
import ecommerceProcess as myprocess
import ecommerceNeuralNet as myneural

if __name__ == '__main__':
    X, Y, Y_encoded = myprocess.get_data()
    D, M, K = myneural.configure_nn_architecture(X, Y)
    W1, B1, W2, B2 = myneural.initialize_weights(D, M, K)
    X[:,1] = myprocess.regularize(X[:,1])
    X[:,2] = myprocess.regularize(X[:,2])
    W2, B1, W2, B2 = myneural.back_propogate(W1, B1, W2, B2, X, Y, D, M, K)
    print("end")
