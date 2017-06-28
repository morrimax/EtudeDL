import numpy as np
from sklearn.datasets import fetch_mldata
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split

def mnist():

    mnist = fetch_mldata('MNIST original', data_home='./datasets/mnist')
    mnist.data = mnist.data / np.max(mnist.data).astype(float)

    encoder = OneHotEncoder()
    labels = encoder.fit_transform(mnist.target.reshape(-1,1)).todense()

    trainX, testX, trainY, testY = train_test_split(mnist.data, labels, test_size=0.2)

    return trainX.T, trainY.T, testX.T, testY.T