import numpy as np

def accuracy(trueLabel, prediction):
    matches = np.argmax(trueLabel, axis=0) == np.argmax(prediction, axis=0)
    return np.sum(matches) / float(trueLabel.shape[1])
