import numpy as np

def vectorize_sequece(sequence, dimensions=10000):
    results = np.zeros((len(sequence), dimensions))
    for i, item in enumerate(sequence):
        results[i, item] = 1
    return results