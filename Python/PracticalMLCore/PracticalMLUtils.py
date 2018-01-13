import numpy as np

def vectorize_sequece(sequence, dimensions=10000):
    results = np.zeros((len(sequence), dimensions))
    for i, item in enumerate(sequence):
        results[i, item] = 1
    return results


def print_encoded_text(encoded_text, words, label = '?'):
    text = '  '.join(words.get(i-3, '?') for i in encoded_text)
    print(f'Text: {text}\n\tLabel: {label}')