import numpy as np
from keras import models

def build_model(model_params):
    model = models.Sequential

    return model

def vectorize_sequece(sequence, dimensions=10000):
    results = np.zeros((len(sequence), dimensions))
    for i, item in enumerate(sequence):
            results[i, item] = 1
    return results

def to_one_hot(values, dimensions=64):
    result = np.zeros((len(values), dimensions))
    for i, value in enumerate(values):
            result[i, value] = 1
    return result

def print_encoded_text(encoded_text, words, label = '?'):
    text = '  '.join(words.get(i-3, '?') for i in encoded_text)
    print(f'Text: {text}\n\tLabel: {label}')

class ProcessorInfo(object):
    '''
    check "sudo pip3 list" for tensorflow-gpu
    the tensorflow device can be set with tf.device('')
    '''
    @staticmethod
    def show_devices():
        from tensorflow.python.client import device_lib
        return device_lib.list_local_devices()

    @staticmethod
    def show_gpu_devices():
        return [d for d in ProcessorInfo.show_devices() if d.device_type == 'GPU']

