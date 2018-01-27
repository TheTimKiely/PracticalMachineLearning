import time
import numpy as np
from keras import models

class Split(object):
    def __init__(self, name, start_time):
        self.Name = name
        self.StartTime = start_time

class Timer(object):
    def __init__(self):
        self.StartTime = None
        self.Splits = []

    def start(self):
        self.StartTime = time.time()

    def start_split(self, name):
        self.Splits.append(Split(name, time.time()))

    def get_split(self, name):
        if name == None:
            return time.time() - self.StartTime
        if self.Splits[name] == None:
            raise KeyError(f'The Split {name} has not been created.')
        return self.Splits[name].StartTime

class Instrumentation(object):
    def __init__(self):
        self.Timer = Timer()


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

