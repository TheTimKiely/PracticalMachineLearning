import matplotlib.pyplot as plt
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

class SeriesData:

    def __init__(self):
        self._x_label = None;

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

class MetricsPlotter(object):
    def __init__(self):
        pass

    def build_series(self, epochs, data, style, label):
        series_data = SeriesData()
        series_data.x_data = epochs
        series_data.y_data = data
        series_data.series_style = style
        series_data.series_label = label
        return series_data

    def plot_metrics(self, metrics):
        loss = metrics.history['loss']
        val_loss = metrics.history['val_loss']
        epochs = range(1, len(loss) + 1)

        loss_series = self.build_series(epochs, loss, 'bo', 'Training Loss')
        self.plot_add_series(loss_series)
        val_loss_series = self.build_series(epochs, val_loss, 'b', 'Validation Loss')
        self.plot_add_series(val_loss_series)
        plt.clf()
        acc = metrics.history['acc']
        val_acc = metrics.history['val_acc']
        acc_series = self.build_series(epochs, acc, 'ro', 'Training Accuracy')
        self.plot_add_series(acc_series)
        val_acc_series = self.build_series(epochs, val_acc, 'r', 'Validation Accuracy')
        self.plot_add_series(val_acc_series)

        plt.title('Training & Validation Loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()

    def plot_add_series(self, series_data):
        plt.plot(series_data.x_data, series_data.y_data, series_data.series_style, label=series_data.series_label)

    def show_plot(self, x, y, xlabel, ylabel, legend=False):
        plt.plot(x, y)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        if (legend == True):
            plt.legend()
        plt.show()

    def show(self):
        plt.show()