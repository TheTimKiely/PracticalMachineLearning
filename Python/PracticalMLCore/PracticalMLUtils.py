import numpy as np
import matplotlib.pyplot as plt

from PracticalMLCore.SeriesData import SeriesData


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


def build_series(epochs, data, style, label):
    series_data = SeriesData()
    series_data.x_data = epochs
    series_data.y_data = data
    series_data.series_style = style
    series_data.series_label = label
    return series_data

def plot_metrics(metrics):
    loss = metrics.history['loss']
    val_loss = metrics.history['val_loss']
    epochs = range(1, len(loss) + 1)

    loss_series = build_series(epochs, loss, 'bo', 'Training Loss')
    plot_add_series(loss_series)
    val_loss_series = build_series(epochs, val_loss, 'b', 'Validation Loss')
    plot_add_series(val_loss_series)
    plt.clf()
    acc = metrics.history['acc']
    val_acc = metrics.history['val_acc']
    acc_series = build_series(epochs, acc, 'ro', 'Training Accuracy')
    plot_add_series(acc_series)
    val_acc_series = build_series(epochs, val_acc, 'r', 'Validation Accuracy')
    plot_add_series(val_acc_series)

    plt.title('Training & Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()


def plot_add_series(series_data):
    plt.plot(series_data.x_data, series_data.y_data, series_data.series_style, label=series_data.series_label)