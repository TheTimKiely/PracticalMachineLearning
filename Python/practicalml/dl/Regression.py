import getopt
import sys
import time

import numpy as np
from ModelParameters import ModelParameters
from PracticalMLUtils import *
from keras.datasets import boston_housing

from practicalml.core.ModelLayer import ModelLayer


def smooth_data(points, factor = 0.9):
    smoothed_points = []
    for point in points:
        if(smoothed_points):
            previous = smoothed_points[-1]
            smoothed_points.append(previous * factor + point * (1 - factor))
        else:
            smoothed_points.append(point)
    return smoothed_points

def build_model():
    model_params = ModelParameters()
    model_params.loss_function = 'mse'
    model_params.optimizer = 'rmsprop'
    model_params.activation = 'relu'
    model_params.metrics.append('mae')
    model_params.layers.append(ModelLayer(64, 'relu', (norm_train_data.shape[1],)))
    model_params.layers.append(ModelLayer(64, 'relu'))
    model_params.layers.append(ModelLayer(1))
    model2 = ModelParameters.build_model(model_params)

    # Because we will need to instantiate
    # the same model multiple times,
    # we use a function to construct it.
    '''
    model = models.Sequential()
    model.add(layers.Dense(64, activation='relu',
                           input_shape=(train_data.shape[1],)))
    model.add(layers.Dense(64, activation='relu'))
    model.add(layers.Dense(1))
    model.compile(optimizer='rmsprop', loss='mse', metrics=['mae'])
    '''
    return model2


epochs = 20
mode = 'p'
print(f'Command: {len(sys.argv)}: {sys.argv}')
opts, args = getopt.getopt(sys.argv[1:], 'e:')
for opt, arg in opts:
    if opt == '-e':
        epochs = int(arg)
    elif opt == '-m':
        mode = arg

start = time.time()

(train_data, train_targets), (test_data, test_targets) = boston_housing.load_data()

mean = train_data.mean(axis=0)
std = train_data.std(axis=0)

norm_train_data = (train_data - mean) / std

# Test data should be normalized using values computed with the training data
norm_test_data = (test_data - mean) / std

k = 4
num_validation_samples = len(norm_train_data) // k
all_scores = []
all_mae = []

if(mode == 't'):
    print(f'Training for {epochs} epochs with {k} folds.')

    for i in range(k):
        start = i * num_validation_samples
        end =(i + 1) * num_validation_samples
        print(f'Processing fold: {i}({start}-{end})')
        val_data = norm_train_data[start:end]
        val_targets = train_targets[start:end]
        partial_train_data = np.concatenate(
            [
                norm_train_data[: start],
                norm_train_data[end:]
            ], axis=0)
        partial_train_targets = np.concatenate(
            [
                train_targets[:start],
                train_targets[end:]
            ], axis=0)

        model = build_model()
        history = model.fit(partial_train_data, partial_train_targets, epochs=epochs, batch_size=1, verbose=0,
                            validation_data=(val_data, val_targets))
        mae_history = history.history['val_mean_absolute_error']
        all_mae.append(mae_history)
        val_mse, val_mae = model.evaluate(val_data, val_targets, verbose=0)
        #val_mae = model.evaluate(val_data, val_targets, verbose=0)
        print(f'val_mse: {val_mse} val_mae: {val_mae} mae_history: {mae_history}')
        all_scores.append(val_mae)

    average_mae = [np.mean([x[i] for x in all_mae]) for i in range(epochs)]
    print(f'Mean MAE: {average_mae}')
    smooth_mean_mae = average_mae
    if(len(average_mae) > 10):
        smooth_mean_mae = smooth_data(average_mae[10:])
    show_plot(range(1, len(smooth_mean_mae) + 1), smooth_mean_mae, 'Epochs', 'Validation MAE')

    print(f'Scores: {all_scores}')
    print(f'Mean Score: {np.mean(all_scores)}')
    print(f'Time: {start - time.time()} sec.')
elif mode == 'p':
    print(f'Predicting for {epochs} epochs.')
    model = build_model()
    model.fit(norm_train_data, train_targets, epochs = epochs, batch_size = 16)
    prediction = model.predict(norm_test_data, batch_size=16)
    print(f'Prediction: {prediction}')
    plt.scatter(test_targets, norm_test_data[:, 5], color='blue')
    prediction_space = np.linspace(min(prediction), max(prediction ), num=len(prediction)).reshape(-1, 1)
    #plt.plot(prediction, prediction_space, color='blue')
    plt.show()
print('done')
