import time

import matplotlib.pyplot as plt
import numpy as np
from keras import layers
from keras import models
from keras.datasets import imdb

from practicalml.core import ModelParameters
from practicalml.core import ModelTrainer
from practicalml.core import vectorize_sequece


def print_reviews():
    reverse_word_index = dict( [(value, key) for (key, value) in word_index.items()])
    print(f'reverse_word_index[1] = {reverse_word_index[1]}')
    for i in range(len(train_data)):
        item = ' '.join(reverse_word_index.get(index-3,'?') for index in train_data[i])
        print(f'train_data[{i}]: {item}\n\tPositive: {train_labels[i]}')


(train_data, train_labels), (test_data, test_labels) = imdb.load_data(num_words=10000)
word_index = imdb.get_word_index()

X_train = vectorize_sequece(train_data)
y_train = np.asanyarray(train_labels).astype('float32')
X_test = vectorize_sequece(test_data)
y_test = np.asanyarray(test_labels).astype('float32')
model = models.Sequential()
model.add(layers.Dense(8, activation='relu', input_shape=(10000,)))
model.add(layers.Dense(8, activation='relu'))
model.add(layers.Dense(1, activation='sigmoid'))
model.compile(optimizer='rmsprop',
              loss='binary_crossentropy',
              metrics=['accuracy'])

X_val = X_train[:10000]
partial_X_train = X_train[10000:]
y_val = y_train[:10000]
partial_y_train = y_train[10000:]

start = time.time()
params = ModelParameters()
params.X = partial_X_train
params.y = partial_y_train
params.epochs = 4
params.batch_size = 512
params.validation_data = (X_val, y_val)

trained_model = ModelTrainer.train(model, params)
'''
trained_model = model.fit(partial_X_train, partial_y_train,
                       epochs=4,
                       batch_size=512,
                       validation_data=(X_val, y_val))
'''
print(f'Time: {time.time() - start}')
print(f"keys: {trained_model.history.keys()}")
results = model.evaluate(X_test[:10000], y_test[:10000])
y = model.predict(X_test, verbose=True)
print(f'Results: {results}')
print(f'y({type(y)}: y')


model_dict = trained_model.history
loss_values = model_dict['loss']
val_loss_values = model_dict['val_loss']
epochs = range(1, len(model_dict['acc']) + 1)

plt.plot(epochs, loss_values, 'bo', label='Training loss')
plt.plot(epochs, val_loss_values, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()
plt.clf()
acc_values = model_dict['acc']
val_acc_values = model_dict['val_acc']
plt.plot(epochs, acc_values, 'bo', label='Training acc')
plt.plot(epochs, val_acc_values, 'b', label='Validation acc')
plt.title('Training and validation accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()
'''
model = models.Sequential()
model.add(layers.Dense(32, activation='relu', input_shape=(784,)))
model.add(layers.Dense(10, activation='softmax'))
model.compile(optimizer=optimizers.RMSprop(lr=0.001), loss='mse', metrics=['accuracy'])
model.fit(input_tensor, target_tensor, batch_size=128, epochs=10)
'''