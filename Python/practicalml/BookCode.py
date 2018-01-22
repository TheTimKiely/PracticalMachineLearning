import os
import numpy as np
from keras.preprocessing.image import ImageDataGenerator

import time

from core.utils import MetricsPlotter

s = time.time()
from keras.models import load_model
from keras import models
from keras import layers
from keras import optimizers
from keras.applications import VGG16
print(f'Import time: {time.time() - s}')

base_dir = 'D:\code\ML\data\dogs_and_cats'

train_dir = os.path.join(base_dir, 'train')
validation_dir = os.path.join(base_dir, 'validation')
test_dir = os.path.join(base_dir, 'test')

datagen = ImageDataGenerator(rescale=1./255)
batch_size = 10
from dl.keras_models import ModelRepository
model_path = 'D:\code\ML\models\\vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5'
conv_base = ModelRepository().get_vgg16()

def extract_features_from_model(model, directory, sample_count):
    features = np.zeros(shape=(sample_count, 4, 4, 512))
    labels = np.zeros(shape=(sample_count))
    generator = datagen.flow_from_directory(
        directory,
        target_size=(150, 150),
        batch_size=batch_size,
        class_mode='binary')
    i = 0
    for inputs_batch, labels_batch in generator:
        print(f'Predicting: {i}')
        features_batch = model.predict(inputs_batch)
        features[i * batch_size : (i + 1) * batch_size] = features_batch
        labels[i * batch_size : (i + 1) * batch_size] = labels_batch
        i += 1
        if i * batch_size >= sample_count:
            # Note that since generators yield data indefinitely in a loop,
            # we must `break` after every image has been seen once.
            break
    return features, labels

train_samples = 2000
validation_samples = 1000
raw_train_features, raw_train_labels = extract_features_from_model(conv_base, train_dir, train_samples)
raw_validation_features, raw_validation_labels = extract_features_from_model(conv_base, validation_dir, validation_samples)
raw_test_features, raw_test_labels = extract_features_from_model(conv_base, test_dir, validation_samples)
print(f'test_feastures: {raw_test_features.shape} test_labels: {raw_test_labels.shape}')

train_features = np.reshape(raw_train_features, (train_samples, 4 * 4 *512))
validation_features = np.reshape(raw_validation_features, (validation_samples, 4 * 4 * 512))
test_features = np.reshape(raw_test_features, (validation_samples, 4 * 4 * 512))

model = models.Sequential()
model.add(layers.Dense(256, activation='relu', input_dim=4 * 4 * 512))
model.add(layers.Dropout(0, 0))
model.add(layers.Dense(1, activation='sigmoid'))

model.compile(optimizer=optimizers.RMSprop(lr=2e-5),
              loss='binary_crossentropy',
              metrics=['acc'])
history = model.fit(train_features, raw_train_labels, epochs=10, batch_size=20,
                    validation_data=(validation_features, raw_validation_labels))

plotter = MetricsPlotter()
plotter.plot_metrics(history)
plotter.show()