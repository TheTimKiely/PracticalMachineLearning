from keras import models
from keras import layers
import time
import matplotlib.pyplot as plt


start = time.time()
model = models.Sequential()
model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)))
model.add(layers.MaxPooling2D((2,2)))
model.add(layers.Conv2D(64, (3,3), activation='relu'))
model.add(layers.MaxPooling2D(2,2))
model.add(layers.Conv2D(64, (3,3), activation='relu'))

model.add(layers.Flatten())
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(10, activation='softmax'))

print(model.summary())

from keras.datasets import mnist
from keras.utils import to_categorical

(raw_train_images, raw_train_labels), (raw_test_images, raw_test_labels) = mnist.load_data()

train_samples = 60
train_images = raw_train_images[:train_samples]
train_labels = raw_train_labels[:train_samples]
train_images = train_images.reshape(train_samples, 28, 28, 1)
train_images = train_images.astype('float32') / 255

test_samples = 10
test_images = raw_test_images[:test_samples]
test_labels = raw_test_labels[:test_samples]
test_images = test_images.reshape(test_samples, 28, 28, 1)
test_images = test_images.astype('float32') / 255

train_labels = to_categorical(train_labels)
test_labels = to_categorical(test_labels)

model.compile(optimizer='rmsprop',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

history = model.fit(train_images, train_labels, epochs=5, batch_size=64)

#plot_metrics(history)

test_loss, test_acc = model.evaluate(test_images, test_labels)
print('Accuracy: {} Loss: {}'.format(test_acc, test_loss))

#plt.imshow(raw_test_images[0], cmap=plt.cm.gray_r, interpolation='nearest')
#prediction = model.predict(test_images)
#print(f'Prediction: {prediction}')

print('Time: {}'.format(time.time() - start))