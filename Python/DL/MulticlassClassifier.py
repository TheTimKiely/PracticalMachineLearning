from keras.datasets import reuters
from keras import models
from keras import layers
from PracticalMLCore.PracticalMLUtils import *


def evaluate_epochs(partial_X_train, partial_y_train, param, param1, param2):
    history = model.fit(partial_X_train, partial_y_train,
                        epochs=20, batch_size=512, validation_data=(X_val, y_val))
    plot_metrics(history)
    plt.show()
    return 9

(train_data, train_labels), (test_data, test_labels) = reuters.load_data(num_words=10000)

word_index = reuters.get_word_index()

reverse_word_index = dict([(value, key) for (key, value) in word_index.items()])

print_encoded_text(train_data[0], reverse_word_index, train_labels[0])

# Our vectorized training data
X_train = vectorize_sequece(train_data, dimensions=10000)
X_test = vectorize_sequece(test_data, dimensions=10000)

y_train = to_one_hot(train_labels)
y_test = to_one_hot(test_labels)
y_test_a = np.array(test_labels)

model = models.Sequential()
model.add(layers.Dense(64, activation='relu', input_shape=(10000,)))
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(64, activation='softmax'))
model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])

X_val = X_train[:1000]
partial_X_train = X_train[1000:]

y_val = y_train[:1000]
partial_y_train = y_train[1000:]


optimum_epochs = 9
optimum_epochs = evaluate_epochs(partial_X_train, partial_y_train, 20, 512, (X_val, y_val))
#history = model.fit(partial_X_train, partial_y_train,
#                    epochs=optimum_epochs, batch_size=512, validation_data=(X_val, y_val))
#plot_metrics(history)

result = model.evaluate(X_test, y_test)
print(f'Result: {result}')

prediction = model.predict(X_test)
sample_index = 5
sample_text = ""#print_encoded_text(X_test[sample_index], reverse_word_index, y_test[sample_index])
print(f'Prediction for sample {sample_index}: {sample_text}\n\tCategory: {np.argmax(prediction[sample_index])}({max(prediction[sample_index])})')
print('done')