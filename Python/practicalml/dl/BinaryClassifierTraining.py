import getopt
import sys

epochs = 4
node_count = 4

a = '1,3,4'
n = [int(i) for i in a.split(',')]
print(n)

nodes = []

opts, args = getopt.getopt(sys.argv[1:], shortopts='e:n:')
for opt, arg in opts:
    if(opt == '-e'):
        epochs = int(arg)
    elif(opt == '-n'):
        node_pairs = arg.split(',')
        for pair in node_pairs:
            c, s = pair.split(':')
            nodes.append([int(c), s])


from keras.datasets import imdb
(train_data, train_labels), (test_data, test_labels) = imdb.load_data(num_words=10000)

# word_index is a dictionary mapping words to an integer index
word_index = imdb.get_word_index()
# We reverse it, mapping integer indices to words
reverse_word_index = dict([(value, key) for (key, value) in word_index.items()])
# We decode the review; note that our indices were offset by 3
# because 0, 1 and 2 are reserved indices for "padding", "start of sequence", and "unknown".
decoded_review = ' '.join([reverse_word_index.get(i - 3, '?') for i in train_data[0]])
print(decoded_review)
import numpy as np

'''
def vectorize_sequences(sequences, dimension=10000):
    # Create an all-zero matrix of shape (len(sequences), dimension)
    results = np.zeros((len(sequences), dimension))
    for i, sequence in enumerate(sequences):
        results[i, sequence] = 1.  # set specific indices of results[i] to 1s
    return results
'''
# Our vectorized training data
x_train = vectorize_sequece(train_data)
# Our vectorized test data
x_test = vectorize_sequece(test_data)
# Our vectorized labels
y_train = np.asarray(train_labels).astype('float32')
y_test = np.asarray(test_labels).astype('float32')

from keras import models
from keras import layers


def train_model(node, epochs):
    model = models.Sequential()
    model.add(layers.Dense(node[0], activation='relu', input_shape=(10000,)))
    model.add(layers.Dense(node[0], activation='relu'))
    model.add(layers.Dense(1, activation='sigmoid'))

    model.compile(optimizer='rmsprop',
                  loss='binary_crossentropy',
                  metrics=['accuracy'])

    x_val = x_train[:10000]
    partial_x_train = x_train[10000:]

    y_val = y_train[:10000]
    partial_y_train = y_train[10000:]
    history = model.fit(partial_x_train,
                        partial_y_train,
                        epochs=epochs,
                        batch_size=512,
                        validation_data=(x_val, y_val))

    loss = history.history['loss']
    epochs = range(1, len(loss) + 1)
    loss_series = build_series(epochs, loss, node[1], f'Training Loss {node[0]} nodes')
    plot_add_series(loss_series)


for node in nodes:
    train_model(node, epochs)

plt.legend()
plt.show()

print('done')