from keras.datasets import reuters

from PracticalMLCore.PracticalMLUtils import *

(train_data, train_labels), (test_data, test_labels) = reuters.load_data()

word_index = reuters.get_word_index()

reverse_word_index = dict([(value, key) for (key, value) in word_index.items()])

print_encoded_text(train_data[0], reverse_word_index, train_labels[0])

X_train = vectorize_sequece(train_data)
X_text = vectorize_sequece(test_data)