import os, time
import numpy as np
from keras import models
from keras import layers
from keras import optimizers
from keras.preprocessing.image import ImageDataGenerator
from practicalml.dl import keras_models


class NeuralNetwork(object):
    def __init__(self, ml_config):
        self._layers = ml_config.Layers
        self._nodes = ml_config.Nodes
        self._epochs = ml_config.Epochs
        self.Config = ml_config
        self.X_train = None
        self.y_train = None
        self.X_test = None
        self.y_test = None
        self.Model = None
        self.ModelFile = ''
        self.HistoryFile = 'FitHistory.pkl'
        self.Model_Directory = os.path.abspath(os.path.join(os.getcwd() ,'../../../models'))
        self.Data_Directory = os.path.abspath(os.path.join(os.getcwd(), '../../../data'))


    @staticmethod
    def plot_filter():
        from keras.applications import VGG16
        from keras import backend as K
        model = keras_models.ModelRepository.get_vgg16()


    @staticmethod
    def plot_activation():
        from keras.models import load_model
        from keras.preprocessing import image
        import numpy as np
        model_path = os.path.abspath(os.path.join(os.getcwd() ,"../../../models/DogsVsCats_small_1.h5"))
        model = load_model(model_path)
        print(model.summary())
        img_path =  os.path.abspath(os.path.join(os.getcwd() ,"../../../data/dogs_and_cats/test/dogs/dog.1045.jpg"))
        img = image.load_img(img_path, target_size=(150, 150))
        img_tensor = image.img_to_array(img)
        print(img_tensor.shape)
        img_tensor = np.expand_dims(img_tensor, axis=0)
        print(img_tensor.shape)
        img_tensor /= 255
        import matplotlib.pyplot as plt
        plt.imshow(img_tensor[0])
        #plt.show()
        plt.clf()
        from keras import models
        layer_outputs = [layer.output for layer in model.layers[:8]]
        activation_model =models.Model(inputs=model.input, outputs=layer_outputs)
        activations = activation_model.predict(img_tensor)
        first_layer_activation = activations[0]
        plt.matshow(first_layer_activation[0,:,:,7], cmap='viridis')
        plt.show()
        #text = input('Press any key to continue...')


    '''verbosity levels: s(silence) q(quiet) m(moderate) d(debug) (noisy)'''
    def log(self, msg, verbosity='d'):
        if(self.Config.Verbose == verbosity):
            print(msg)

    def build_model(self):
        pass

    def prepare_data(self):
        pass

    def fit_and_save(self):
        pass

    def evaluate(self):
        pass

    def predict(self):
        pass

class ConvolutionalNeuralNetwork(NeuralNetwork):

    pass


class ConvnetDogsVsCats(NeuralNetwork):

    def __init__(self, ml_config):
        super(ConvnetDogsVsCats, self).__init__(ml_config)
        self.ValidationGenerator = None
        self.TrainGenerator = None
        self.ModelFile = 'DogsVsCats_small_1.h5'

    def build_model(self):
        self.Model = models.Sequential()

        self.Model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(150, 150, 3)))
        self.Model.add(layers.MaxPool2D((2,2)))
        self.Model.add(layers.Conv2D(64, (3, 3), activation='relu'))
        self.Model.add(layers.MaxPool2D((2,2)))
        self.Model.add(layers.Conv2D(128, (3, 3), activation='relu'))
        self.Model.add(layers.MaxPool2D((2,2)))
        self.Model.add(layers.Conv2D(128, (3, 3), activation='relu'))
        self.Model.add(layers.MaxPool2D((2,2)))

        self.Model.add(layers.Flatten())
        self.Model.add(layers.Dense(512, activation='relu'))
        self.Model.add(layers.Dense(1, activation='sigmoid'))

        self.log(self.Model.summary())

        self.Model.compile(loss='binary_crossentropy',
                            optimizer=optimizers.RMSprop(lr=1e-4),
                            metrics=['acc'])


    def prepare_data(self):
        train_datagen = ImageDataGenerator(rescale=1./255)
        test_datagen = ImageDataGenerator(rescale=1./255)
        self.TrainGenerator = train_datagen.flow_from_directory(self.Config.TrainDir,
                                                            target_size=(150, 150),
                                                            batch_size=20,
                                                            class_mode='binary')
        self.ValidationGenerator = test_datagen.flow_from_directory(self.Config.ValidationDir,
                                                                target_size=(150, 150),
                                                                batch_size=20,
                                                                class_mode='binary')
    def fit_and_save(self):
        if(self.TrainGenerator == None) or (self.ValidationGenerator == None):
            self.prepare_data()

        conv_base = None
        if(self.Config.Mode == 'p'):
            start = time.time()
            conv_base = keras_models.ModelRepository().get_vgg16()
            print(f'elapsed: {time.time() - start}')

        history = self._model.fit_generator(self.TrainGenerator,
                                            steps_per_epoch=100,
                                            epochs=30,
                                            validation_data=self.ValidationGenerator,
                                            validation_steps=50)

        self._model.save(self.ModelFile)

        '''
        if(self.Config.Verbose == True):
            with open(self.HistoryFile, 'wb') as file_handle:
                dill.dump(history, file_handle)
        '''

    def create_dir(base_dir, name):
        new_dir = os.path.join(base_dir, name)
        if(os.path.isdir(new_dir) == False):
            os.mkdir(new_dir)
        return new_dir

    def copy_files(file_name, index_range, src_dir, dest_dir):
        fnames = ['{}.{}.jpg'.format(file_name, i) for i in index_range]
        for fname in fnames:
            src = os.path.join(src_dir, fname)
            dest = os.path.join(dest_dir, fname)
            shutil.copyfile(src, dest)
        print(f'Copied {len(fnames)} files from {src_dir} to {dest_dir}')

    def copy_image_files(src, dest):
        if(os.path.isdir(dest) == False):
            os.mkdir(dest)

        train_dir = create_dir(dest, 'train')
        val_dir = create_dir(dest, 'validation')
        test_dir = create_dir(dest, 'test')
        train_cats_dir = create_dir(train_dir, 'cats')
        train_dogs_dir = create_dir(train_dir, 'dogs')
        val_cats_dir = create_dir(val_dir, 'cats')
        val_dogs_dir = create_dir(val_dir, 'dogs')
        test_cats_dir = create_dir(test_dir, 'cats')
        test_dogs_dir = create_dir(test_dir, 'dogs')

        copy_files('dog', range(1000), src, train_dogs_dir)
        copy_files('dog', range(1000, 1500), 'D:\code\ML\data\dogs_and_cats\kaggle\\train\dogs', test_dogs_dir)
        copy_files('dog', range(1500, 2000), 'D:\code\ML\data\dogs_and_cats\kaggle\\train\dogs', val_dogs_dir)


        copy_files('cat', range(1000), 'D:\code\ML\data\dogs_and_cats\kaggle\\train\cats', train_cats_dir)
        copy_files('cat', range(1000, 1500), 'D:\code\ML\data\dogs_and_cats\kaggle\\train\cats', test_cats_dir)
        copy_files('cat', range(1500, 2000), 'D:\code\ML\data\dogs_and_cats\kaggle\\train\cats', val_cats_dir)


class RecurrentNeuralNetwork(NeuralNetwork):

    def __init__(self, ml_config):
        super(RecurrentNeuralNetwork, self).__init__(ml_config)
        self._max_len = 20
        self._max_features = 10000
        self._max_words = 10000
        self._training_samples = 200
        self._validation_samples = 10000
        self.ModelFile = 'ImdbModel.h5'

    def get_text_data(self):
        train_dir = os.path.join(self.Data_Directory, 'imdb\\aclImdb\\train')

        labels = []
        texts = []

        for label_type in ['neg', 'pos']:
            dir_name = os.path.join(train_dir, label_type)
            for fname in os.listdir(dir_name):
                if (fname[-4:] == '.txt'):
                    data_file = os.path.join(dir_name, fname)
                    self.log(f'Opening {data_file}', 'n')
                    with open(data_file, encoding='utf8') as f:
                        texts.append(f.read())
                    f.close()
                    if label_type == 'neg':
                        labels.append(0)
                    else:
                        labels.append(1)
        return texts, labels

    def tokenize_raw_data(self, raw_texts):
        from keras.preprocessing.text import Tokenizer

        tokenizer = Tokenizer(num_words=self._max_words)
        tokenizer.fit_on_texts(raw_texts)
        sequences = tokenizer.texts_to_sequences(raw_texts)
        word_index = tokenizer.word_index

        self.log(f'Found {len(word_index)} unique tokens')
        return sequences, word_index

    def get_embeddings(self):
        glove_dir = os.path.join(self.Data_Directory, 'wordEmbeddings\GloVe')
        glove_file = os.path.join(glove_dir, 'glove.6B.100d.txt')
        self.log(f'Reading word embeddings from {glove_dir}', 'd')
        embeddings_index = {}
        with open(glove_file, encoding='utf8') as f:
            for line in f:
                values = line.split()
                word = values[0]
                coefs = np.asarray(values[1:], dtype='float32')
                embeddings_index[word] = coefs
        f.close()
        print(f'Found {len(embeddings_index)} word vectors.', 'd')
        return embeddings_index

    def build_embedding_matrix(self, words, embeddings):
        embeding_dim = 100
        embeding_matrix = np.zeros((self._max_words, embeding_dim))
        for word, i in words.items():
            if i < self._max_words:
                embedding_vector = embeddings.get(word)
                if(embedding_vector is not None):
                    embeding_matrix[i] = embedding_vector

        return embeding_matrix

    def prepare_data(self):
        from keras.preprocessing.sequence import pad_sequences
        import os

        max_len = 100

        # Get X and y (training data and labels) from imdb dataset
        raw_texts, raw_labels = self.get_text_data()
        tokens, word_index = self.tokenize_raw_data(raw_texts)
        data = pad_sequences(tokens, maxlen=max_len)
        labels = np.asarray(raw_labels)
        self.log(f'Shape of X: {data.shape}')
        self.log(f'Shape of y: {labels.shape}')

        # Shuffle data so that positive and negative reviews are not grouped together
        indeces = np.arrange(data.shape[0])
        np.random.shuffle(indeces)
        data = data[indeces]
        labels = labels[indeces]

        # Splice validation set
        X_train = data[:self._training_samples]
        y_train = labels[:self._training_samples]
        X_val = data[self._training_samples:]
        y_val = data[self._training_samples:]

        # Build word embeddings
        embeddings = self.get_embeddings()
        embeddings_matrix = self.build_embedding_matrix(word_index, embeddings)

    def prepare_dataOLD(self):
        from keras.datasets import imdb
        from keras import preprocessing
        (raw_X_train, self.y_train), (raw_X_test, self.y_test) = imdb.load_data()
        self.X_train = preprocessing.sequence.pad_sequences(raw_X_train, maxlen=self._max_len)
        self.X_test = preprocessing.sequence.pad_sequences(raw_X_test, maxlen=self._max_len)

    def build_model(self):
        from keras.models import Sequential
        from keras.layers import Flatten, Dense, Embedding

        self.Model = models.Sequential()
        self.Model.add(Embedding(10000, 8, input_length = self._max_len))
        self.Model.add(Flatten())
        self.Model.add(Dense(1, activation='sigmoid'))
        self.Model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['acc'])

        if(self.Config.Verbose):
            print(self.Model.summary())

    def fit_and_save(self):

        history = self.Model.fit(self.X_train, self.y_train, epochs=self.Config.Epochs, batch_size=self.Config.BatchSize, validation_split=0.2)
        self.Model.save(self.ModelFile)

def main():
    print('hellow from nn.py')
    NeuralNetwork.plot_activation()

if(__name__ == '__main__'):
    main()