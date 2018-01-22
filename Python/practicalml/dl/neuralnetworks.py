import time
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
        self._model = None
        self.ModelFile = ''
        self.HistoryFile = 'FitHistory.pkl'

    @staticmethod
    def plot_activation():
        from keras.models import load_model
        model_path = path.abspath(os.path.join(os.getcwd() ,"../../../data/dogs_and_cats"))



    def log(self, msg):
        if(self.Config.Verbose == True):
            print(self._model.summary())

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
        self._model = models.Sequential()

        self._model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(150, 150, 3)))
        self._model.add(layers.MaxPool2D((2,2)))
        self._model.add(layers.Conv2D(64, (3, 3), activation='relu'))
        self._model.add(layers.MaxPool2D((2,2)))
        self._model.add(layers.Conv2D(128, (3, 3), activation='relu'))
        self._model.add(layers.MaxPool2D((2,2)))
        self._model.add(layers.Conv2D(128, (3, 3), activation='relu'))
        self._model.add(layers.MaxPool2D((2,2)))

        self._model.add(layers.Flatten())
        self._model.add(layers.Dense(512, activation='relu'))
        self._model.add(layers.Dense(1, activation='sigmoid'))

        self.log(self._model.summary())

        self._model.compile(loss='binary_crossentropy',
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
    pass