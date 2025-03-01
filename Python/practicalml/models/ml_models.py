import os, time, dill
import numpy as np
from keras.models import Sequential, load_model
from keras.layers import Dense, Flatten
from core.entities import MLEntityBase
from keras import layers
from keras.optimizers import RMSprop


# ABS for ML and DL models
class MLModelBase(MLEntityBase):
    def __init__(self, name, ml_config):
        super(MLModelBase, self).__init__()
        self.Name = name
        self.Config = ml_config
        self._model_file_attribute = 0
        self._weights_file_attribute = 0
        self._tokenizer_file_attribute = 0
        self._history_file_attribute = 0
        self._model = None
        self._tokenizer = None
        self.DataContainer = None

    @property
    def Model(self):
        if (self._model == None):
            raise ValueError(f'The model is not initialize and we are in {self.Config.Mode} mode.')

        return self._model

    @Model.setter
    def Model(self, model):
        self._model = model

    def get_file_name(self, directory, file_type, class_name, attribute, extension):
        attribute_divider = ''
        attribute_str = ''
        if (attribute > 0):
            attribute_divider = '_'
            attribute_str = str(attribute)
        if (extension[0] != '.'):
            extension = '.' + extension
        path = os.path.join(directory, f'{file_type}_{class_name}{attribute_divider}{attribute_str}{extension}')
        return path

    '''
    These file attribute should be move to a File class to handle name management (e.g. When we need to find
    a unique name).
    I just wnted to see what Python properties are capable of.
    '''

    @property
    def ModelFileAttribute(self):
        return self._model_file_attribute

    @ModelFileAttribute.setter
    def ModelFileAttribute(self, value):
        self._model_file_attribute = value

    @property
    def WeightsFileAttribute(self):
        return self._weights_file_attribute

    @WeightsFileAttribute.setter
    def WeightsFileAttribute(self, value):
        self._weights_file_attribute = value

    @property
    def HistoryFileAttribute(self):
        return self._history_file_attribute

    @HistoryFileAttribute.setter
    def HistoryFileAttribute(self, value):
        self._history_file_attribute = value

    @property
    def TokenizerFileAttribute(self):
        return self._model_file_attribute

    @TokenizerFileAttribute.setter
    def TokenizerFileAttribute(self, value):
        self._token_file_attribute = value

    @property
    def TokenizerFile(self):
        return self.get_file_name(os.path.join(self.Base_Directory, 'models'), 'Tokenizer', self.__class__.__name__,
                                  self.TokenizerFileAttribute, 'pkl')

    @property
    def WeightsFile(self):
        return self.get_file_name(os.path.join(self.Base_Directory, 'models'),
                                  'Weights', self.__class__.__name__,
                                  self.WeightsFileAttribute, 'h5')

    @property
    def ModelFile(self):
        return self.get_file_name(os.path.join(self.Base_Directory, 'models'),
                                  'Model', self.__class__.__name__,
                                  self.ModelFileAttribute, 'h5')

    @property
    def HistoryFile(self):
        return self.get_file_name(os.path.join(self.Base_Directory, 'models'),
                                  'History', self.__class__.__name__,
                                  self.HistoryFileAttribute, 'h5')

    @property
    def Tokenizer(self):
        if (self._tokenizer == None):
            if (os.path.isfile(self.TokenizerFile) == False):
                raise FileNotFoundError(
                    f'The Tokenizer is null(prepare_data() hasn\'t been called) and there is no tokenizer file at: {tokenizer_file}')
            with open(self.TokenizerFile, 'rb') as file_handle:
                self._tokenizer = dill.load(file_handle)
        return self._tokenizer

    @Tokenizer.setter
    def Tokenizer(self, tokenizer):
        self._tokenizer = tokenizer
        with open(self.TokenizerFile, 'wb') as file_handle:
            dill.dump(self._tokenizer, file_handle, protocol=dill.HIGHEST_PROTOCOL)


    def prepare_data(self):
        pass

    def build_model(self):
        pass

    def load_from_file(self, file_name = None):
        pass

    def fit_and_save(self):
        pass

    def evaluate(self, val_X, val_y):
        pass

    def predict(self, X):
        pass


class MathModel(MLModelBase):
    def __init__(self, ml_config):
        self.Config = ml_config

    def evaluate(self, data_container):
        self.log(f'MathModel.evaluate(): steps: {data_container.val_steps}')
        batch_maes = []
        for step in range(data_container.val_steps):
            samples, targets = next(data_container.val_generator)
            preds = samples[:, -1, 1]
            mae = np.mean(np.abs(preds - targets))
            batch_maes.append(mae)
        print(np.mean(batch_maes))
        return batch_maes

class MLModel(MLModelBase):
    def __init__(self, name, ml_config):
        super(MLModel, self).__init__( name,ml_config)

    # Requires a DataContainer because we might need to know the data shape to initialize layers
    def build_model(self, data_container):
        # Should to a state change to a data-bound model
        self.DataContainer = data_container
        self.Model = Sequential()
        lookback = 1440
        step = 6
        self.Model.add(Flatten(input_shape=(lookback // step, self.DataContainer.Data.shape[-1])))
        self.Model.add(Dense(32, activation='relu'))
        self.Model.add(Dense(1))
        self.Model.compile(optimizer=self.Config.ModelConfig.optimizer,
                           loss=self.Config.ModelConfig.loss_function,
                           metrics=self.Config.ModelConfig.metrics)


    def fit_and_save(self, weights_file=None, model_file=None):
        if self.DataContainer == None:
            raise AttributeError('This model had no DataContainer, please call build_model(data_container) first.')
        if self.DataContainer.train_generator == None:
            raise AttributeError('This model does not have a training generator.  Please check your configuration')

        '''
        lookback = 1440
        step = 6
        model = Sequential()
        model.add(layers.Flatten(input_shape=(lookback // step, self.DataContainer.Data.shape[-1])))
        model.add(layers.Dense(32, activation='relu'))
        model.add(layers.Dense(1))

        model.compile(optimizer=RMSprop(), loss='mae', metrics=['acc'])
        history = model.fit_generator(self.DataContainer.train_generator,
        history = model.fit_generator(self.DataContainer.train_generator,
                                      steps_per_epoch=500,
                                      epochs=self.Config.Epochs,
                                      validation_data=self.DataContainer.val_generator,
                                      validation_steps=self.DataContainer.val_steps)

        '''
        self.log(f'Training model: {self.Name}')
        history = self.Model.fit_generator(self.DataContainer.train_generator,
                       steps_per_epoch=500,
                       epochs=self.Config.Epochs,
                       validation_data=self.DataContainer.val_generator,
                       validation_steps=self.DataContainer.val_steps,
                        verbose=2)

        weights_file = self.unique_file_name(MLModelBase.__dict__['WeightsFile'], MLModelBase.__dict__['WeightsFileAttribute'])
        self.Model.save_weights(weights_file)
        model_file = self.unique_file_name(MLModelBase.__dict__['ModelFile'], MLModelBase.__dict__['ModelFileAttribute'])
        self.Model.save(model_file)
        history_file = self.unique_file_name(MLModelBase.__dict__['HistoryFile'], MLModelBase.__dict__['HistoryFileAttribute'])

        '''
        # Hiting recursion depth exceeded errors
        with open(history_file, 'bw') as f:
            dill.dump(history, f, protocol=dill.HIGHEST_PROTOCOL)
        f.close()
        '''

        return history
        ''' Once file management has been refactored into it's own class
        if weights_file != None:
            self.Model.save_weights(weights_file)
        if model_file != None:
            self.Model.save(model_file)
        '''


    def load_from_file(self, data_container, file_name = None):
        # for testing
        model_file = 'D:\code\ML\models\Model_MLModel.h5'
        self.log(f'Loading model from {model_file}')
        self.Model = load_model(model_file)
        self.log(f'Loaded model: {self.Model.summary()}')
        self.DataContainer = data_container
        self.log('Added DataContainer')

    def evaluate(self, val_X = None, val_y = None):
        loss = None
        if val_X == None:
            self.log(f'X and y were not passed as parameters.  Using DataContainer.')
            loss = self.Model.evaluate_generator(self.DataContainer.val_generator, steps=self.DataContainer.val_steps)
        else:
            loss = self.Model.evaluate(val_X, val_y)
        return loss


