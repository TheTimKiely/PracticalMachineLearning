import numpy as np
from keras import models
from keras import layers

class MLConfig(object):
    def __init__(self, nn_type, mode, layers = 4, nodes = 16, epochs = 10, batch_size = 32, verbose = False):
        self._verbose = verbose
        self._nn_type = nn_type
        # Properties probably aren't necessary, so experimenting with public fields
        self.Layers = layers
        self.Nodes = nodes
        self._epochs = epochs
        self.BatchSize = batch_size
        self.TrainDir = ''
        self.TestDir = ''
        self.ValidationDir = ''
        self.Mode = mode
        self.ModelConfig = ModelParameters()

    @property
    def Verbose(self):
        return self._verbose

    @property
    def NnType(self):
        return self._nn_type

    @NnType.setter
    def NnType(self, nn_type):
        self._nn_type = nn_type

    @property
    def Epochs(self):
        return self._epochs


class ModelParameters(object):
    @classmethod
    def build_model(cls, model_parameters):
        model = models.Sequential()
        for i in range(len(model_parameters.layers)):
            layer_params = model_parameters.layers[i]
            model.add(layers.Dense(layer_params.node_count,
                                   activation=layer_params.activation,
                                   input_shape=layer_params.input_shape))
        model.compile(optimizer=model_parameters.optimizer, loss=model_parameters.loss_function, metrics=model_parameters.metrics)
        return model

    def __init__(self):
        self._metrics = []
        self._layers = []
        self.LayerCount = 1
        self.Dropout = 0
        self.RecurrentDropout = 0


    @property
    def loss_function(self):
        return self._loss_function

    @loss_function.setter
    def loss_function(self, loss_function):
        self._loss_function = loss_function

    @property
    def optimizer(self):
        return self._optimizer

    @optimizer.setter
    def optimizer(self, optimizer):
        self._optimizer = optimizer

    @property
    def epochs(self):
        return self._epochs

    @epochs.setter
    def epochs(self, epochs):
        self._epochs = epochs

    @property
    def batch_size(self):
        return self._batch_size

    @batch_size.setter
    def batch_size(self, batch_size):
        self._batch_size = batch_size

    @property
    def validation_data(self):
        return self._validation_data
    @validation_data.setter
    def validation_data(self, validation_data):
        self._validation_data = validation_data

    @property
    def metrics(self):
        return self._metrics
    @metrics.setter
    def metrics(self, metrics):
        self._metrics = metrics

    @property
    def layers(self):
        return self._layers
    @layers.setter
    def layers(self, layers):
        self._layers = layers
