from dl import *
from dl.neuralnetworks import ConvolutionalNeuralNetwork
from dl.neuralnetworks import RecurrentNeuralNetwork

from practicalml.dl.neuralnetworks import ConvnetDogsVsCats


class NetworkFactory(object):

    @classmethod
    def create(cls, ml_config):
        if(ml_config.NnType == 'cnn'):
            network = ConvolutionalNeuralNetwork(ml_config)
        elif(ml_config.NnType == 'rnn'):
            network = RecurrentNeuralNetwork(ml_config)
        elif(ml_config.NnType == 'DvsC'):
            network = ConvnetDogsVsCats(ml_config)
        return network