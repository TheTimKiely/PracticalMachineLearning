from practicalml.neuralnetworks import *

from practicalml.dl.neuralnetworks import ConvnetDogsVsCats


class ModelFactory(object):

    @classmethod
    def create(cls, ml_config):
        if(ml_config.NnType == 'cnn'):
            network = ConvolutionalNeuralNetwork(ml_config)
        elif(ml_config.NnType == 'math')
            network = MathModel(ml_config)
        elif(ml_config.NnType == 'rnn'):
            network = RecurrentNeuralNetwork(ml_config)
        elif(ml_config.NnType == 'DvsC'):
            network = ConvnetDogsVsCats(ml_config)
        elif(ml_config.NnType == 'lstm'):
            network = LstmRNN(ml_config)
        else:
            raise TypeError(f'Network type {ml_config.NnType} is not defined.')
        return network