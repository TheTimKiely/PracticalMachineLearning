from dl.neuralnetworks import *
from models.ml_models import *

class ModelFactory(object):

    @classmethod
    def create(cls, name, ml_config):
        if(ml_config.NnType == 'cnn'):
            network = ConvolutionalNeuralNetwork( name,ml_config)
        elif(ml_config.NnType == 'math'):
            network = MathModel( name,ml_config)
        elif (ml_config.NnType == 'ml'):
            network = MLModel( name,ml_config)
        elif(ml_config.NnType == 'rnn'):
            network = RecurrentNeuralNetwork( name,ml_config)
        elif(ml_config.NnType == 'DvsC'):
            network = ConvnetDogsVsCats( name,ml_config)
        elif(ml_config.NnType == 'lstm'):
            network = LstmRNN( name,ml_config)
        elif(ml_config.NnType == 'gru'):
            network = GruNN( name,ml_config)
        else:
            raise TypeError(f'Network type {ml_config.NnType} is not defined.')
        return network