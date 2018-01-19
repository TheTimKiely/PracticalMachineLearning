from DL import *
from DL.ConvolutionalNeuralNetwork import ConvolutionalNeuralNetwork
from DL.RecurrentNeuralNetwork import RecurrentNeuralNetwork


class NetworkFactory(object):

    @classmethod
    def create(cls, nn_type, layers, nodes, epochs):
        if(nn_type == 'cnn'):
            network = ConvolutionalNeuralNetwork(layers, nodes, epochs)
        elif(nn_type == ''):
            network = RecurrentNeuralNetwork(layers, nodes, epochs)
        return network