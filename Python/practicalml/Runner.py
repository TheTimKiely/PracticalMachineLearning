import getopt
import sys
import numpy as np
from practicalml.core.data_container import DataContainer
import practicalml, dl
from practicalml.core.plotting import *
from practicalml.dl.neuralnetworks import ConvnetDogsVsCats
from practicalml.core.utils import ProcessorInfo
from practicalml.core.factories import ModelFactory

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

    @property
    def Verbose(self):
        return self._verbose

    @property
    def NnType(self):
        return self._nn_type

    @property
    def Epochs(self):
        return self._epochs


def parse_command_line(params):
    layers = 3
    nodes = 16
    epochs = 10
    batch_size = 64
    nn_type = 'cnn'
    mode = 'p'
    verbose = 'q'
    sample_size = 0
    dataset = ''
    opts, args = getopt.getopt(params, shortopts='t:m:l:n:d:e:b:s:v:')
    for opt, arg in opts:
        if(opt == '-b'):
            batch_size = int(arg)
        elif(opt == '-d'):
            dataset = arg
        elif (opt == '-e'):
            epochs = int(arg)
        elif(opt == '-l'):
            layers = int(arg)
        elif(opt == '-m'):
            mode = arg
        elif(opt == '-n'):
            nodes = int(arg)
        elif(opt == '-s'):
            sample_size = int(arg)
        elif (opt == '-t'):
            nn_type = arg
        elif(opt == '-v'):
            verbose = arg

    return dataset, sample_size, MLConfig(nn_type, mode, layers, nodes, epochs, batch_size, verbose)

def prepare_convnet_dogs_and_cats(network):
    network.Config.TestDir = 'd:\code\ml\data\dogs_and_cats\\test'
    network.Config.TrainDir = 'd:\code\ml\data\dogs_and_cats\\train'
    network.Config.ValidationDir = 'd:\code\ml\data\dogs_and_cats\\validation'

    # The first run, copy image files
    copyFiles = False
    if(copyFiles == True):
        network.copy_image_files('D:\code\ML\data\dogs_and_cats\kaggle\\train\dogs', 'd:\code\ml\data\dogs_and_cats')

def main(params):
    print(f'Running main with args: {params}')
    dataset, sample_size, ml_config = parse_command_line(params)
    network = ModelFactory.create(ml_config)

    if(isinstance(network, practicalml.dl.neuralnetworks.ConvnetDogsVsCats)):
        prepare_convnet_dogs_and_cats(network)


    if(isinstance(network, practicalml.dl.neuralnetworks.NeuralNetwork)):
        process_dl_model(dataset, sample_size)
    else:
        process_ml_model(network, dataset, sample_size)

def process_ml_model(network, dataset, sample_size):
    data_container = DataContainer(dataset)
    data_container.prepare_data(sample_size)
    batch_maes = network.evaluate(data_container)
    print(np.mean(batch_maes))

def process_dl_model(network, dataset, sample_size):
    #X_test = network.TestData
    network.prepare_data(dataset, sample_size)
    network.build_model()
    history = network.fit_and_save()
    plotter = MetricsPlotter()
    plotter.plot_metrics(history)
    plotter.show()
    #history = network.evaluate()
    X_test, y_test = network.TestData
    prediction = network.predict(network.X_test)
    accuracy = y_test - prediction

if(__name__ == '__main__'):
    params = sys.argv[1:]
    # overwrite params for specific tests
    cnn_params = ['-t', 'DvsC', '-m', 'p', '-e', '5', '-l', '3', '-n', '64', '-b', 32, '-v', 'd']
    rnn_params = ['-t', 'rnn', '-m', 'p', '-e', '10', '-l', '3', '-n', '64', '-b', 32, '-v', 'd']
    lstm_params = ['-t', 'lstm', '-m', 't', '-e', '10', '-l', '3', '-n', '64', '-b', 32, '-v', 'd']
    climate_lstm_params = ['-s', '200000', '-d', 'jena_climate', '-t', 'lstm', '-m', 't', '-e', '10', '-l', '3', '-n',
                           '64', '-b', 32, '-v', 'd']
    climate_ml_params = ['-s', '200000', '-d', 'jena_climate', '-t', 'ml', '-m', 't', '-e', '10', '-l', '3', '-n',
                           '64', '-b', 32, '-v', 'd']
    climate_math_params = ['-s', '200000', '-d', 'jena_climate', '-t', 'math', '-m', 't', '-e', '10', '-l', '3', '-n', '64', '-b', 32, '-v', 'd']
    main(climate_math_params)