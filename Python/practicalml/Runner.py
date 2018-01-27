import getopt
import os, sys
import numpy as np

print(os.getcwd())
sys.path.append(os.getcwd())
from core.configuration import *
from core.data_container import DataContainer
from core.entities import *
from core.plotting import *
from dl.neuralnetworks import ConvnetDogsVsCats
from core.utils import ProcessorInfo
from core.factories import ModelFactory
from core.utils import *

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
    metrics = ['acc']
    model_config = ModelParameters()
    opts, args = getopt.getopt(params, shortopts='t:m:l:o:n:d:e:b:s:v:x:')
    for opt, arg in opts:
        if(opt == '-b'):
            batch_size = int(arg)
        elif(opt == '-d'):
            dataset = arg
        elif (opt == '-e'):
            epochs = int(arg)
        elif(opt == '-l'):
            model_config.loss_function = arg
        elif(opt == '-o'):
            model_config.optimizer = arg
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
        elif(opt == 'x'):
            metrics = arg

    ml_config = MLConfig(nn_type, mode, layers, nodes, epochs, batch_size, verbose)
    model_config.metrics = metrics
    ml_config.ModelConfig = model_config
    return dataset, sample_size, ml_config

def prepare_convnet_dogs_and_cats(network):
    network.Config.TestDir = 'd:\code\ml\data\dogs_and_cats\\test'
    network.Config.TrainDir = 'd:\code\ml\data\dogs_and_cats\\train'
    network.Config.ValidationDir = 'd:\code\ml\data\dogs_and_cats\\validation'

    # The first run, copy image files
    copyFiles = False
    if(copyFiles == True):
        network.copy_image_files('D:\code\ML\data\dogs_and_cats\kaggle\\train\dogs', 'd:\code\ml\data\dogs_and_cats')

def climate_prediction(dataset, sample_size, ml_config):
    data_container = DataContainer(dataset)
    data_container.prepare_data(sample_size)
    metrics = []

    '''
    ml_model = ModelFactory.create(ml_config)
    if ml_model.Config.Mode == 't':
        ml_model.build_model(data_container)
        ml_history = ml_model.fit_and_save()
    elif ml_model.Config.Mode == 'p':
        ml_model.load_from_file(data_container)
        ml_history = ml_model.evaluate()
    #ml_history = ml_model.evaluate()
    ml_metrics = ModelMetrics('MlMetrics', ml_history, 'bo')
    metrics.append(ml_metrics)
    '''

    ml_config.NnType = 'gru'
    gru_model = ModelFactory.create('GRU Simple', ml_config)
    gru_model.build_model(data_container)
    gru_history = gru_model.fit_and_save()
    gru_metrics = ModelMetrics('GruMetrics', gru_history, ('b', 'r'))
    metrics.append(gru_metrics)


    ml_config.ModelConfig.Dropout = 0.2
    ml_config.ModelConfig.RecurrentDropout = 0.2
    gru_dropout_model = ModelFactory.create('GRU Dropout', ml_config)
    gru_dropout_model.build_model(data_container)
    gru_dropout_history = gru_dropout_model.fit_and_save()
    gru_dropout_metrics = ModelMetrics('GruDropoutMetrics', gru_dropout_history, ('g', 'c'))
    metrics.append(gru_dropout_metrics)

    ml_config.ModelConfig.LayerCount = 2
    gru_two_layers_model = ModelFactory.create('GRU 2 Layers', ml_config)
    gru_two_layers_model.build_model(data_container)
    gru_two_layers_history = gru_two_layers_model.fit_and_save()
    gru_two_layers_metrics = ModelMetrics('Gru2LayersMetrics', gru_two_layers_history, ('k','m'))
    metrics.append(gru_two_layers_metrics)

    plotter = MetricsPlotter()
    plotter.plot_histories(metrics, ((0, ('loss', 'val_loss')),))
    plotter.save('Test.png')
    plotter.show()


def main(params):
    print(f'Running main with args: {params}')
    dataset, sample_size, ml_config = parse_command_line(params)
    ml_config.Instrumentation = Instrumentation()
    ml_config.Instrumentation.Timer.start()
    climate_prediction(dataset, sample_size, ml_config)
    print(f'Time: {ml_config.Instrumentation.Timer.get_split()}')
    exit()

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
    cnn_params = ['-t', 'DvsC', '-m', 'p', '-e', '5', '-n', '64', '-b', 32, '-v', 'd']
    rnn_params = ['-t', 'rnn', '-m', 'p', '-e', '10', '-n', '64', '-b', 32, '-v', 'd']
    lstm_params = ['-t', 'lstm', '-m', 't', '-e', '10','-n', '64', '-b', 32, '-v', 'd']
    climate_lstm_params = ['-s', '200000', '-d', 'jena_climate', '-t', 'lstm', '-m', 't', '-e', '10', '-n',
                           '64', '-b', 32, '-v', 'd']
    climate_ml_params = ['-m', 't', '-s', '200000', '-d', 'jena_climate', '-t', 'ml', '-l', 'mae', '-o', 'rmsprop',
                         '-e', '10', '-n',
                           '64', '-b', 32, '-v', 'd']
    climate_math_params = ['-s', '200000', '-d', 'jena_climate', '-t', 'math', '-m', 't', '-e', '10', '-n', '64', '-b', 32, '-v', 'd']
    main(climate_ml_params)