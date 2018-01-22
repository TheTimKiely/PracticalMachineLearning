import getopt
import sys
import practicalml, dl
from dl.neuralnetworks import ConvnetDogsVsCats
from core.PracticalMLUtils import MetricsPlotter
from dl.factories import NetworkFactory

class MLConfig(object):
    def __init__(self, nn_type, mode, layers = 4, nodes = 16, epochs = 10, verbose = False):
        self._verbose = verbose
        self._nn_type = nn_type
        # Properties probably aren't necessary, so experimenting with public fields
        self.Layers = layers
        self.Nodes = nodes
        self._epochs = epochs
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
    nn_type = 'cnn'
    mode = 'p'
    verbose = False
    opts, args = getopt.getopt(params, shortopts='t:m:l:n:e:v:')
    for opt, arg in opts:
        if (opt == '-e'):
            epochs = int(arg)
        elif(opt == '-l'):
            layers = int(arg)
        elif(opt == '-m'):
            mode = arg
        elif(opt == '-n'):
            nodes = int(arg)
        elif (opt == '-t'):
            nn_type = arg
        elif(opt == '-v'):
            verbose = bool(opt)

    return MLConfig(nn_type, mode, layers, nodes, epochs, verbose)

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
    ml_config = parse_command_line(params)
    network = NetworkFactory.create(ml_config)

    if(isinstance(network, practicalml.dl.neuralnetworks.ConvnetDogsVsCats)):
        prepare_convnet_dogs_and_cats(network)

    network.build_model()
    network.prepare_data()
    history = network.fit_and_save()
    plotter = MetricsPlotter()
    plotter.plot_metrics(history)
    history = network.evaluate()
    prediction = network.predict()

if(__name__ == '__main__'):
    params = sys.argv[1:]
    # overwrite params for specific tests
    params = ['-t', 'DvsC', '-m', 'p', '-e', '5', '-l', '3', '-n', '64', '-v', 'True']
    main(params)