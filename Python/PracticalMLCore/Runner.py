import sys, time, getopt
from DL import NetworkFactory

def parse_command_line(params):
    layers = 3
    nodes = 16
    epochs = 10
    nn_type = 'cnn'
    opts, args = getopt.getopt(params, shortopts='t:l:n:e:')
    for opt, arg in opts:
        if(opt == 'n'):
            nn_type = arg
        elif(opt == '-l'):
            layers = int(arg)
        elif(opt == '-n'):
            nodes = int(arg)
        elif(opt == '-e'):
            epochs = int(arg)
    return (nn_type, layers, nodes, epochs)

def main(params):
    print(f'Running main with args: {params}')
    nn_type, layers, nodes, epochs = parse_command_line(params)
    network = NetworkFactory.NetworkFactory.create(nn_type, layers, nodes, epochs)


if(__name__ == '__main__'):
    main(sys.argv[1:])