

class ModelLayer(object):

    def __init__(self, node_count, activation = None, input_shape = ()):
        self._node_count = node_count
        self._activation = activation
        self._input_shape = input_shape


    @property
    def activation(self):
        return self._activation
    @activation.setter
    def activation(self, activation):
        self._activation = activation


    @property
    def node_count(self):
        return self._node_count
    @node_count.setter
    def node_count(self, node_count):
        self._node_count = node_count


    @property
    def input_shape(self):
        return self._input_shape
    @input_shape.setter
    def input_shape(self, input_shape):
        self._input_shape = input_shape