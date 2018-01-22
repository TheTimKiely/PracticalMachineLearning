
class ModelMetrics:

    def __init__(self):
        _loss = None
        _accuracy = None
        _history = None

    @property
    def loss(self):
        return self._loss

    @loss.setter
    def loss(self, loss):
        self._loss = loss

