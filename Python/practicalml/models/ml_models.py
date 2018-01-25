from practicalml.core.entities import MLEntity

class MathModel(MLEntity):
    def __init__(self, ml_config):
        self.Config = ml_config

    def evaluate(self, data_container):
        batch_maes = []
        for step in range(data_container.val_steps):
            samples, targets = next(data_container.val_generator)

class MLModel(MLEntity):
    def __init__(self, ml_config):
        super(MLModel, self).__init__(ml_config)
