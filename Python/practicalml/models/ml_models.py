from practicalml.core.entities import MLEntity
import numpy as np

class MathModel(MLEntity):
    def __init__(self, ml_config):
        self.Config = ml_config

    def evaluate(self, data_container):
        self.log(f'MathModel.evaluate(): steps: {data_container.val_steps}')
        batch_maes = []
        for step in range(data_container.val_steps):
            samples, targets = next(data_container.val_generator)
            preds = samples[:, -1, 1]
            mae = np.mean(np.abs(preds - targets))
            batch_maes.append(mae)
        print(np.mean(batch_maes))
        return batch_maes

class MLModel(MLEntity):
    def __init__(self, ml_config):
        super(MLModel, self).__init__(ml_config)
