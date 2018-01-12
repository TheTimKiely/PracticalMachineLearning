
class ModelTrainer:
    def train(model, params):
        result = model.fit(params.X, params.y,
                                  epochs=params.epochs,
                                  batch_size=params.batch_size,
                                  validation_data=params.validation_data)
        return result