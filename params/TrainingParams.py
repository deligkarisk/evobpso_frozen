
class TrainingParams:
    def __init__(self, batch_size, epochs, loss, optimizer, metrics):
        self.batch_size = batch_size
        self.epochs = epochs
        self.loss = loss
        self.optimizer = optimizer
        self.metrics = metrics
