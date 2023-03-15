
class TrainingParams:
    def __init__(self, batch_size, train_eval_epochs, best_solution_training_epochs, loss, optimizer, metrics):
        self.batch_size = batch_size
        self.train_eval_epochs = train_eval_epochs
        self.best_solution_training_epochs = best_solution_training_epochs
        self.loss = loss
        self.optimizer = optimizer
        self.metrics = metrics
