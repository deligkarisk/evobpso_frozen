import copy

from matplotlib import pyplot as plt
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from evaluator.Evaluator import Evaluator
from utils import data_load_utils


class StandardNNEvaluator(Evaluator):

    def evaluate(self, position):
        tf.keras.backend.clear_session()

        decoded_architecture = self.architecture_decoder.decode(position)
        architecture_model = self.model_creator.create_model(architecture=decoded_architecture)
        architecture_model.compile(loss=self.training_params.loss,
                                   optimizer=self.training_params.optimizer,
                                   metrics=self.training_params.metrics)
        (x_train, y_train), (x_val, y_val), (x_test, y_test) = self.data_loader()
        history = architecture_model.fit(x_train, y_train, epochs=self.training_params.epochs, batch_size=self.training_params.batch_size,
                                         validation_data=(x_val, y_val))

        evaluator_data = {'position': copy.deepcopy(position), 'decoded_architecture': copy.deepcopy(decoded_architecture),
                          'architecture_model': copy.deepcopy(architecture_model), 'history': copy.deepcopy(history)}

        evaluation_result = history.history['val_loss'][-1]

        return evaluation_result, evaluator_data






