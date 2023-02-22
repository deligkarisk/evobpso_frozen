from matplotlib import pyplot as plt
from tensorflow import keras
from tensorflow.keras import layers
from evaluator.Evaluator import Evaluator
from utils import data_load_utils


class StandardNNEvaluator(Evaluator):

    def evaluate(self, position):
        decoded_architecture = self.architecture_decoder.decode(position)
        architecture_model = self.model_creator.create_model(architecture=decoded_architecture)
        architecture_model.compile(loss='sparse_categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
        (x_train, y_train), (x_val, y_val), (x_test, y_test) = self.data_loader()
        history = architecture_model.fit(x_train, y_train, epochs=10, batch_size=512,
                                         validation_data=(x_val, y_val))
        loss = history.history["loss"]
        val_loss = history.history["val_loss"]
        epochs = range(1, len(loss) + 1)
        plt.plot(epochs, loss, "bo", label="Training loss")
        plt.plot(epochs, val_loss, "b", label="Validation loss")
        plt.title("Training and validation loss")
        plt.xlabel("Epochs")
        plt.ylabel("Loss")
        plt.legend()
        plt.show()






