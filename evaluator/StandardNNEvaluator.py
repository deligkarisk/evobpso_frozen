from tensorflow import keras
from tensorflow.keras import layers
from evaluator.Evaluator import Evaluator


class StandardNNEvaluator(Evaluator):

    def evaluate(self, position):
        decoded_position = self.decoder.decode(position)


        # First, rescale inputs
        input = keras.Input(shape=(180, 180, 3))
        x = layers.Rescaling(1./255)(input)
        # Add the subsequent layers
        for layer in decoded_position:
            x = layers.Conv2D(filters=32, kernel_size=3, activation="relu")(x)

        raise NotImplementedError