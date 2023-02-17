from tensorflow import keras
from tensorflow.keras import layers
from evaluator.Evaluator import Evaluator


class StandardNNEvaluator(Evaluator):

    def evaluate(self, position):
        decoded_architecture = self.architecture_decoder.decode(position)
        architecture_model = self.model_creator.create_model(architecture=decoded_architecture)


        # First, rescale inputs
        input = keras.Input(shape=(180, 180, 3))
        x = layers.Rescaling(1./255)(input)
        # Add the subsequent layers

        raise NotImplementedError