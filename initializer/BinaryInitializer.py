import random

from initializer.Initializer import Initializer
from utils.utils import create_rnd_binary_vector


class BinaryInitializer(Initializer):
    def get_initial_position(self):

        # get random number to see how many layers this position would have
        num_layers = random.randint(self.architecture.min_layers, self.architecture.max_layers)
        position = []
        for i in range(0, num_layers):
            position.append(create_rnd_binary_vector(0.5, self.params.n_bits))
        return position

