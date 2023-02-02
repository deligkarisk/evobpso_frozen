import random

from initializer.Initializer import Initializer


class RealInitializer(Initializer):
    def get_initial_position(self):

        # get random number to see how many layers this position would have
        num_layers = random.randint(self.architecture.min_layers, self.architecture.max_layers)
        position = []
        for i in range(0, num_layers):
            position.append(random.uniform(self.params.min_value, self.params.max_value))
        return position

