import random

from evobpso.position_update_strategy.PositionUpdateStrategy import PositionUpdateStrategy
from evobpso.utils.utils import create_rnd_binary_vector


class BooleanRandomPositionUpdateStrategy(PositionUpdateStrategy):

    def get_new_position(self, current_position, current_velocity):
        # get random number to see how many layers this position would have
        num_layers = random.randint(self.optimization_params.neural_architecture_params.min_layers,
                                    self.optimization_params.neural_architecture_params.max_layers)
        position = []
        for i in range(0, num_layers):
            position.append(create_rnd_binary_vector(0.5, self.optimization_params.pso_params.n_bits))
        return position
