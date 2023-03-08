import abc
import random
from typing import List

from position_update_strategy.PositionUpdateStrategy import PositionUpdateStrategy
from utils.utils import create_rnd_binary_vector
from velocity_factor.VelocityFactor import VelocityFactor, VelocityFactorEvolve, VelocityFactorAdd, VelocityFactorRemove


class StandardPositionUpdateStrategy(PositionUpdateStrategy):

    def get_new_position(self, current_position, current_velocity: List[VelocityFactor]):
        new_position = []
        for i in range(0, len(current_velocity)):
            current_position_placeholder = current_position[i] if i < len(current_position) else None
            new_position_in_dim = self._update_to_position(current_velocity[i], current_position_placeholder)
            if new_position_in_dim is not None:
                new_position.append(new_position_in_dim)

        new_position = self._mutate_position(new_position)
        return new_position

    def _update_to_position(self, current_velocity, current_position):

        if isinstance(current_velocity, VelocityFactorEvolve):
            if current_position is  None:
                raise Exception("Attempt to XOR a velocity component with an empty position.")
            updated_position = current_position ^ current_velocity.data
        elif isinstance(current_velocity, VelocityFactorAdd):
            updated_position = current_velocity.data
        elif isinstance(current_velocity, VelocityFactorRemove):
            updated_position = None
        else:
            raise Exception("Unknown velocity component during position update.")

        return updated_position


    def _mutate_position(self, position):

        mut_prob = self.optimization_params.pso_params.mutation_prob
        if mut_prob > 0:
            new_position = []
            n_bits = self.optimization_params.pso_params.n_bits
            for layer in position:
                bin_vector = create_rnd_binary_vector(prob=mut_prob, n_bits=n_bits)
                new_layer = layer ^ bin_vector
                new_position.append(new_layer)
            return new_position
        else:
            return position




