import abc
import random
from typing import List

from evobpso.position_update_strategy.PositionUpdateStrategy import PositionUpdateStrategy
from evobpso.utils.utils import create_rnd_binary_vector
from evobpso.velocity_factor.VelocityFactor import VelocityFactor, VelocityFactorEvolve, VelocityFactorAdd, VelocityFactorRemove


class StandardPositionUpdateStrategy(PositionUpdateStrategy):

    def get_new_position(self, current_position, current_velocity: List[VelocityFactor]):
        new_position = []
        for i in range(0, len(current_velocity)):
            current_position_placeholder = current_position[i] if i < len(current_position) else None
            new_position_in_dim = self._update_to_position(current_velocity[i], current_position_placeholder)
            if new_position_in_dim is not None:
                new_position.append(new_position_in_dim)
        return new_position

    def _update_to_position(self, current_velocity, current_position):

        if isinstance(current_velocity, VelocityFactorEvolve):
            if current_position is None:
                raise Exception("Attempt to XOR a velocity component with an empty position.")
            updated_position = current_position ^ current_velocity.data
        elif isinstance(current_velocity, VelocityFactorAdd):
            updated_position = current_velocity.data
        elif isinstance(current_velocity, VelocityFactorRemove):
            updated_position = None
        else:
            raise Exception("Unknown velocity component during position update.")

        return updated_position





