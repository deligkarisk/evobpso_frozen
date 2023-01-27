import abc
from abc import ABC

from position_update_strategy.PositionUpdateStrategy import PositionUpdateStrategy


class RealPSOPositionUpdateStrategy(PositionUpdateStrategy, ABC):
    pass


# The real pso standard position update algorithm adds the current position to the current velocity
class RealPSOStandardPositionUpdateStrategy(RealPSOPositionUpdateStrategy):

    def get_new_position(self, current_position, current_velocity):
        new_position = [current_velocity_in_dim + current_position_in_dim for
                        (current_velocity_in_dim, current_position_in_dim) in zip(current_velocity, current_position)]