import abc
from abc import ABC

from position_update_strategy.PositionUpdateStrategy import PositionUpdateStrategy


class BooleanPSOPositionUpdateStrategy(PositionUpdateStrategy, ABC):
    pass


class BooleanPSOStandardPositionUpdateStrategy(BooleanPSOPositionUpdateStrategy):

    def get_new_position(self, current_position, current_velocity):
        new_position = [current_position_in_dim ^ current_velocity_in_dim for (current_position_in_dim, current_velocity_in_dim) in
                        zip(current_position, current_velocity)]
        return new_position
