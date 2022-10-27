import abc
from abc import ABC

from position_update_strategy.PositionUpdateStrategy import PositionUpdateStrategy


class RealPSOPositionUpdateStrategy(PositionUpdateStrategy, ABC):
    pass


class RealPSOStandardPositionUpdateStrategy(RealPSOPositionUpdateStrategy):

    def get_new_position(self, current_position, current_velocity):
        pass