import abc
from typing import List

from position_update_strategy.PositionUpdateStrategy import PositionUpdateStrategy
from velocity_component.VelocityComponent import VelocityComponent, VelocityComponentEvolve, VelocityComponentAdd, VelocityComponentRemove


class NeuralRealPSOPositionUpdateStrategy(PositionUpdateStrategy, abc.ABC):
    pass


class NeuralRealPSOStandardPositionUpdateStrategy(NeuralRealPSOPositionUpdateStrategy):

    def get_new_position(self, current_position, current_velocity: List[VelocityComponent]):
        pass

    def _update_to_position(self, current_velocity, current_position):
        pass
