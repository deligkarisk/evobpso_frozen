import abc
from typing import List

from position_update_strategy.PositionUpdateStrategy import PositionUpdateStrategy
from velocity_factor.VelocityFactor import VelocityFactor, VelocityFactorEvolve, VelocityFactorAdd, VelocityFactorRemove


class NeuralRealPSOPositionUpdateStrategy(PositionUpdateStrategy, abc.ABC):
    @abc.abstractmethod
    def get_new_position(self, current_position, current_velocity: List[VelocityFactor]):
        raise NotImplementedError


class NeuralRealPSOStandardPositionUpdateStrategy(NeuralRealPSOPositionUpdateStrategy):

    def get_new_position(self, current_position, current_velocity: List[VelocityFactor]):
        pass

    def _update_to_position(self, current_velocity, current_position):
        pass
