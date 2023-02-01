import abc
from typing import List

from position_update_strategy.PositionUpdateStrategy import PositionUpdateStrategy
from velocity_component.VelocityComponent import VelocityComponent, VelocityComponentEvolve, VelocityComponentAdd, VelocityComponentRemove


class NeuralBooleanPSOPositionUpdateStrategy(PositionUpdateStrategy, abc.ABC):
    @abc.abstractmethod
    def get_new_position(self, current_position, current_velocity: List[VelocityComponent]):
        raise NotImplementedError


class NeuralBooleanPSOStandardPositionUpdateStrategy(NeuralBooleanPSOPositionUpdateStrategy):

    def get_new_position(self, current_position, current_velocity: List[VelocityComponent]):
        new_position = []
        for i in range(0, len(current_velocity)):
            current_position_placeholder = current_position[i] if i < len(current_position) else None
            new_position_in_dim = self._update_to_position(current_velocity[i], current_position_placeholder)
            if new_position_in_dim is not None:
                new_position.append(new_position_in_dim)
        return new_position

    def _update_to_position(self, current_velocity, current_position):

        if isinstance(current_velocity, VelocityComponentEvolve):
            if current_position is  None:
                raise Exception("Attempt to XOR a velocity component with an empty position.")
            updated_position = current_position ^ current_velocity.data
        elif isinstance(current_velocity, VelocityComponentAdd):
            updated_position = current_velocity.data
        elif isinstance(current_velocity, VelocityComponentRemove):
            updated_position = None
        else:
            raise Exception("Unknown velocity component during position update.")

        return updated_position



