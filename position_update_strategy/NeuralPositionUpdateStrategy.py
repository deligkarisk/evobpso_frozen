import abc

from position_update_strategy.PositionUpdateStrategy import PositionUpdateStrategy
from velocity_component.VelocityComponent import VelocityComponentEvolve, VelocityComponentAdd, VelocityComponentRemove


class NeuralPositionUpdateStrategy(PositionUpdateStrategy, abc.ABC):
    pass


class BooleanPSONeuralPositionUpdateStrategy(NeuralPositionUpdateStrategy):
    def get_new_position(self, current_position, current_velocity):
        new_position = []
        for i in range(0, len(current_velocity)):
            current_position_placeholder = current_position[i] if i < len(current_position) else None
            new_position_in_dim = current_velocity[i].get_new_position(current_position_placeholder)
            if new_position_in_dim is not None:
                new_position.append(new_position_in_dim)
        return new_position
