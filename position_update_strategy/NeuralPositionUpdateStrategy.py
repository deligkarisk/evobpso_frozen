import abc

from position_update_strategy.PositionUpdateStrategy import PositionUpdateStrategy
from velocity_component.VelocityComponent import VelocityComponentEvolve, VelocityComponentAdd, VelocityComponentRemove


class NeuralPositionUpdateStrategy(PositionUpdateStrategy, abc.ABC):
    pass


class BooleanPSONeuralPositionUpdateStrategy(NeuralPositionUpdateStrategy):
    def get_new_position(self, current_position, current_velocity):
        new_position = []
        for i in range(0, len(current_velocity)):
            if isinstance(current_velocity[i], VelocityComponentEvolve):
                result = current_velocity[i].data ^ current_position[i]
                new_position.append(result)
            elif isinstance(current_velocity[i], VelocityComponentAdd):
                new_position.append(current_velocity[i].data)
            elif isinstance(current_velocity[i], VelocityComponentRemove):
                pass
        return new_position