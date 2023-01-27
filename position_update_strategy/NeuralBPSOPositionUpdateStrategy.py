import abc
from typing import List

from neural_component_to_position_visitor.ComponentToPositionStandardVisitor import ComponentToPositionStandardVisitor
from position_update_strategy.PositionUpdateStrategy import PositionUpdateStrategy
from velocity_component.VelocityComponent import VelocityComponent


class NeuralBPSOPositionUpdateStrategy(PositionUpdateStrategy, abc.ABC):
    pass


# In the NeuralBPSO the standard position update strategy is to either convert the component to a position, or to add None if the current
# position vector is smaller than the current velocity vector. For conversion, from a velocity component to the position vector,
# the visitor pattern is used, and the exact way of conversion is defined in the component to position visitor.
class NeuralBPSOStandardPositionUpdateStrategy(NeuralBPSOPositionUpdateStrategy):

    def __init__(self, component_to_position_visitor):
        self.component_to_position_visitor = component_to_position_visitor

    def get_new_position(self, current_position, current_velocity: List[VelocityComponent]):
        visitor = self.component_to_position_visitor
        new_position = []
        for i in range(0, len(current_velocity)):
            current_position_placeholder = current_position[i] if i < len(current_position) else None
            new_position_in_dim = current_velocity[i].convert_to_position(current_position_placeholder, visitor)
            if new_position_in_dim is not None:
                new_position.append(new_position_in_dim)
        return new_position
