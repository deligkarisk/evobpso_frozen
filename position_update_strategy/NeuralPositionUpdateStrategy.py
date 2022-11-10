import abc
from typing import List

from neural_position_component_visitor.ConventionalNeuralPositionUpdateVisitor import ConventionalNeuralPositionUpdateVisitor
from position_update_strategy.PositionUpdateStrategy import PositionUpdateStrategy
from velocity_component.VelocityComponent import VelocityComponent


class NeuralPositionUpdateStrategy(PositionUpdateStrategy, abc.ABC):
    pass


class BooleanPSONeuralPositionUpdateStrategy(NeuralPositionUpdateStrategy):
    def get_new_position(self, current_position, current_velocity: List[VelocityComponent]):

        visitor = ConventionalNeuralPositionUpdateVisitor()
        new_position = []
        for i in range(0, len(current_velocity)):
            current_position_placeholder = current_position[i] if i < len(current_position) else None
            new_position_in_dim = current_velocity[i].convert_to_position(current_position_placeholder, visitor)
            if new_position_in_dim is not None:
                new_position.append(new_position_in_dim)
        return new_position
