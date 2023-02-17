import abc

from component_creator.ComponentCreator import ComponentCreator
from params.OptimizationParams import OptimizationParams


class VelocityUpdateStrategy(abc.ABC):

    def __init__(self, component_creator: ComponentCreator, params: OptimizationParams) -> None:
        self.component_creator = component_creator
        self.params = params
        super().__init__()

    def get_new_velocity(self, current_velocity, current_position, pbest_position, gbest_position):
        raise NotImplementedError
