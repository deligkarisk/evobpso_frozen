import abc

from evobpso.component_creator.ComponentCreator import ComponentCreator
from evobpso.params.OptimizationParams import OptimizationParams
from evobpso.component_merger.ComponentMerger import ComponentMerger


class VelocityUpdateStrategy(abc.ABC):

    def __init__(self, component_creator: ComponentCreator, component_merger: ComponentMerger, params: OptimizationParams) -> None:
        self.component_creator = component_creator
        self.component_merger = component_merger
        self.params = params
        super().__init__()

    def get_new_velocity(self, current_velocity, current_position, pbest_position, gbest_position):
        raise NotImplementedError
