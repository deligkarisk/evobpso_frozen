import abc
from typing import List

from evobpso.component_creator.ComponentCreator import ComponentCreator
from evobpso.params.OptimizationParams import OptimizationParams
from evobpso.component_merger.ComponentMerger import ComponentMerger
from evobpso.velocity_update_extension.VelocityUpdateExtension import VelocityUpdateExtension


class VelocityUpdateStrategy(abc.ABC):

    def __init__(self, component_creator: ComponentCreator, component_merger: ComponentMerger, params: OptimizationParams,
                 velocity_update_extensions: List[VelocityUpdateExtension]):
        self.component_creator = component_creator
        self.component_merger = component_merger
        self.params = params

        if velocity_update_extensions is None:
            self.velocity_update_extensions = []
        else:
            self.velocity_update_extensions = velocity_update_extensions

    def get_new_velocity(self, current_position, pbest_position, gbest_position):
        raise NotImplementedError
