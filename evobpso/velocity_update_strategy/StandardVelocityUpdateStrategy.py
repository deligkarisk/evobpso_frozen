from typing import List

from evobpso.component_creator.ComponentCreator import ComponentCreator
from evobpso.component_merger.ComponentMerger import ComponentMerger
from evobpso.params.OptimizationParams import OptimizationParams
from evobpso.velocity_factor.VelocityFactor import VelocityFactorRemove
from evobpso.velocity_update_extension.VelocityUpdateExtension import VelocityUpdateExtension
from evobpso.velocity_update_strategy.VelocityUpdateStrategy import VelocityUpdateStrategy


class StandardVelocityUpdateStrategy(VelocityUpdateStrategy):

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
        pso_params = self.params.pso_params
        personal_component = self.component_creator.create_component(pbest_position, current_position, pso_params.c1 )
        global_component = self.component_creator.create_component(gbest_position, current_position, pso_params.c2 )
        new_velocity = self.component_merger.merge_personal_and_global_components(personal_component, global_component, self.params.pso_params)

        for extension in self.velocity_update_extensions:
            new_velocity = extension.get_new_velocity(new_velocity)

        return new_velocity



