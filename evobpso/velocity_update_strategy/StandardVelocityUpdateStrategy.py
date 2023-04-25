from evobpso.velocity_factor.VelocityFactor import VelocityFactorRemove
from evobpso.velocity_update_strategy.VelocityUpdateStrategy import VelocityUpdateStrategy


class StandardVelocityUpdateStrategy(VelocityUpdateStrategy):

    def get_new_velocity(self, current_velocity, current_position, pbest_position, gbest_position):
        pso_params = self.params.pso_params
        personal_component = self.component_creator.create_component(pbest_position, current_position, pso_params.c1 )
        global_component = self.component_creator.create_component(gbest_position, current_position, pso_params.c2 )
        new_velocity = self.component_merger.merge_personal_and_global_components(personal_component, global_component, self.params.pso_params)
        return new_velocity




