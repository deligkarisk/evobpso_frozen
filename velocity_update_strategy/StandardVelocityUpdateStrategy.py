import abc
from functools import partial

from utils import utils
from utils.utils import find_largest_size, create_rnd_binary_vector, find_smallest_size, find_largest_index
from velocity_update_strategy.VelocityUpdateStrategy import VelocityUpdateStrategy
from velocity_factor.VelocityFactor import VelocityFactorRemove, VelocityFactorAdd, VelocityFactorEvolve


class StandardVelocityUpdateStrategy(VelocityUpdateStrategy):

    def get_new_velocity(self, current_velocity, current_position, pbest_position, gbest_position):

        pso_params = self.params.pso_params
        personal_component = self.component_creator.create_component(pbest_position, current_position, pso_params.c1 )
        global_component = self.component_creator.create_component(gbest_position, current_position, pso_params.c2 )
        personal_component, global_component = self._equalize_sizes(personal_component, global_component)
        new_velocity = self.component_merger.merge_personal_and_global_components(personal_component, global_component, self.params.pso_params)
        return new_velocity

    def _equalize_sizes(self, personal_component, global_component):
        # this method equalizes the sizes of the two lists of components, by adding Remove components in the shortest list
        p_size = len(personal_component)
        g_size = len(global_component)

        # if the sizes are equal, nothing to do
        if p_size == g_size:
            return personal_component, global_component

        if p_size < g_size:
            diff = g_size - p_size
            for i in range(0, diff):
                personal_component.append(VelocityFactorRemove())
        else:
            diff = p_size - g_size
            for i in range(0, diff):
                global_component.append(VelocityFactorRemove())

        return personal_component, global_component


