import abc
from functools import partial

from utils import utils
from utils.utils import find_largest_size, create_rnd_binary_vector, find_smallest_size, find_largest_index
from velocity_update_strategy.VelocityUpdateStrategy import VelocityUpdateStrategy
from velocity_factor.VelocityFactor import VelocityFactorRemove, VelocityFactorAdd, VelocityFactorEvolve


class StandardVelocityUpdateStrategy(VelocityUpdateStrategy):

    def get_new_velocity(self, current_velocity, current_position, pbest_position, gbest_position):

        pso_params = self.params.pso_params
        #rnd_vector_personal_partial = partial(create_rnd_binary_vector, pso_params.c1, pso_params.n_bits)
        personal_component = self.component_creator.create_component(pbest_position, current_position, pso_params.c1 )
        #personal_component = self._create_component(pbest_position, current_position, rnd_vector_personal_partial)
        #rnd_vector_social_partial = partial(create_rnd_binary_vector, pso_params.c2, pso_params.n_bits)
        global_component = self.component_creator.create_component(gbest_position, current_position, pso_params.c2 )
        #global_component = self._create_component(gbest_position, current_position, rnd_vector_social_partial)
        personal_component, global_component = self._equalize_sizes(personal_component, global_component)
        new_velocity = self._merge_personal_and_global_component(personal_component, global_component)
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

    def _merge_personal_and_global_component(self, personal_component, global_component):
        # this method merges the personal and global components of the velocity equation.
        # it corresponds to the formula personal_component + global_component

        if len(personal_component) != len(global_component):
            raise ValueError('input components must have the same size')

        pso_params = self.params.pso_params
        merged_components = []
        for p_entry, g_entry in zip(personal_component, global_component):
            # if both components are Evolve, then just OR their data
            if (isinstance(personal_component, VelocityFactorEvolve)) and isinstance(global_component, VelocityFactorEvolve):
                result_data = p_entry.data | g_entry.data
                new_entry = VelocityFactorEvolve(data=result_data)
                merged_components.append(new_entry)
            # otherwise, just select one of the components, either the personal or the global
            else:
                new_entry = utils.random_choice(p_entry, g_entry, pso_params.k)
                merged_components.append(new_entry)
        return merged_components
