import abc
from functools import partial

from utils import utils
from utils.utils import find_largest_size, create_rnd_binary_vector, find_smallest_size, find_largest_index
from velocity_update_strategy.VelocityUpdateStrategy import VelocityStrategy
from velocity_component.VelocityComponent import VelocityComponentRemove, VelocityComponentAdd, VelocityComponentEvolve


class NeuralBooleanPSOVelocityUpdateStrategy(VelocityStrategy, abc.ABC):
    pass


class NeuralBooleanPSOStandardVelocityUpdateStrategy(NeuralBooleanPSOVelocityUpdateStrategy):

    def __init__(self, params):
        self.params = params

    def get_new_velocity(self, current_velocity, current_position, pbest_position, gbest_position):

        params = self.params
        rnd_vector_personal_partial = partial(create_rnd_binary_vector, params.c1, params.n_bits)
        personal_factor = self._create_component(pbest_position, current_position, rnd_vector_personal_partial)
        rnd_vector_social_partial = partial(create_rnd_binary_vector, params.c2, params.n_bits)
        global_factor = self._create_component(gbest_position, current_position, rnd_vector_social_partial)
        personal_factor, global_factor = self._equalize_sizes(personal_factor, global_factor)
        new_velocity = self._merge_personal_and_global_component(personal_factor, global_factor)
        return new_velocity

    def _create_component(self, best_position, current_position, rnd_vector_partial):
        # creates the social and personal components of the velocity equation.
        largest_size = find_largest_size(best_position, current_position)
        smallest_size = find_smallest_size(best_position, current_position)
        best_position_is_larger = find_largest_index(best_position, current_position) == 0
        result = []

        # for the dimensions that both positions have, produce component using the xor and "and" operations
        for current_index in range(0, smallest_size):
            component = (best_position[current_index] ^ current_position[current_index]) & rnd_vector_partial()
            velocity_component = VelocityComponentEvolve(data=component)
            result.append(velocity_component)

        # subsequently, fill the rest with either 'Add' or 'Remove'.
        # this depends on whether the best (global or personal) is larger than the current position or vice versa
        for current_index in range(smallest_size, largest_size):
            if best_position_is_larger:
                velocity_component = VelocityComponentAdd(data=best_position[current_index])
                result.append(velocity_component)
            else:
                velocity_component = VelocityComponentRemove()
                result.append(velocity_component)
        return result

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
                personal_component.append(VelocityComponentRemove())
        else:
            diff = p_size - g_size
            for i in range(0, diff):
                global_component.append(VelocityComponentRemove())

        return personal_component, global_component

    def _merge_personal_and_global_component(self, personal_component, global_component):
        # this method merges the personal and global components of the velocity equation.
        # it corresponds to the formula personal_component + global_component

        if len(personal_component) != len(global_component):
            raise ValueError('input components must have the same size')

        merged_components = []
        for p_entry, g_entry in zip(personal_component, global_component):
            # if both components are Evolve, then just OR their data
            if (isinstance(personal_component, VelocityComponentEvolve)) and isinstance(global_component, VelocityComponentEvolve):
                result_data = p_entry.data | g_entry.data
                new_entry = VelocityComponentEvolve(data=result_data)
                merged_components.append(new_entry)
            # otherwise, just select one of the components, either the personal or the global
            else:
                new_entry = utils.random_choice(p_entry, g_entry, self.params.k)
                merged_components.append(new_entry)
        return merged_components
