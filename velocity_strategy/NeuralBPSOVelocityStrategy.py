import abc
from functools import partial

from utils import find_largest_size, create_rnd_binary_vector, find_smallest_size, find_largest_index
from velocity_strategy.VelocityStrategy import VelocityStrategy
from velocity_component.VelocityComponent import VelocityComponentRemove, VelocityComponentAdd, VelocityComponentEvolve, VelocityComponentProcessor


class NeuralBPSOVelocityStrategy(VelocityStrategy, abc.ABC):
    pass


class NeuralBPSOStandardVelocityStrategy(NeuralBPSOVelocityStrategy):

    def __init__(self, processor=None):
        self.processor = processor

    def get_new_velocity(self, current_velocity, current_position, pbest_position, gbest_position, params):

        # check for singleton processor
        if self.processor is None:
            self.processor = VelocityComponentProcessor(params)


        rnd_vector_personal_partial = partial(create_rnd_binary_vector, params.c1, params.n_bits)
        personal_factor = self._create_factor(pbest_position, current_position, rnd_vector_personal_partial)
        rnd_vector_social_partial = partial(create_rnd_binary_vector, params.c2, params.n_bits)
        global_factor = self._create_factor(gbest_position, current_position, rnd_vector_social_partial)
        personal_factor, global_factor = self._equalize_sizes(personal_factor, global_factor)
        new_velocity = self._merge_personal_and_global_components(personal_factor, global_factor)
        return new_velocity

    def _create_factor(self, best_position, current_position, rnd_vector_partial):
        # creates the social and personal factors of the velocity equation.
        # factor = (pbest - current_position) * rnd()
        largest_size = find_largest_size(best_position, current_position)
        smallest_size = find_smallest_size(best_position, current_position)
        best_position_is_larger = find_largest_index(best_position, current_position) == 0
        result = []

        # for the dimensions that both positions have, produce factor using the xor and "and" operations
        for current_index in range(0, smallest_size):
            factor = (best_position[current_index] ^ current_position[current_index]) & rnd_vector_partial()
            velocity_component = VelocityComponentEvolve(data=factor, processor=self.processor)
            result.append(velocity_component)

        # subsequently, fill the rest with either 'Add' or 'Remove'.
        # this depends on whether the best (global or personal) is larger than the current position or vice versa
        for current_index in range(smallest_size, largest_size):
            if best_position_is_larger:
                velocity_component = VelocityComponentAdd(data=best_position[current_index], processor=self.processor)
                result.append(velocity_component)
            else:
                velocity_component = VelocityComponentRemove(processor=self.processor)
                result.append(velocity_component)
        return result

    def _equalize_sizes(self, personal_factor, global_factor):
        # this method equalizes the sizes of the two factors, by adding Remove components in the shortest factor
        p_size = len(personal_factor)
        g_size = len(global_factor)

        # if the sizes are equal, nothing to do
        if p_size == g_size:
            return personal_factor, global_factor

        if p_size < g_size:
            diff = g_size - p_size
            for i in range(0, diff):
                personal_factor.append(VelocityComponentRemove(self.processor))
        else:
            diff = p_size - g_size
            for i in range(0, diff):
                global_factor.append(VelocityComponentRemove(self.processor))

        return personal_factor, global_factor

    def _merge_personal_and_global_components(self, personal_component, global_component):
        # this method merges the personal and global factors of the velocity equation.
        # it corresponds to the formula personal_factor + global_factor
        # where e.g. a factor is equal to (pbest - current_position) * rnd()

        if len(personal_component) != len(global_component):
            raise ValueError('input components must have the same size')

        merged_components = []
        for p_entry, g_entry in zip(personal_component, global_component):
            merged_components.append(p_entry.merge(g_entry))

        return merged_components


