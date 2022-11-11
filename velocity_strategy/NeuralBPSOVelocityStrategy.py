import abc
from functools import partial

from utils import find_largest_size, create_rnd_binary_vector, find_smallest_size, find_largest_index
from velocity_strategy.VelocityStrategy import VelocityStrategy
from velocity_component.VelocityComponent import VelocityComponentRemove, VelocityComponentAdd, VelocityComponentEvolve, VelocityComponentProcessor


class NeuralBPSOVelocityStrategy(VelocityStrategy, abc.ABC):
    pass


class NeuralBPSOStandardNeuralVelocityStrategy(NeuralBPSOVelocityStrategy):

    def __init__(self, processor=None):
        self.processor = processor

    def get_new_velocity(self, current_velocity, current_position, pbest_position, gbest_position, params):

        # check for singleton processor
        if self.processor is None:
            self.processor = VelocityComponentProcessor(params)


        rnd_vector_personal_partial = partial(create_rnd_binary_vector, params.c1, params.n_bits)
        personal_component = self._create_component(pbest_position, current_position, rnd_vector_personal_partial)
        rnd_vector_social_partial = partial(create_rnd_binary_vector, params.c2, params.n_bits)
        global_component = self._create_component(gbest_position, current_position, rnd_vector_social_partial)
        personal_component, global_component = self._equalize_sizes(personal_component, global_component)
        new_velocity = self._merge_personal_and_global_components(personal_component, global_component)
        return new_velocity

    def _create_component(self, best_position, current_position, rnd_vector_partial):
        largest_size = find_largest_size(best_position, current_position)
        smallest_size = find_smallest_size(best_position, current_position)
        best_position_is_larger = find_largest_index(best_position, current_position) == 0
        result = []

        # for the dimensions that both positions have, produce the xor result
        for current_index in range(0, smallest_size):
            component = (best_position[current_index] ^ current_position[current_index]) & rnd_vector_partial()
            velocity_entry = VelocityComponentEvolve(data=component, processor=self.processor)
            result.append(velocity_entry)

        # subsequently, fill the rest with either 'Add' or 'Remove'.
        # this depends on whether the best (global or personal) is larger than the current position or vice versa
        for current_index in range(smallest_size, largest_size):
            if best_position_is_larger:
                velocity_entry = VelocityComponentAdd(data=best_position[current_index], processor=self.processor)
                result.append(velocity_entry)
            else:
                velocity_entry = VelocityComponentRemove(processor=self.processor)
                result.append(velocity_entry)
        return result

    def _equalize_sizes(self, personal_component, global_component):
        p_size = len(personal_component)
        g_size = len(global_component)

        # if the sizes are equal, nothing to do
        if p_size == g_size:
            return personal_component, global_component

        if p_size < g_size:
            diff = g_size - p_size
            for i in range(0, diff):
                personal_component.append(VelocityComponentRemove(self.processor))
        else:
            diff = p_size - g_size
            for i in range(0, diff):
                global_component.append(VelocityComponentRemove(self.processor))

        return personal_component, global_component

    def _merge_personal_and_global_components(self, personal_component, global_component):

        if len(personal_component) != len(global_component):
            raise ValueError('input components must have the same size')

        merged_components = []
        for p_entry, g_entry in zip(personal_component, global_component):
            merged_components.append(p_entry.merge(g_entry))

        return merged_components


