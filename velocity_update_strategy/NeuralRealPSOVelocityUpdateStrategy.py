import abc
from functools import partial

from utils import utils
from utils.utils import find_largest_size, create_rnd_binary_vector, find_smallest_size, find_largest_index
from velocity_update_strategy.VelocityUpdateStrategy import VelocityUpdateStrategy
from velocity_component.VelocityComponent import VelocityComponentRemove, VelocityComponentAdd, VelocityComponentEvolve


class NeuralRealPSOVelocityUpdateStrategy(VelocityUpdateStrategy, abc.ABC):
    pass


class NeuralRealPSOStandardVelocityUpdateStrategy(NeuralRealPSOVelocityUpdateStrategy):

    def __init__(self, params):
        self.params = params

    def get_new_velocity(self, current_velocity, current_position, pbest_position, gbest_position):
        pass

    def _create_component(self, best_position, current_position, rnd_vector_partial):
        pass

    def _equalize_sizes(self, personal_component, global_component):
        pass

    def _merge_personal_and_global_component(self, personal_component, global_component):
        pass
