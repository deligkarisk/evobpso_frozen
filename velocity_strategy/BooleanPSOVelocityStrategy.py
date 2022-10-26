import abc
import random

from utils import create_rnd_binary_vector

class BooleanPSOVelocityStrategy(abc.ABC):
    def get_new_velocity(self, current_velocity, current_position, pbest_position, gbest_position, params):
        raise NotImplementedError

class BooleanPSOStandardVelocityStrategy(BooleanPSOVelocityStrategy):
    def get_new_velocity(self, current_velocity, current_position, pbest_position, gbest_position, params):

        omega_vector = create_rnd_binary_vector(params.omega, params.n_bits)
        c1_vector = create_rnd_binary_vector(params.c1, params.n_bits)
        c2_vector = create_rnd_binary_vector(params.c2, params.n_bits)

        new_velocity = [omega_vector & current_vel |
                        (c1_vector & (pbest ^ current_pos)) |
                        (c2_vector & (gbest ^ current_pos)) for
                        (current_vel, current_pos, pbest, gbest) in
                        zip(current_velocity, current_position, pbest_position,
                            gbest_position)]

        return new_velocity



class BooleanPSOVelocityMutationStrategy(BooleanPSOVelocityStrategy):
    def get_new_velocity(self, current_velocity, current_position, pbest_position, gbest_position, params):
        pass


