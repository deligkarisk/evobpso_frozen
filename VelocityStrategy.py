import abc
from utils import create_rnd_binary_vector

class VelocityStrategy(abc.ABC):
    def update_velocity(self, current_velocity, current_position, pbest_position, gbest_position,params, n_bits):
        raise NotImplementedError

class BooleanPSOStandardVelocityStrategy(VelocityStrategy):
    def update_velocity(self, current_velocity, current_position, pbest_position, gbest_position,params, n_bits):

        omega_vector = create_rnd_binary_vector(params.omega, n_bits)
        c1_vector = create_rnd_binary_vector(params.c1, n_bits)
        c2_vector = create_rnd_binary_vector(params.c2, n_bits)

        new_velocity = [omega_vector & current_vel |
                        (c1_vector & (pbest ^ current_pos)) |
                        (c2_vector & (gbest ^ current_pos)) for
                        (current_vel, current_pos, pbest, gbest) in
                        zip(current_velocity, current_position, pbest_position,
                            gbest_position)]

        return new_velocity



class BooleanPSOVelocityMutationStrategy(VelocityStrategy):
    def update_velocity(self, current_velocity, current_position, pbest_position, gbest_position,params, n_bits):
        pass