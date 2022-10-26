import abc
import random

class RealPSOVelocityStrategy(abc.ABC):
    def get_new_velocity(self, current_velocity, current_position, pbest_position, gbest_position, params):
        raise NotImplementedError


class RealPSOStandardVelocityStrategy(RealPSOVelocityStrategy):
    def get_new_velocity(self, current_velocity, current_position, pbest_position, gbest_position, params):
        new_velocity = [params.omega * current_vel +
                        (params.c1 * random.uniform(0, 1) * (pbest - current_pos)) +
                        (params.c2 * random.uniform(0, 1) * (gbest - current_pos)) for
                        (current_vel, current_pos, pbest, gbest) in
                        zip(current_velocity, current_position, pbest_position,
                            gbest_position)]
        return new_velocity
