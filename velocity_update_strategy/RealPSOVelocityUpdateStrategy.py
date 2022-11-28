import abc
import random

from velocity_update_strategy.VelocityUpdateStrategy import VelocityStrategy


class RealPSOVelocityStrategy(VelocityStrategy, abc.ABC):
    pass


class RealPSOStandardVelocityStrategy(RealPSOVelocityStrategy):

    def __init__(self, params):
        self.params = params

    def get_new_velocity(self, current_velocity, current_position, pbest_position, gbest_position):
        params = self.params
        new_velocity = [params.omega * current_vel +
                        (params.c1 * random.uniform(0, 1) * (pbest - current_pos)) +
                        (params.c2 * random.uniform(0, 1) * (gbest - current_pos)) for
                        (current_vel, current_pos, pbest, gbest) in
                        zip(current_velocity, current_position, pbest_position,
                            gbest_position)]
        return new_velocity
