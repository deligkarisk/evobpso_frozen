import abc


class VelocityUpdateStrategy(abc.ABC):

    def get_new_velocity(self, current_velocity, current_position, pbest_position, gbest_position):
        raise NotImplementedError

