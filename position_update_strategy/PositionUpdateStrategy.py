import abc


# The position update strategy defines how the position will be calculated given the current position
# and current velocity

class PositionUpdateStrategy(abc.ABC):

    def __init__(self, optimization_params):
        self.optimization_params = optimization_params

    @abc.abstractmethod
    def get_new_position(self, current_position, current_velocity):
        raise NotImplementedError
