import abc


class PositionUpdateStrategy(abc.ABC):
    def get_new_position(self, current_position, current_velocity):
        raise NotImplementedError
