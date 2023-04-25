import abc


class VelocityUpdateExtension(abc.ABC):

    def __init__(self, params):
        self.params = params

    def get_new_velocity(self, velocity):
        raise NotImplementedError