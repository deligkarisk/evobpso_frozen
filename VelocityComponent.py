import abc
import random


class VelocityComponent(abc.ABC):

    def merge(self, other):
        raise NotImplementedError



class VelocityComponentProcessor:

    def __init__(self, params):
        self.params = params

    def xor(self, personal_component: VelocityComponent, global_component: VelocityComponent):
        pass

    def random_choice(self, personal_component: VelocityComponent, global_component: VelocityComponent):
        if random.uniform(0, 1) < self.params.k:
            return personal_component
        else:
            return global_component


class VelocityComponentEvolve(VelocityComponent):
    def __init__(self, data, velocity_component_processor):
        self.data = data
        self.processor = velocity_component_processor

    def __eq__(self, other):
        if self.data == other.data and isinstance(self, type(other)):
            return True
        else:
            return False

    def merge(self, other):
        if (isinstance(other, VelocityComponentEvolve)):
            return self.processor.xor(self, other)
        elif isinstance(other, VelocityComponentAdd) or isinstance(other, VelocityComponentRemove):
            return self.processor.random_choice(self, other)


class VelocityComponentRemove(VelocityComponent):
    def __init__(self, velocity_component_processor):
        self.data = None
        self.processor = velocity_component_processor

    def merge(self, other):
        return self.processor.random_choice(self, other)

    def __eq__(self, other):
        if isinstance(self, type(other)):
            return True
        else:
            return False


class VelocityComponentAdd(VelocityComponent):
    def __init__(self, data, velocity_component_processor):
        self.data = data
        self.processor = velocity_component_processor

    def __eq__(self, other):
        if self.data == other.data and isinstance(self, type(other)):
            return True
        else:
            return False

    def merge(self, other):
        return self.processor.random_choice(self, other)
