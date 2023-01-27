import abc
import random
from utils import utils


class VelocityComponent(abc.ABC):

    def __init__(self):
        self.data = None

    def convert_to_position(self, current_position, position_conversion_visitor):
        raise NotImplementedError


class VelocityComponentEvolve(VelocityComponent):

    def __init__(self, data):
        self.data = data

    def __eq__(self, other):
        if self.data == other.data and isinstance(self, type(other)):
            return True
        else:
            return False

    def convert_to_position(self, current_position, position_conversion_visitor):
        return position_conversion_visitor.do_for_component_evolve(self, current_position)


class VelocityComponentRemove(VelocityComponent):
    def __init__(self):
        self.data = None

    def __eq__(self, other):
        if isinstance(self, type(other)):
            return True
        else:
            return False

    def convert_to_position(self, current_position, position_conversion_visitor):
        return position_conversion_visitor.do_for_component_remove(self)


class VelocityComponentAdd(VelocityComponent):
    def __init__(self, data):
        self.data = data

    def __eq__(self, other):
        if self.data == other.data and isinstance(self, type(other)):
            return True
        else:
            return False

    def convert_to_position(self, current_position, position_conversion_visitor):
        return position_conversion_visitor.do_for_component_add(self)
