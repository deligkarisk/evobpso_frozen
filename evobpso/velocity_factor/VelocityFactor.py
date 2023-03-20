import abc
import random
from evobpso.utils import utils


# The velocity component class defines what components exist, and how to convert each velocity component to a position vector.
class VelocityFactor(abc.ABC):

    def __init__(self):
        self.data = None

    def convert_to_position(self, current_position, position_conversion_visitor):
        raise NotImplementedError


class VelocityFactorEvolve(VelocityFactor):

    def __init__(self, data):
        self.data = data

    def __eq__(self, other):
        if self.data == other.data and isinstance(self, type(other)):
            return True
        else:
            return False

    def convert_to_position(self, current_position, position_conversion_visitor):
        return position_conversion_visitor.do_for_component_evolve(self, current_position)


class VelocityFactorRemove(VelocityFactor):
    def __init__(self):
        self.data = None

    def __eq__(self, other):
        if isinstance(self, type(other)):
            return True
        else:
            return False

    def convert_to_position(self, current_position, position_conversion_visitor):
        return position_conversion_visitor.do_for_component_remove(self)


class VelocityFactorAdd(VelocityFactor):
    def __init__(self, data):
        self.data = data

    def __eq__(self, other):
        if self.data == other.data and isinstance(self, type(other)):
            return True
        else:
            return False

    def convert_to_position(self, current_position, position_conversion_visitor):
        return position_conversion_visitor.do_for_component_add(self)
