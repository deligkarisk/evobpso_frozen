import abc


# A validator should validate the position of the particle after update.
# if they are not compliant to the min and max settings, then it should update them to bring them to valid values.

class Validator(abc.ABC):

    @abc.abstractmethod
    def validate(self, position, params):
        raise NotImplementedError