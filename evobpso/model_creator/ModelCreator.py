import abc

from evobpso.params.FixedArchitectureProperties import FixedArchitectureProperties


class ModelCreator(abc.ABC):

    def __init__(self, fixed_architecture_properties: FixedArchitectureProperties) -> None:
        self.fixed_architecture_properties = fixed_architecture_properties

    @abc.abstractmethod
    def create_model(self, architecture):
        raise NotImplementedError
