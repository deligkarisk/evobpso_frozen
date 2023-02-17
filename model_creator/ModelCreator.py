import abc

from params.FixedArchitectureParams import FixedArchitectureParams


class ModelCreator(abc.ABC):

    def __init__(self, fixed_architecture_params: FixedArchitectureParams) -> None:
        self.fixed_architecture_params = fixed_architecture_params

    @abc.abstractmethod
    def create_model(self, architecture):
        raise NotImplementedError
