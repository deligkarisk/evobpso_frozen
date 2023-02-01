import abc


class Problem(abc.ABC):

    @abc.abstractmethod
    def evaluate(self, position):
        raise NotImplementedError


class NeuralArchitecture(Problem):

    def __init__(self, min_out_conv, max_out_conv, min_kernel_conv, max_kernel_conv, min_layers, max_layers):
        self.min_out_conv = min_out_conv
        self.max_out_conv = max_out_conv
        self.min_kernel_conv = min_kernel_conv
        self.max_kernel_conv = max_kernel_conv
        self.min_layers = min_layers
        self.max_layers = max_layers

    def evaluate(self, position):
        raise NotImplementedError
