# This class describes a layer.


class ConvLayer:

    def __init__(self, filters, kernel_size) -> None:
        self.filters = filters
        self.kernel_size = kernel_size


class MaxPooling:
    def __init__(self, pooling_size) -> None:
        self.pooling_size = pooling_size


class AvgPooling:
    def __init__(self, pooling_size) -> None:
        self.pooling_size = pooling_size
