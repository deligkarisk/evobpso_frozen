
class FixedArchitectureParams:
    def __init__(self,conv_stride, activation_function, pool_layer_kernel_size, pool_layer_stride, padding) -> None:
        self.conv_stride = conv_stride
        self.activation_function = activation_function
        self.pool_layer_kernel_size = pool_layer_kernel_size
        self.pool_layer_stride = pool_layer_stride
        self.padding = padding
