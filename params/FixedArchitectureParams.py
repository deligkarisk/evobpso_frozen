
class FixedArchitectureParams:
    def __init__(self, input_shape, conv_stride, activation_function, pool_layer_kernel_size, pool_layer_stride, padding, dense_layer_units) -> None:
        self.input_shape = input_shape
        self.conv_stride = conv_stride
        self.activation_function = activation_function
        self.pool_layer_kernel_size = pool_layer_kernel_size
        self.pool_layer_stride = pool_layer_stride
        self.padding = padding
        self.dense_layer_units = dense_layer_units
