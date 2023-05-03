from evobpso.encoding.Encoding import Encoding


class RealEncoding(Encoding):
    def __init__(self, min_conv_filters, max_conv_filters, min_kernel_size, max_kernel_size):
        self.min_conv_filters = min_conv_filters
        self.max_conv_filters = max_conv_filters
        self.min_kernel_size = min_kernel_size
        self.max_kernel_size = max_kernel_size
        super().__init__()
