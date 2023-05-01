from evobpso.encoding.Encoding import Encoding


class BooleanEncoding(Encoding):

    def __init__(self, filter_bits, kernel_size_bits):
        self.filter_bits = filter_bits
        self.kernel_size_bits = kernel_size_bits
        self.pooling_layer_bit_position = self.filter_bits + self.kernel_size_bits
        self.total_bits = self.filter_bits + self.kernel_size_bits + 2 # add two for the pooling layer and pooling type bits

        super().__init__()

