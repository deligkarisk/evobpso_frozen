from evobpso.encoding.Encoding import Encoding


class BooleanEncoding(Encoding):

    def __init__(self, filter_bits, kernel_size_bits):
        self.filter_bits = filter_bits
        self.kernel_size_bits = kernel_size_bits
        self.pooling_layer_bit_position = self.filter_bits + self.kernel_size_bits
        self.pooling_layer_bits = 1  # one bit for pooling layer
        self.pooling_type_bits = 1  # one bit for pooling layer type
        self.pooling_type_bit_position = self.pooling_layer_bit_position + self.pooling_layer_bits
        self.total_bits = self.filter_bits + self.kernel_size_bits + self.pooling_layer_bits + self.pooling_type_bits # add two for the pooling layer and pooling type bits

        super().__init__()

