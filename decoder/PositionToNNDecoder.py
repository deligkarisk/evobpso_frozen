from decoder.Decoder import Decoder
from layer.Layer import ConvLayer, MaxPooling, AvgPooling
from utils.utils import extract_integer_from_bits


class PositionToNNDecoder(Decoder):

    def decode(self, encoded_position):
        decoded_position = []

        for encoded_layer in encoded_position:
            # first 9 bits represent the number of filters
            filters = extract_integer_from_bits(encoded_layer, 0, 9) + 1
            kernel_size = extract_integer_from_bits(encoded_layer, 9, 3) + 2  # kernel size minimum is 2
            pooling_layers = extract_integer_from_bits(encoded_layer, 12, 2)
            pooling_type = extract_integer_from_bits(encoded_layer, 14, 1)

            decoded_position.append(ConvLayer(filters=filters, kernel_size=kernel_size))

            if pooling_type == 0:
                for i in range(0, pooling_layers):
                    decoded_position.append(MaxPooling(pooling_size=pooling_layers))
            elif pooling_type == 0:
                for i in range(0, pooling_layers):
                    decoded_position.append(AvgPooling(pooling_size=pooling_layers))

    raise NotImplementedError
