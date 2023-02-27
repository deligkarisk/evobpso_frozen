from architecture_decoder.ArchitectureDecoder import ArchitectureDecoder
from layer.Layer import ConvLayer, MaxPooling, AvgPooling
from utils.utils import extract_integer_from_subset_of_bits


class StandardArchitectureDecoder(ArchitectureDecoder):

    def decode(self, encoded_position):

        decoded_position = []

        for encoded_layer in encoded_position:
            # first nine bits represent the number of filters
            filters = extract_integer_from_subset_of_bits(encoded_layer, 0, 8) + 1 # filters minimum value is one

            # subsequent three bits represent the kernel size
            kernel_size = extract_integer_from_subset_of_bits(encoded_layer, 8, 3) + 2  # kernel size minimum value is two

            # subsequent two bits represent the number of pooling layers
            pooling_layers = extract_integer_from_subset_of_bits(encoded_layer, 11, 2)

            # subsequent bit represents the pooling type
            pooling_type = extract_integer_from_subset_of_bits(encoded_layer, 13, 1)

            decoded_position.append(ConvLayer(filters=filters, kernel_size=kernel_size))

            if pooling_type == 0:
                for i in range(0, pooling_layers):
                    decoded_position.append(MaxPooling())
            elif pooling_type == 1:
                for i in range(0, pooling_layers):
                    decoded_position.append(AvgPooling())

        return decoded_position

