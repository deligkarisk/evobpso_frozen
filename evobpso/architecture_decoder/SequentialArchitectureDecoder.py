from evobpso.architecture_decoder.ArchitectureDecoder import ArchitectureDecoder
from evobpso.layer.Layer import ConvLayer, MaxPooling, AvgPooling, FlattenLayer, DenseLayer
from evobpso.utils.utils import extract_integer_from_subset_of_bits


class SequentialArchitectureDecoder(ArchitectureDecoder):


    def decode(self, encoded_position):

        decoded_position = []

        for encoded_layer in encoded_position:
            # first nine bits represent the number of filters
            filters = extract_integer_from_subset_of_bits(encoded_layer, start_index=0, length=self.encoding.filter_bits) + 1 # filters minimum value is one

            # subsequent three bits represent the kernel size
            kernel_size = extract_integer_from_subset_of_bits(encoded_layer, start_index=self.encoding.filter_bits, length=self.encoding.kernel_size_bits) + 2  # kernel size minimum value is two

            # subsequent bit represents the existence of a pooling layer
            pooling_layer = extract_integer_from_subset_of_bits(encoded_layer, start_index=self.encoding.pooling_layer_bit_position, length=self.encoding.pooling_layer_bits)

            # subsequent bit represents the pooling type
            pooling_type = extract_integer_from_subset_of_bits(encoded_layer, start_index=self.encoding.pooling_type_bit_position, length=self.encoding.pooling_type_bits)

            decoded_position.append(ConvLayer(filters=filters, kernel_size=kernel_size))

            if pooling_type == 0:
                for i in range(0, pooling_layer):
                    decoded_position.append(MaxPooling())
            elif pooling_type == 1:
                for i in range(0, pooling_layer):
                    decoded_position.append(AvgPooling())

        # Add a flat layer at the end
        decoded_position.append(FlattenLayer())
        decoded_position.append(DenseLayer())

        return decoded_position


