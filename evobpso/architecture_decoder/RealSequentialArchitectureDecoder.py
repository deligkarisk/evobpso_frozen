from evobpso.architecture_decoder.ArchitectureDecoder import ArchitectureDecoder
from evobpso.layer.Layer import ConvLayer, MaxPooling, AvgPooling, FlattenLayer, DenseLayer
from evobpso.utils.utils import extract_integer_from_subset_of_bits


class RealSequentialArchitectureDecoder(ArchitectureDecoder):


    def decode(self, encoded_position):

        decoded_position = []

        for encoded_layer in encoded_position:

            if encoded_layer['pooling'] > 1:
                raise Exception('pooling layer value needs to be below or equal to 1')


            filters = encoded_layer['conv_filters']
            kernel_size = encoded_layer['kernel_size']

            decoded_position.append(ConvLayer(filters=filters, kernel_size=kernel_size))

            if 0.33 < encoded_layer['pooling'] <= 0.66:
                decoded_position.append(MaxPooling())
            elif 0.66 < encoded_layer['pooling'] <= 1:
                decoded_position.append(AvgPooling())

        # Add a flat layer at the end
        decoded_position.append(FlattenLayer())
        decoded_position.append(DenseLayer())

        return decoded_position


