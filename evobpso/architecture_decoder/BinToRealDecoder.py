from evobpso.architecture_decoder.ArchitectureDecoder import ArchitectureDecoder


class BinToRealDecoder(ArchitectureDecoder):

    def __init__(self, n_bits):
        self.n_bits = n_bits
        self.resolution = 2 ** self.n_bits - 1

    def decode(self, problem, encoded_value):
        step = (problem.max_value - problem.min_value) / self.resolution
        decoded_value = [problem.min_value + single_dim_value * step for single_dim_value in encoded_value]
        return decoded_value
