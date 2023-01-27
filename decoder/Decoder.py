import abc

# The decoder class is responsible for decoding values before those are passed
# to the problem for evaluation.
# For example, when we use binary coding to solve a real-valued problem,
# then the values need to be decoded from binary to real before evaluating them
# with the specified problem.


class Decoder(abc.ABC):

    def decode(self, problem, encoded_value):
        raise NotImplementedError

class BinToRealDecoder(Decoder):

    def __init__(self, n_bits):
        self.n_bits = n_bits
        self.resolution = 2 ** self.n_bits - 1

    def decode(self, problem, encoded_value):
        step = (problem.max_value - problem.min_value) / self.resolution
        decoded_value = [problem.min_value + single_dim_value * step for single_dim_value in encoded_value]
        return decoded_value


class RealToRealDecoder(Decoder):
    def decode(self, problem, encoded_value):
        return encoded_value
