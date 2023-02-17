import abc

# The architecture_decoder class is responsible for decoding values before those are passed
# to the problem for evaluation.
# For example, when we use binary coding to solve a real-valued problem,
# then the values need to be decoded from binary to real before evaluating them
# with the specified problem.


class ArchitectureDecoder(abc.ABC):

    def decode(self, encoded_value):
        raise NotImplementedError



