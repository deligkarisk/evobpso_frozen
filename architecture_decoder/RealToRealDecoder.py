from architecture_decoder.ArchitectureDecoder import ArchitectureDecoder


class RealToRealDecoder(ArchitectureDecoder):
    def decode(self, problem, encoded_value):
        return encoded_value
