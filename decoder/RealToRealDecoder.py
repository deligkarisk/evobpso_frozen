from decoder.Decoder import Decoder


class RealToRealDecoder(Decoder):
    def decode(self, problem, encoded_value):
        return encoded_value
