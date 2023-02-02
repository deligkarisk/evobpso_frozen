from decoder.Decoder import Decoder


class DoNothingDecoder(Decoder):

    def decode(self, encoded_value):
        return encoded_value
