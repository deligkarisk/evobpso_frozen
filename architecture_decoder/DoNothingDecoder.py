from architecture_decoder.ArchitectureDecoder import ArchitectureDecoder


class DoNothingDecoder(ArchitectureDecoder):

    def decode(self, encoded_value):
        return encoded_value
