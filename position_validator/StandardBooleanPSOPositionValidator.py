from position_validator.PositionValidator import PositionValidator


class StandardBooleanPSOPositionValidator(PositionValidator):

    def validate(self, position, params):
        raise NotImplementedError