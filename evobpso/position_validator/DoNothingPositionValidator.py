from evobpso.position_validator.PositionValidator import PositionValidator


class DoNothingPositionValidator(PositionValidator):

    def validate(self, position, params):
        return position