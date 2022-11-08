from unittest import TestCase
from unittest.mock import patch

from position_update_strategy.NeuralPositionUpdateStrategy import BooleanPSONeuralPositionUpdateStrategy
from velocity_component.VelocityComponent import VelocityComponentEvolve, VelocityComponentRemove, VelocityComponentAdd


class TestBooleanPSONeuralPositionUpdateStrategy(TestCase):

    @patch('velocity_component.VelocityComponent.VelocityComponentProcessor')
    def test_get_new_position(self, mock_processor):
        strategy = BooleanPSONeuralPositionUpdateStrategy()
        current_position = [0b000111, 0b110000]

        component_a = VelocityComponentEvolve(data=0b100111, processor=mock_processor)
        component_b = VelocityComponentRemove(processor=mock_processor)
        component_c = VelocityComponentAdd(data=0b100111, processor=mock_processor)
        current_velocity = [component_a, component_b, component_c]
        result = strategy.get_new_position(current_position=current_position, current_velocity=current_velocity)
        expected_result = [0b100000, 0b100111]
        assert expected_result == result



