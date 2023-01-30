from unittest import TestCase

from neural_component_to_position_visitor.ComponentToPositionStandardVisitor import ComponentToPositionStandardVisitor
from position_update_strategy.NeuralBooleanPSOPositionUpdateStrategy import NeuralBooleanPSOStandardPositionUpdateStrategy
from velocity_component.VelocityComponent import VelocityComponentEvolve, VelocityComponentRemove, VelocityComponentAdd


class TestNeuralBooleanPSOStandardPositionUpdateStrategy(TestCase):

    def test_get_new_position_with_standard_position_visitor(self):
        comp_to_pos_visitor = ComponentToPositionStandardVisitor()

        strategy = NeuralBooleanPSOStandardPositionUpdateStrategy(component_to_position_visitor=comp_to_pos_visitor)
        current_position = [0b000111, 0b110000]

        component_a = VelocityComponentEvolve(data=0b100111)
        component_b = VelocityComponentRemove()
        component_c = VelocityComponentAdd(data=0b100111)
        current_velocity = [component_a, component_b, component_c]
        result = strategy.get_new_position(current_position=current_position, current_velocity=current_velocity)
        expected_result = [0b100000, 0b100111]
        assert expected_result == result



