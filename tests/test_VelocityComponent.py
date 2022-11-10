from unittest import TestCase
from unittest.mock import patch, Mock

from velocity_component.VelocityComponent import VelocityComponentEvolve, VelocityComponentAdd, VelocityComponentRemove


class TestVelocityComponentEvolve(TestCase):

    @patch('velocity_component.VelocityComponent.VelocityComponentProcessor')
    def test_merge_does_xor_when_both_are_evolve(self, mock_processor):
        component_a = VelocityComponentEvolve(data=0b000111, processor=mock_processor)
        component_b = VelocityComponentEvolve(data=0b100111, processor=mock_processor)

        result_a = component_a.merge(component_b)
        result_b = component_b.merge(component_a)

        assert mock_processor.xor.call_count == 2

    @patch('velocity_component.VelocityComponent.VelocityComponentProcessor')
    def test_merge_does_random_choice_when_one_is_add(self, mock_processor):
        component_a = VelocityComponentEvolve(data=0b000111, processor=mock_processor)
        component_b = VelocityComponentAdd(data=0b100111, processor=mock_processor)

        result_a = component_a.merge(component_b)

        assert mock_processor.xor.call_count == 0
        assert mock_processor.random_choice.call_count == 1

    @patch('velocity_component.VelocityComponent.VelocityComponentProcessor')
    def test_merge_does_random_choice_when_one_is_remove(self, mock_processor):
        component_a = VelocityComponentEvolve(data=0b000111, processor=mock_processor)
        component_b = VelocityComponentRemove(processor=mock_processor)
        result_a = component_a.merge(component_b)

        assert mock_processor.xor.call_count == 0
        assert mock_processor.random_choice.call_count == 1

    @patch('velocity_component.VelocityComponent.VelocityComponentProcessor')
    def test_get_new_position(self, mock_processor):
        component = VelocityComponentEvolve(data=0b111000, processor=mock_processor)
        position_visitor = Mock()
        current_position = 0b0000001
        new_position = component.convert_to_position(current_position=current_position, position_conversion_visitor=position_visitor)
        assert position_visitor.do_for_component_evolve.call_count == 1
        #assert new_position == 0b111001


class TestVelocityComponentAdd(TestCase):
    @patch('velocity_component.VelocityComponent.VelocityComponentProcessor')
    def test_merge_does_always_random_choice(self, mock_processor):
        component_a = VelocityComponentAdd(data=0b100111, processor=mock_processor)
        component_b = VelocityComponentEvolve(data=0b100111, processor=mock_processor)
        component_c = VelocityComponentAdd(data=0b100111, processor=mock_processor)
        component_d = VelocityComponentRemove(processor=mock_processor)

        result_1 = component_a.merge(component_b)
        result_2 = component_a.merge(component_c)
        result_3 = component_a.merge(component_d)

        assert mock_processor.random_choice.call_count == 3

    @patch('velocity_component.VelocityComponent.VelocityComponentProcessor')
    def test_get_new_position(self, mock_processor):
        component = VelocityComponentAdd(data=0b111000, processor=mock_processor)
        current_position = None
        position_visitor = Mock()
        new_position = component.convert_to_position(current_position=current_position, position_conversion_visitor=position_visitor)
        assert position_visitor.do_for_component_add.call_count == 1


class TestVelocityComponentRemove(TestCase):
    @patch('velocity_component.VelocityComponent.VelocityComponentProcessor')
    def test_merge_does_always_random_choice(self, mock_processor):
        component_a = VelocityComponentRemove(processor=mock_processor)
        component_b = VelocityComponentAdd(data=0b100111, processor=mock_processor)
        component_c = VelocityComponentEvolve(data=0b100111, processor=mock_processor)
        component_d = VelocityComponentRemove(processor=mock_processor)

        result_1 = component_a.merge(component_b)
        result_2 = component_a.merge(component_c)
        result_3 = component_a.merge(component_d)

        assert mock_processor.random_choice.call_count == 3

    @patch('velocity_component.VelocityComponent.VelocityComponentProcessor')
    def test_get_new_position(self, mock_processor):
        component = VelocityComponentRemove(processor=mock_processor)
        position_visitor = Mock()
        current_position = [0b0000001]
        new_position = component.convert_to_position(current_position=current_position, position_conversion_visitor=position_visitor)
        assert  position_visitor.do_for_component_remove.call_count == 1

