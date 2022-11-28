from unittest import TestCase
from unittest.mock import patch, Mock

from velocity_component.VelocityComponent import VelocityComponentEvolve, VelocityComponentAdd, VelocityComponentRemove


class TestVelocityComponentEvolve(TestCase):

    def test_merge_does_xor_when_both_are_evolve(self):
        component_a = VelocityComponentEvolve(data=0b000111)
        component_b = VelocityComponentEvolve(data=0b100111)

        mock_params = Mock()

        result_a = component_a.merge(component_b, mock_params)
        result_b = component_b.merge(component_a, mock_params)

        expected_result = VelocityComponentEvolve(data=0b100000)
        assert expected_result == result_a
        assert expected_result == result_b

    @patch('utils.utils.random_choice')
    def test_merge_does_random_choice_when_one_is_add(self, random_choice):
        component_a = VelocityComponentEvolve(data=0b000111)
        component_b = VelocityComponentAdd(data=0b100111)
        mock_params = Mock()
        mock_params.k = 1
        result_a = component_a.merge(component_b, mock_params)
        assert random_choice.call_count == 1

    @patch('utils.utils.random_choice')
    def test_merge_does_random_choice_when_one_is_remove(self, random_choice):
        component_a = VelocityComponentEvolve(data=0b000111)
        component_b = VelocityComponentRemove()
        mock_params = Mock()
        mock_params.k = 1
        result_a = component_a.merge(component_b, mock_params)

        assert random_choice.call_count == 1

    def test_get_new_position(self):
        component = VelocityComponentEvolve(data=0b111000)
        position_visitor = Mock()
        current_position = 0b0000001
        new_position = component.convert_to_position(current_position=current_position, position_conversion_visitor=position_visitor)
        assert position_visitor.do_for_component_evolve.call_count == 1
        #assert new_position == 0b111001


class TestVelocityComponentAdd(TestCase):

    @patch('utils.utils.random_choice')
    def test_merge_does_always_random_choice(self, random_choice):
        component_a = VelocityComponentAdd(data=0b100111)
        component_b = VelocityComponentEvolve(data=0b100111)
        component_c = VelocityComponentAdd(data=0b100111)
        component_d = VelocityComponentRemove()

        mock_params = Mock()
        mock_params.k = 1

        result_1 = component_a.merge(component_b, mock_params)
        result_2 = component_a.merge(component_c, mock_params)
        result_3 = component_a.merge(component_d, mock_params)

        assert random_choice.call_count == 3

    def test_get_new_position(self):
        component = VelocityComponentAdd(data=0b111000)
        current_position = None
        position_visitor = Mock()
        new_position = component.convert_to_position(current_position=current_position, position_conversion_visitor=position_visitor)
        assert position_visitor.do_for_component_add.call_count == 1


class TestVelocityComponentRemove(TestCase):


    @patch('utils.utils.random_choice')
    def test_merge_does_always_random_choice(self, random_choice):
        component_a = VelocityComponentRemove()
        component_b = VelocityComponentAdd(data=0b100111)
        component_c = VelocityComponentEvolve(data=0b100111)
        component_d = VelocityComponentRemove()

        mock_params = Mock()
        mock_params.k = 1

        result_1 = component_a.merge(component_b, mock_params)
        result_2 = component_a.merge(component_c, mock_params)
        result_3 = component_a.merge(component_d, mock_params)

        assert random_choice.call_count == 3

    def test_get_new_position(self):
        component = VelocityComponentRemove()
        position_visitor = Mock()
        current_position = [0b0000001]
        new_position = component.convert_to_position(current_position=current_position, position_conversion_visitor=position_visitor)
        assert  position_visitor.do_for_component_remove.call_count == 1

