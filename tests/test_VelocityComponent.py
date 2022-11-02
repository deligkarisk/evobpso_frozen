from unittest import TestCase
from unittest.mock import patch

from VelocityComponent import VelocityComponentEvolve, VelocityComponentAdd, VelocityComponentRemove


class TestVelocityComponentEvolve(TestCase):

    @patch('VelocityComponent.VelocityComponentProcessor')
    def test_merge_does_xor_when_both_are_evolve(self, mock_processor):
        component_a = VelocityComponentEvolve(data=0b000111, velocity_component_processor=mock_processor)
        component_b = VelocityComponentEvolve(data=0b100111, velocity_component_processor=mock_processor)

        result_a = component_a.merge(component_b)
        result_b = component_b.merge(component_a)

        assert mock_processor.xor.call_count == 2

    @patch('VelocityComponent.VelocityComponentProcessor')
    def test_merge_does_random_choice_when_one_is_add(self, mock_processor):
        component_a = VelocityComponentEvolve(data=0b000111, velocity_component_processor=mock_processor)
        component_b = VelocityComponentAdd(data=0b100111, velocity_component_processor=mock_processor)

        result_a = component_a.merge(component_b)

        assert mock_processor.xor.call_count == 0
        assert mock_processor.random_choice.call_count == 1

    @patch('VelocityComponent.VelocityComponentProcessor')
    def test_merge_does_random_choice_when_one_is_remove(self, mock_processor):
        component_a = VelocityComponentEvolve(data=0b000111, velocity_component_processor=mock_processor)
        component_b = VelocityComponentRemove(velocity_component_processor=mock_processor)
        result_a = component_a.merge(component_b)

        assert mock_processor.xor.call_count == 0
        assert mock_processor.random_choice.call_count == 1


class TestVelocityComponentAdd(TestCase):
    @patch('VelocityComponent.VelocityComponentProcessor')
    def test_merge_does_always_random_choice(self, mock_processor):
        component_a = VelocityComponentAdd(data=0b100111, velocity_component_processor=mock_processor)
        component_b = VelocityComponentEvolve(data=0b100111, velocity_component_processor=mock_processor)
        component_c = VelocityComponentAdd(data=0b100111, velocity_component_processor=mock_processor)
        component_d = VelocityComponentRemove(velocity_component_processor=mock_processor)

        result_1 = component_a.merge(component_b)
        result_2 = component_a.merge(component_c)
        result_3 = component_a.merge(component_d)

        assert mock_processor.random_choice.call_count == 3


class TestVelocityComponentRemove(TestCase):
    @patch('VelocityComponent.VelocityComponentProcessor')
    def test_merge_does_always_random_choice(self, mock_processor):
        component_a = VelocityComponentRemove(velocity_component_processor=mock_processor)
        component_b = VelocityComponentAdd(data=0b100111, velocity_component_processor=mock_processor)
        component_c = VelocityComponentEvolve(data=0b100111, velocity_component_processor=mock_processor)
        component_d = VelocityComponentRemove(velocity_component_processor=mock_processor)

        result_1 = component_a.merge(component_b)
        result_2 = component_a.merge(component_c)
        result_3 = component_a.merge(component_d)

        assert mock_processor.random_choice.call_count == 3
