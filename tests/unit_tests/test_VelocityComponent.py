from unittest import TestCase
from unittest.mock import patch, Mock

from evobpso.velocity_factor.VelocityFactor import VelocityFactorEvolve, VelocityFactorAdd, VelocityFactorRemove


class TestVelocityComponentEvolve(TestCase):

    def test_get_new_position(self):
        component = VelocityFactorEvolve(data=0b111000)
        position_visitor = Mock()
        current_position = 0b0000001
        new_position = component.convert_to_position(current_position=current_position, position_conversion_visitor=position_visitor)
        assert position_visitor.do_for_component_evolve.call_count == 1
        # assert new_position == 0b111001


class TestVelocityComponentAdd(TestCase):

    def test_get_new_position(self):
        component = VelocityFactorAdd(data=0b111000)
        current_position = None
        position_visitor = Mock()
        new_position = component.convert_to_position(current_position=current_position, position_conversion_visitor=position_visitor)
        assert position_visitor.do_for_component_add.call_count == 1


class TestVelocityComponentRemove(TestCase):

    def test_get_new_position(self):
        component = VelocityFactorRemove()
        position_visitor = Mock()
        current_position = [0b0000001]
        new_position = component.convert_to_position(current_position=current_position, position_conversion_visitor=position_visitor)
        assert position_visitor.do_for_component_remove.call_count == 1
