from unittest import TestCase
from unittest.mock import Mock

from evobpso.position_update_strategy.BooleanPositionUpdateStrategy import BooleanPositionUpdateStrategy
from evobpso.position_update_strategy.RealPositionUpdateStrategy import RealPositionUpdateStrategy
from evobpso.velocity_factor.VelocityFactor import VelocityFactorEvolve, VelocityFactorRemove, VelocityFactorAdd


class TestRealPositionUpdateStrategy(TestCase):

    def test_get_new_position_valid_case(self):
        optimization_params = Mock()
        strategy = RealPositionUpdateStrategy(optimization_params)
        current_position = []
        current_position.append({'conv_filters': 131, 'kernel_size': 5, 'pooling': 0.8})
        current_position.append({'conv_filters': 131, 'kernel_size': 5, 'pooling': 0.88})


        component_a = VelocityFactorEvolve(data={'conv_filters': 24, 'kernel_size': -1, 'pooling': -0.1})
        component_b = VelocityFactorRemove()
        component_c = VelocityFactorAdd(data={'conv_filters': 33, 'kernel_size': 4, 'pooling': 0.7})
        current_velocity = [component_a, component_b, component_c]
        result = strategy.get_new_position(current_position=current_position, current_velocity=current_velocity)
        expected_position = []
        expected_position.append({'conv_filters': 131 + 24, 'kernel_size': 5 - 1, 'pooling': 0.8 - 0.1})
        expected_position.append({'conv_filters': 33, 'kernel_size': 4, 'pooling': 0.7})

        assert expected_position == result


    def test_get_new_position_invalid_case_when_evolving_an_empty_position(self):
        # in case the algorithm tries to evolve (i.e. add) a velocity vector with an empty position vector, this should
        # result in an exception. In summary, when there is an Evolve velocity component, there must always be a current position
        # to add with (for the specific dimension).

        optimization_params = Mock()
        strategy = RealPositionUpdateStrategy(optimization_params)
        current_position = []
        current_position.append({'conv_filters': 131, 'kernel_size': 5, 'pooling': 0.8})

        component_a = VelocityFactorEvolve(data={'conv_filters': 24, 'kernel_size': -1, 'pooling': -0.1})
        component_b = VelocityFactorEvolve(data={'conv_filters': 24, 'kernel_size': -1, 'pooling': -0.1})
        component_c = VelocityFactorAdd(data={'conv_filters': 33, 'kernel_size': 4, 'pooling': 0.7})

        current_velocity = [component_a, component_b, component_c]
        with self.assertRaises(Exception):
            result = strategy.get_new_position(current_position=current_position, current_velocity=current_velocity)
