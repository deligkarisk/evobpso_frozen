from unittest import TestCase
from unittest.mock import Mock, patch
import unittest.mock as mock

from evobpso.position_update_strategy.StandardPositionUpdateStrategy import StandardPositionUpdateStrategy
from evobpso.velocity_factor.VelocityFactor import VelocityFactorEvolve, VelocityFactorRemove, VelocityFactorAdd


class TestNeuralBooleanPSOStandardPositionUpdateStrategy(TestCase):

    def test_get_new_position_valid_case(self):
        optimization_params = Mock()
        optimization_params.pso_params.n_bits = 6
        optimization_params.pso_params.mutation_prob = 0
        strategy = StandardPositionUpdateStrategy(optimization_params)
        current_position = [0b000111, 0b110000]

        component_a = VelocityFactorEvolve(data=0b100111)
        component_b = VelocityFactorRemove()
        component_c = VelocityFactorAdd(data=0b100111)
        current_velocity = [component_a, component_b, component_c]
        result = strategy.get_new_position(current_position=current_position, current_velocity=current_velocity)
        expected_result = [0b100000, 0b100111]
        assert expected_result == result


    def test_get_new_position_invalid_case_when_evolving_an_empty_position(self):
        # in case the algorithm tries to evolve (i.e. XOR) a velocity vector with an empty position vector, this should
        # result in an exception. In summary, when there is an Evolve velocity component, there must always be a current position
        # to XOR with (for the specific dimension).

        optimization_params = Mock()
        optimization_params.pso_params.n_bits = 6
        optimization_params.pso_params.mutation_prob = 0
        strategy = StandardPositionUpdateStrategy(optimization_params)
        current_position = [0b000111]

        component_a = VelocityFactorEvolve(data=0b100111)
        component_b = VelocityFactorEvolve(data=0b000000)
        component_c = VelocityFactorAdd(data=0b100111)
        current_velocity = [component_a, component_b, component_c]
        with self.assertRaises(Exception):
            result = strategy.get_new_position(current_position=current_position, current_velocity=current_velocity)
