from unittest import TestCase
from unittest.mock import Mock

from evobpso.component_creator.StandardBooleanComponentCreator import StandardBooleanComponentCreator
from evobpso.component_data_calculator.StandardBoolenComponentDataCalculator import StandardBooleanComponentDataCalculator
from evobpso.params.OptimizationParams import OptimizationParams
from evobpso.velocity_factor.VelocityFactor import VelocityFactorEvolve, VelocityFactorAdd, VelocityFactorRemove


class TestStandardBooleanComponentCreator(TestCase):

    def test_create_component_when_best_position_longer_than_current_position(self):
        pso_params = Mock()
        architecture_params = Mock()
        pso_params.c1 = 1
        pso_params.c2 = 1
        pso_params.n_bits = 8
        training_params = Mock()
        params = OptimizationParams(pso_params, architecture_params, training_params)

        best_position = [0b000001, 0b000000, 0b000111]
        current_position = [0b010000, 0b000001]
        data_calculator = StandardBooleanComponentDataCalculator(params=params)
        data_calculator.calculate = Mock(return_value=0b111000)
        component_creator = StandardBooleanComponentCreator(data_calculator=data_calculator)
        new_component = component_creator.create_component(best_position=best_position, current_position=current_position, c_factor=1)

        expected_result = [VelocityFactorEvolve(data=0b111000),
                           VelocityFactorEvolve(data=0b111000),
                           VelocityFactorAdd(data=0b000111)]

        assert new_component == expected_result

    def test_create_component_when_best_position_shorter_than_current_position(self):
        pso_params = Mock()
        architecture_params = Mock()
        pso_params.c1 = 1
        pso_params.c2 = 1
        pso_params.n_bits = 8
        training_params = Mock()
        params = OptimizationParams(pso_params, architecture_params, training_params)

        best_position = [0b111101]
        current_position = [0b010000, 0b000001]
        data_calculator = StandardBooleanComponentDataCalculator(params=params)
        data_calculator.calculate = Mock(return_value=0b111000)
        component_creator = StandardBooleanComponentCreator(data_calculator=data_calculator)
        new_component = component_creator.create_component(best_position=best_position, current_position=current_position, c_factor=1)

        expected_result = [VelocityFactorEvolve(data=0b111000),
                           VelocityFactorRemove()]

        assert new_component == expected_result

    def test_create_component_when_best_position_equal_size_to_current_position(self):
        pso_params = Mock()
        architecture_params = Mock()
        pso_params.c1 = 1
        pso_params.c2 = 1
        pso_params.n_bits = 8
        training_params = Mock()
        params = OptimizationParams(pso_params, architecture_params, training_params)

        best_position = [0b111101, 0b010101]
        current_position = [0b010000, 0b000001]
        data_calculator = StandardBooleanComponentDataCalculator(params=params)
        data_calculator.calculate = Mock(return_value=0b111000)
        component_creator = StandardBooleanComponentCreator(data_calculator=data_calculator)
        new_component = component_creator.create_component(best_position=best_position, current_position=current_position, c_factor=1)

        expected_result = [VelocityFactorEvolve(data=0b111000),
                           VelocityFactorEvolve(data=0b111000)]

        assert new_component == expected_result
