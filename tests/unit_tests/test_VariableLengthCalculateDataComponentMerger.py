from unittest import TestCase
from unittest.mock import Mock

from evobpso.velocity_factor.VelocityFactor import VelocityFactorEvolve, VelocityFactorRemove, VelocityFactorAdd
from evobpso.component_merger.VariableLengthCalculateDataComponentMerger import \
    VariableLengthCalculateDataComponentMerger


class TestVariableLengthCalculateDataComponentMerger(TestCase):
    def test_merge_personal_and_global_components_correct_number_of_calculator_calls(self):

        component_a = VelocityFactorEvolve(data=0b100111)
        component_b = VelocityFactorRemove()
        component_d = VelocityFactorEvolve(data=0b100111)
        component_c = VelocityFactorAdd(data=0b100111)

        personal_component = [component_a, component_b, component_c, component_d]

        component_a = VelocityFactorEvolve(data=0b111111)
        component_b = VelocityFactorRemove()
        component_d = VelocityFactorEvolve(data=0b000100)
        global_component = [component_a, component_b, component_c, component_d]


        mock_data_calculator = Mock()
        pso_params = Mock()
        pso_params.k = 1

        component_merge_strategy = VariableLengthCalculateDataComponentMerger(component_merger_data_calculator=mock_data_calculator)
        result = component_merge_strategy.merge_personal_and_global_components(personal_component=personal_component, global_component=global_component, pso_params=pso_params)

        # There are two layers in the personal and global components that both are Evolve, so the calculator should have been called twice
        assert mock_data_calculator.calculate.call_count == 2
