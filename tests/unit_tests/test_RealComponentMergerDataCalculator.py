from unittest import TestCase

from evobpso.component_merger.data_calculator.RealComponentMergerDataCalculator import RealComponentMergerDataCalculator


class TestRealComponentMergerDataCalculator(TestCase):
    def test_calculate(self):
        personal_component_data = {'conv_filters': 20, 'kernel_size': 4, 'pooling': 0.2}
        global_component_data = {'conv_filters': 12, 'kernel_size': -1, 'pooling': 0.3}
        data_calculator = RealComponentMergerDataCalculator()
        result = data_calculator.calculate(personal_component_data=personal_component_data, global_component_data=global_component_data)
        assert result['conv_filters'] == 32
        assert result['kernel_size'] == 3
        assert 0.499 <= result['pooling'] <= 0.5001

