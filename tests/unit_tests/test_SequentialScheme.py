from unittest import TestCase
from unittest.mock import Mock

from evobpso.component_creator.VariableLengthComponentCreator import VariableLengthComponentCreator
from evobpso.component_creator.data_calculator.BoolenComponentCreatorDataCalculator import BooleanComponentCreatorDataCalculator
from evobpso.component_merger.VariableLengthCalculateDataComponentMerger import VariableLengthCalculateDataComponentMerger
from evobpso.component_merger.data_calculator.BooleanComponentMergerDataCalculator import BooleanComponentMergerDataCalculator
from evobpso.initializer.BinaryInitializer import BinaryInitializer
from evobpso.scheme.SequentialScheme import SequentialScheme
from evobpso.velocity_update_extension.BooleanVmaxExtension import BooleanVmaxExtension
from evobpso.velocity_update_extension.BooleanVmutExtension import BooleanVmutExtension


class TestSequentialScheme(TestCase):

    def test_init_boolean_version_no_vmax_no_vmut(self):
        scheme = SequentialScheme(version='boolean', variable_length=True, vmax=False, vmut=False, optimization_params=Mock(), encoding=Mock())
        assert scheme.initializer.__class__.__name__ == BinaryInitializer.__name__
        assert scheme.velocity_update_strategy.component_creator.data_calculator.__class__.__name__ == BooleanComponentCreatorDataCalculator.__name__
        assert scheme.velocity_update_strategy.component_merger.component_merger_data_calculator.__class__.__name__ == BooleanComponentMergerDataCalculator.__name__
        assert scheme.velocity_update_strategy.component_creator.__class__.__name__ == VariableLengthComponentCreator.__name__
        assert scheme.velocity_update_strategy.component_merger.__class__.__name__ == VariableLengthCalculateDataComponentMerger.__name__
        assert len(scheme.velocity_update_strategy.velocity_update_extensions) == 0

    def test_init_boolean_version_no_vmax_with_vmut(self):
        scheme = SequentialScheme(version='boolean', variable_length=True, vmax=False, vmut=True, optimization_params=Mock(), encoding=Mock())
        assert scheme.initializer.__class__.__name__ == BinaryInitializer.__name__
        assert scheme.velocity_update_strategy.component_creator.data_calculator.__class__.__name__ == BooleanComponentCreatorDataCalculator.__name__
        assert scheme.velocity_update_strategy.component_merger.component_merger_data_calculator.__class__.__name__ == BooleanComponentMergerDataCalculator.__name__
        assert scheme.velocity_update_strategy.component_creator.__class__.__name__ == VariableLengthComponentCreator.__name__
        assert scheme.velocity_update_strategy.component_merger.__class__.__name__ == VariableLengthCalculateDataComponentMerger.__name__
        assert len(scheme.velocity_update_strategy.velocity_update_extensions) == 1
        assert scheme.velocity_update_strategy.velocity_update_extensions[0].__class__.__name__ == BooleanVmutExtension.__name__

    def test_init_boolean_version_with_vmax_no_vmut(self):
        scheme = SequentialScheme(version='boolean', variable_length=True, vmax=True, vmut=False, optimization_params=Mock(), encoding=Mock())
        assert scheme.initializer.__class__.__name__ == BinaryInitializer.__name__
        assert scheme.velocity_update_strategy.component_creator.data_calculator.__class__.__name__ == BooleanComponentCreatorDataCalculator.__name__
        assert scheme.velocity_update_strategy.component_merger.component_merger_data_calculator.__class__.__name__ == BooleanComponentMergerDataCalculator.__name__
        assert scheme.velocity_update_strategy.component_creator.__class__.__name__ == VariableLengthComponentCreator.__name__
        assert scheme.velocity_update_strategy.component_merger.__class__.__name__ == VariableLengthCalculateDataComponentMerger.__name__
        assert len(scheme.velocity_update_strategy.velocity_update_extensions) == 1
        assert scheme.velocity_update_strategy.velocity_update_extensions[0].__class__.__name__ == BooleanVmaxExtension.__name__

    def test_init_boolean_version_with_vmax_with_vmut(self):
        scheme = SequentialScheme(version='boolean', variable_length=True, vmax=True, vmut=True, optimization_params=Mock(), encoding=Mock())
        assert scheme.initializer.__class__.__name__ == BinaryInitializer.__name__
        assert scheme.velocity_update_strategy.component_creator.data_calculator.__class__.__name__ == BooleanComponentCreatorDataCalculator.__name__
        assert scheme.velocity_update_strategy.component_merger.component_merger_data_calculator.__class__.__name__ == BooleanComponentMergerDataCalculator.__name__
        assert scheme.velocity_update_strategy.component_creator.__class__.__name__ == VariableLengthComponentCreator.__name__
        assert scheme.velocity_update_strategy.component_merger.__class__.__name__ == VariableLengthCalculateDataComponentMerger.__name__
        assert len(scheme.velocity_update_strategy.velocity_update_extensions) == 2
        assert scheme.velocity_update_strategy.velocity_update_extensions[0].__class__.__name__ == BooleanVmaxExtension.__name__
        assert scheme.velocity_update_strategy.velocity_update_extensions[1].__class__.__name__ == BooleanVmutExtension.__name__
