from unittest import TestCase
from unittest.mock import Mock

from evobpso.component_creator.VariableLengthComponentCreator import VariableLengthComponentCreator
from evobpso.component_creator.data_calculator.BoolenComponentCreatorDataCalculator import BooleanComponentCreatorDataCalculator
from evobpso.encoding.BooleanEncoding import BooleanEncoding
from evobpso.initializer.BinaryInitializer import BinaryInitializer
from evobpso.params.NeuralArchitectureParams import NeuralArchitectureParams
from evobpso.params.OptimizationParams import OptimizationParams
from evobpso.params.PsoParams import BooleanPSOParams
from evobpso.particle.Particle import Particle
from evobpso.position_update_strategy.BooleanPositionUpdateStrategy import BooleanPositionUpdateStrategy
from evobpso.position_validator.DoNothingPositionValidator import DoNothingPositionValidator
from evobpso.velocity_update_strategy.StandardVelocityUpdateStrategy import StandardVelocityUpdateStrategy
from evobpso.component_merger.VariableLengthCalculateDataComponentMerger import VariableLengthCalculateDataComponentMerger
from evobpso.component_merger.data_calculator.BooleanComponentMergerDataCalculator import \
    BooleanComponentMergerDataCalculator


class TestParticle(TestCase):

    def test_iterate(self):
        encoding = BooleanEncoding(filter_bits=24, kernel_size_bits=6)
        pso_params = BooleanPSOParams(pop_size=10, iters=10, c1=0.3, c2=0.3, k=0.5, mutation_prob=0, encoding=encoding)
        architecture = NeuralArchitectureParams(min_layers=10, max_layers=20, max_pooling_layers=2)
        training_params = Mock()
        optimization_params = OptimizationParams(pso_params=pso_params, neural_architecture_params=architecture, training_params=training_params)
        validator = DoNothingPositionValidator()
        initializer = BinaryInitializer(params=optimization_params, encoding=encoding)
        mock_evaluator = Mock()
        data_calculator = BooleanComponentCreatorDataCalculator(params=optimization_params)
        component_creator = VariableLengthComponentCreator(data_calculator=data_calculator)
        component_merger_data_calculator = BooleanComponentMergerDataCalculator()
        component_merger = VariableLengthCalculateDataComponentMerger(component_merger_data_calculator=component_merger_data_calculator)
        velocity_strategy = StandardVelocityUpdateStrategy(component_creator, component_merger, optimization_params, velocity_update_extensions=[])
        position_update_strategy = BooleanPositionUpdateStrategy(optimization_params)
        mock_population = Mock()
        mock_population.global_best_position = [11020303, 3042323]

        particle = Particle(mock_population, optimization_params, validator, initializer, mock_evaluator,
                            velocity_strategy, position_update_strategy)

        mock_evaluator.evaluate_for_train.return_value = 100

        particle.iterate(first_iter=True)

        # some quick initial tests
        assert len(particle.current_position) > 0
        assert particle.personal_best_position == particle.current_position
        assert particle.personal_best_result == particle.current_result

        # in the next iteration, the particle finds a better solution
        mock_evaluator.evaluate_for_train.return_value = 51.5
        old_position = particle.current_position
        particle.iterate(first_iter=False)

        assert particle.personal_best_result == 51.5
        assert particle.personal_best_position == particle.current_position
        assert particle.current_position != old_position

        # subsequently, in the next iteration, the particle finds a worse solution
        mock_evaluator.evaluate_for_train.return_value = 114.9
        old_position = particle.current_position
        particle.iterate(first_iter=False)

        assert particle.personal_best_result == 51.5
        assert particle.personal_best_position == old_position
        assert particle.current_position != old_position
