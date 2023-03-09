from unittest import TestCase
from unittest.mock import Mock

from component_creator.StandardBooleanComponentCreator import StandardBooleanComponentCreator
from component_data_calculator.StandardBoolenComponentDataCalculator import StandardBooleanComponentDataCalculator
from architecture_decoder.DoNothingDecoder import DoNothingDecoder
from initializer.BinaryInitializer import BinaryInitializer
from params.NeuralArchitectureParams import NeuralArchitectureParams
from params.OptimizationParams import OptimizationParams
from params.PsoParams import BooleanPSOParams
from particle.Particle import Particle
from position_update_strategy.StandardPositionUpdateStrategy import StandardPositionUpdateStrategy
from position_validator.DoNothingPositionValidator import DoNothingPositionValidator
from velocity_update_strategy.StandardVelocityUpdateStrategy import StandardVelocityUpdateStrategy
from velocity_update_strategy.component_merge_strategy.StandardComponentMergeStrategy import StandardComponentMergeStrategy


class TestParticle(TestCase):

    def test_iterate(self):
        pso_params = BooleanPSOParams(pop_size=10, iters=10, c1=0.3, c2=0.3, n_bits=32, k=0.5, mutation_prob=0)
        architecture = NeuralArchitectureParams(min_layers=10, max_layers=20)
        training_params = Mock()
        optimization_params = OptimizationParams(pso_params=pso_params, architecture_params=architecture, training_params=training_params)
        validator = DoNothingPositionValidator()
        initializer = BinaryInitializer(params=optimization_params)
        mock_evaluator = Mock()
        data_calculator = StandardBooleanComponentDataCalculator(params=optimization_params)
        component_creator = StandardBooleanComponentCreator(data_calculator=data_calculator)
        component_merger = StandardComponentMergeStrategy()
        velocity_strategy = StandardVelocityUpdateStrategy(component_creator, component_merger, optimization_params)
        position_update_strategy = StandardPositionUpdateStrategy(optimization_params)
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
