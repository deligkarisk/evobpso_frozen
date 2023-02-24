from unittest import TestCase

from component_creator.StandardBooleanComponentCreator import StandardBooleanComponentCreator
from component_data_calculator.StandardBoolenComponentDataCalculator import StandardBooleanComponentDataCalculator
from architecture_decoder.DoNothingDecoder import DoNothingDecoder
from evaluator.MockIncreasingEvaluator import MockIncreasingEvaluator
from initializer.BinaryInitializer import BinaryInitializer
from params.NeuralArchitectureParams import NeuralArchitectureParams
from params.OptimizationParams import OptimizationParams
from params.PsoParams import BooleanPSOParams
from population.Population import Population
from position_update_strategy.StandardPositionUpdateStrategy import StandardPositionUpdateStrategy
from position_validator.DoNothingPositionValidator import DoNothingPositionValidator
from velocity_update_strategy.StandardVelocityUpdateStrategy import StandardVelocityUpdateStrategy


class TestPopulation(TestCase):

    def test_iterate(self):
        pso_params = BooleanPSOParams(c1=0.3, c2=0.3, n_bits=32, k=0.5)
        architecture = NeuralArchitectureParams(min_out_conv=2, max_out_conv=4,
                                                min_kernel_conv=2, max_kernel_conv=4,
                                                min_layers=10, max_layers=20)
        all_params = OptimizationParams(pso_params=pso_params, architecture_params=architecture)
        decoder = DoNothingDecoder()
        validator = DoNothingPositionValidator()
        initializer = BinaryInitializer(params=all_params)
        mock_evaluator = MockIncreasingEvaluator()
        data_calculator = StandardBooleanComponentDataCalculator(params=all_params)
        component_creator = StandardBooleanComponentCreator(data_calculator=data_calculator)
        velocity_strategy = StandardVelocityUpdateStrategy(component_creator, all_params)
        position_update_strategy = StandardPositionUpdateStrategy()

        population = Population(50, all_params, validator, initializer, mock_evaluator,
                                velocity_strategy, position_update_strategy)

        # some basic initial checks
        assert len(population.particles) == 50

        population.iterate(first_iter=True)

        assert population.global_best_result == population.particles[0].current_result

        population.iterate(first_iter=False)

        # due to the use of the increasing evaluator we know that there is no best particle at this iteration.
        assert population.global_best_result == population.particles[0].personal_best_result
        assert population.particles[0].personal_best_result != population.particles[0].current_result
        assert population.global_best_position == population.particles[0].personal_best_position

        # the personal best of a random particle should not be the current position, as now the evaluator returns larger numbers
        assert population.particles[15].personal_best_position != population.particles[15].current_position

        # in the next iteration, the evaluator should start from -200, so we will have a new global minimum
        mock_evaluator.count = -200
        population.iterate(first_iter=False)

        assert population.global_best_result == population.particles[0].current_result
        assert population.global_best_result == -199
