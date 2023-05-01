from unittest import TestCase
from unittest.mock import Mock

from evobpso.evaluator.MockIncreasingEvaluator import MockIncreasingEvaluator
from evobpso.params.NeuralArchitectureParams import NeuralArchitectureParams
from evobpso.params.OptimizationParams import OptimizationParams
from evobpso.params.PsoParams import BooleanPSOParams
from evobpso.population.Population import Population
from evobpso.scheme.SequentialScheme import SequentialScheme


class TestPopulation(TestCase):

    def test_iterate(self):
        pso_params = BooleanPSOParams(pop_size=50, iters=10, c1=0.3, c2=0.3, n_bits=32, k=0.5, mutation_prob=0)
        architecture = NeuralArchitectureParams(min_layers=10, max_layers=20, max_pooling_layers=2)
        training_params = Mock()
        optimization_params = OptimizationParams(pso_params=pso_params, neural_architecture_params=architecture, training_params=training_params)
        mock_evaluator = MockIncreasingEvaluator()
        scheme = SequentialScheme(version='boolean', variable_length=True, vmax=False, vmut=False, optimization_params=optimization_params)
        scheme.compile(fixed_architecture_properties=architecture, data_loader=Mock(), results_folder=None)
        scheme.evaluator = mock_evaluator


        population = Population(scheme)

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
