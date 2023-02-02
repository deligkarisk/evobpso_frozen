from unittest import TestCase
from unittest.mock import Mock

from decoder.DoNothingDecoder import DoNothingDecoder
from initializer.BinaryInitializer import BinaryInitializer
from params.NeuralArchitectureParams import NeuralArchitectureParams
from params.Params import Params
from params.PsoParams import BooleanPSOParams
from particle.Particle import Particle
from position_update_strategy.NeuralBooleanPSOPositionUpdateStrategy import NeuralBooleanPSOStandardPositionUpdateStrategy
from validator.DoNothingValidator import DoNothingValidator
from velocity_update_strategy.NeuralBooleanPSOVelocityUpdateStrategy import NeuralBooleanPSOStandardVelocityUpdateStrategy


class TestParticle(TestCase):


    def test_iterate(self):
        pso_params = BooleanPSOParams(c1=0.3, c2=0.3, omega=0.1, n_bits=32, k=0.5)
        architecture = NeuralArchitectureParams(min_out_conv=2, max_out_conv=4,
                                                min_kernel_conv=2, max_kernel_conv=4,
                                                min_layers=10, max_layers=20)
        all_params = Params(pso_params=pso_params, architecture_params=architecture)
        decoder = DoNothingDecoder()
        validator = DoNothingValidator()
        initializer = BinaryInitializer(params=all_params)
        mock_evaluator = Mock()
        mock_evaluator.evaluate.return_value = 100
        velocity_strategy = NeuralBooleanPSOStandardVelocityUpdateStrategy(pso_params)
        position_update_strategy = NeuralBooleanPSOStandardPositionUpdateStrategy()
        mock_population = Mock()
        mock_population.global_best_position = [11020303, 3042323]


        particle = Particle(mock_population, all_params, decoder, validator, initializer, mock_evaluator,
                            velocity_strategy, position_update_strategy)

        # some quick initial tests
        assert len(particle.current_position) > 0
        assert particle.personal_best_position == particle.current_position
        assert particle.personal_best_result == particle.current_result

        # in the next iteration, the particle finds a better solution
        mock_evaluator.evaluate.return_value = 51.5
        old_position = particle.current_position
        particle.iterate()

        assert particle.personal_best_result == 51.5
        assert particle.personal_best_position == particle.current_position
        assert particle.current_position != old_position

        # subsequently, in the next iteration, the particle finds a worse solution
        mock_evaluator.evaluate.return_value = 114.9
        old_position = particle.current_position
        particle.iterate()

        assert particle.personal_best_result == 51.5
        assert particle.personal_best_position == old_position
        assert particle.current_position != old_position









