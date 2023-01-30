from unittest import TestCase
from unittest.mock import Mock, patch

from particle.BooleanPSOParticle import BooleanPSOParticle
from particle.NeuralBooleanPSOParticle import NeuralBooleanPSOParticle
from particle.RealPSOParticle import RealPSOParticle
from position_update_strategy.BooleanPSOPositionUpdateStrategy import BooleanPSOStandardPositionUpdateStrategy
from position_update_strategy.NeuralBooleanPSOPositionUpdateStrategy import NeuralBooleanPSOStandardPositionUpdateStrategy
from position_update_strategy.RealPSOPositionUpdateStrategy import RealPSOStandardPositionUpdateStrategy
from velocity_update_strategy.BooleanPSOVelocityUpdateStrategy import BooleanPSOStandardVelocityStrategy
from velocity_update_strategy.NeuralBooleanPSOVelocityUpdateStrategy import NeuralBooleanPSOStandardVelocityStrategy
from velocity_update_strategy.RealPSOVelocityUpdateStrategy import RealPSOStandardVelocityStrategy


class TestBooleanPSOParticle(TestCase):

    def test_initialization_with_real_position_update_should_fail(self):
        position_update_strategy = RealPSOStandardPositionUpdateStrategy()
        parent_pop = Mock()
        problem = Mock()
        decoder = Mock()
        pso_params = Mock()
        velocity_strategy = BooleanPSOStandardVelocityStrategy(pso_params)

        with self.assertRaises(ValueError):
            particle = BooleanPSOParticle(parent_pop, problem, decoder, pso_params, velocity_strategy, position_update_strategy)

    def test_initialization_with_real_velocity_update_should_fail(self):
        position_update_strategy = BooleanPSOStandardPositionUpdateStrategy()
        parent_pop = Mock()
        problem = Mock()
        decoder = Mock()
        pso_params = Mock()
        velocity_strategy = RealPSOStandardVelocityStrategy(pso_params)

        with self.assertRaises(ValueError):
            particle = BooleanPSOParticle(parent_pop, problem, decoder, pso_params, velocity_strategy, position_update_strategy)

    @patch("particle.Particle.Particle.__init__")
    def test_initialization_with_boolean_velocity_and_position_should_call_the_parent_constructor(self, mock_particle):
        position_update_strategy = BooleanPSOStandardPositionUpdateStrategy()
        parent_pop = Mock()
        problem = Mock()
        decoder = Mock()
        pso_params = Mock()
        velocity_strategy = BooleanPSOStandardVelocityStrategy(pso_params)
        particle = BooleanPSOParticle(parent_pop, problem, decoder, pso_params, velocity_strategy, position_update_strategy)
        assert mock_particle.called


class TestRealPSOParticle(TestCase):

    def test_initialization_with_boolean_position_update_should_fail(self):
        position_update_strategy = BooleanPSOStandardPositionUpdateStrategy()
        parent_pop = Mock()
        problem = Mock()
        decoder = Mock()
        pso_params = Mock()
        velocity_strategy = RealPSOStandardVelocityStrategy(pso_params)

        with self.assertRaises(ValueError):
            particle = RealPSOParticle(parent_pop, problem, decoder, pso_params, velocity_strategy, position_update_strategy)

    def test_initialization_with_boolean_velocity_update_should_fail(self):
        position_update_strategy = RealPSOStandardPositionUpdateStrategy()
        parent_pop = Mock()
        problem = Mock()
        decoder = Mock()
        pso_params = Mock()
        velocity_strategy = BooleanPSOStandardVelocityStrategy(pso_params)
        with self.assertRaises(ValueError):
            particle = RealPSOParticle(parent_pop, problem, decoder, pso_params, velocity_strategy, position_update_strategy)

    @patch("particle.Particle.Particle.__init__")
    def test_initialization_with_real_velocity_and_position_should_call_the_parent_constructor(self, mock_particle):
        position_update_strategy = RealPSOStandardPositionUpdateStrategy()
        parent_pop = Mock()
        problem = Mock()
        decoder = Mock()
        pso_params = Mock()
        velocity_strategy = RealPSOStandardVelocityStrategy(pso_params)
        particle = RealPSOParticle(parent_pop, problem, decoder, pso_params, velocity_strategy, position_update_strategy)
        assert mock_particle.called


class TestNeuralBPSOParticle(TestCase):

    def test_initialization_with_real_position_update_should_fail(self):
        position_update_strategy = RealPSOStandardPositionUpdateStrategy()
        parent_pop = Mock()
        problem = Mock()
        decoder = Mock()
        pso_params = Mock()
        velocity_strategy = BooleanPSOStandardVelocityStrategy(pso_params)
        with self.assertRaises(ValueError):
            particle = NeuralBooleanPSOParticle(parent_pop, problem, decoder, pso_params, velocity_strategy, position_update_strategy)

    def test_initialization_with_real_velocity_update_should_fail(self):
        position_update_strategy = BooleanPSOStandardPositionUpdateStrategy()
        parent_pop = Mock()
        problem = Mock()
        decoder = Mock()
        pso_params = Mock()
        velocity_strategy = RealPSOStandardVelocityStrategy(pso_params)
        with self.assertRaises(ValueError):
            particle = NeuralBooleanPSOParticle(parent_pop, problem, decoder, pso_params, velocity_strategy, position_update_strategy)

    @patch("particle.Particle.Particle.__init__")
    def test_initialization_with_boolean_velocity_and_position_should_call_the_parent_constructor(self, mock_particle):
        position_update_strategy = BooleanPSOStandardPositionUpdateStrategy()
        parent_pop = Mock()
        problem = Mock()
        decoder = Mock()
        pso_params = Mock()
        velocity_strategy = BooleanPSOStandardVelocityStrategy(pso_params)
        with self.assertRaises(ValueError):
            particle = NeuralBooleanPSOParticle(parent_pop, problem, decoder, pso_params, velocity_strategy, position_update_strategy)

    @patch("particle.Particle.Particle.__init__")
    def test_initialization_with_real_velocity_and_position_should_call_the_parent_constructor(self, mock_particle):
        position_update_strategy = NeuralBooleanPSOStandardPositionUpdateStrategy()
        parent_pop = Mock()
        problem = Mock()
        decoder = Mock()
        pso_params = Mock()
        velocity_strategy = NeuralBooleanPSOStandardVelocityStrategy(pso_params)
        particle = NeuralBooleanPSOParticle(parent_pop, problem, decoder, pso_params, velocity_strategy, position_update_strategy)
        assert mock_particle.called