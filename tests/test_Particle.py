from unittest import TestCase
from unittest.mock import Mock, patch

from particle.Particle import BooleanPSOParticle, RealPSOParticle, NeuralBPSOParticle
from neural_component_to_position_visitor.ComponentToPositionStandardVisitor import ComponentToPositionStandardVisitor
from position_update_strategy.BooleanPSOPositionUpdateStrategy import BooleanPSOStandardPositionUpdateStrategy
from position_update_strategy.NeuralBPSOPositionUpdateStrategy import NeuralBPSOStandardPositionUpdateStrategy
from position_update_strategy.RealPSOPositionUpdateStrategy import RealPSOStandardPositionUpdateStrategy
from velocity_update_strategy.BooleanPSOVelocityUpdateStrategy import BooleanPSOStandardVelocityStrategy
from velocity_update_strategy.NeuralBPSOVelocityUpdateStrategy import NeuralBPSOStandardVelocityStrategy
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
            particle = NeuralBPSOParticle(parent_pop, problem, decoder, pso_params, velocity_strategy, position_update_strategy)

    def test_initialization_with_real_velocity_update_should_fail(self):
        position_update_strategy = BooleanPSOStandardPositionUpdateStrategy()
        parent_pop = Mock()
        problem = Mock()
        decoder = Mock()
        pso_params = Mock()
        velocity_strategy = RealPSOStandardVelocityStrategy(pso_params)
        with self.assertRaises(ValueError):
            particle = NeuralBPSOParticle(parent_pop, problem, decoder, pso_params, velocity_strategy, position_update_strategy)

    @patch("particle.Particle.Particle.__init__")
    def test_initialization_with_boolean_velocity_and_position_should_call_the_parent_constructor(self, mock_particle):
        position_update_strategy = BooleanPSOStandardPositionUpdateStrategy()
        parent_pop = Mock()
        problem = Mock()
        decoder = Mock()
        pso_params = Mock()
        velocity_strategy = BooleanPSOStandardVelocityStrategy(pso_params)
        with self.assertRaises(ValueError):
            particle = NeuralBPSOParticle(parent_pop, problem, decoder, pso_params, velocity_strategy, position_update_strategy)

    @patch("particle.Particle.Particle.__init__")
    def test_initialization_with_real_velocity_and_position_should_call_the_parent_constructor(self, mock_particle):
        comp_to_pos_visitor = ComponentToPositionStandardVisitor()
        position_update_strategy = NeuralBPSOStandardPositionUpdateStrategy(component_to_position_visitor=comp_to_pos_visitor)
        parent_pop = Mock()
        problem = Mock()
        decoder = Mock()
        pso_params = Mock()
        velocity_strategy = NeuralBPSOStandardVelocityStrategy(pso_params)
        particle = NeuralBPSOParticle(parent_pop, problem, decoder, pso_params, velocity_strategy, position_update_strategy)
        assert mock_particle.called