from unittest import TestCase
from unittest.mock import Mock, patch

from Particle import BooleanPSOParticle, Particle
from position_update_strategy.BooleanPSOPositionUpdateStrategy import BooleanPSOStandardPositionUpdateStrategy
from position_update_strategy.RealPositionUpdateStrategy import RealPSOStandardPositionUpdateStrategy
from velocity_strategy.BooleanPSOVelocityStrategy import BooleanPSOStandardVelocityStrategy
from velocity_strategy.RealPSOVelocityStrategy import RealPSOStandardVelocityStrategy


class TestBooleanPSOParticle(TestCase):

    def test_initialization_with_real_position_update_should_fail(self):
        velocity_strategy = BooleanPSOStandardVelocityStrategy()
        position_update_strategy = RealPSOStandardPositionUpdateStrategy()
        parent_pop = Mock()
        problem = Mock()
        decoder = Mock()
        pso_params = Mock()
        with self.assertRaises(ValueError):
            particle = BooleanPSOParticle(parent_pop, problem, decoder, pso_params, velocity_strategy, position_update_strategy)



    def test_initialization_with_real_velocity_update_should_fail(self):
        velocity_strategy = RealPSOStandardVelocityStrategy()
        position_update_strategy = BooleanPSOStandardPositionUpdateStrategy()
        parent_pop = Mock()
        problem = Mock()
        decoder = Mock()
        pso_params = Mock()
        with self.assertRaises(ValueError):
            particle = BooleanPSOParticle(parent_pop, problem, decoder, pso_params, velocity_strategy, position_update_strategy)




    @patch("Particle.Particle.__init__")
    def test_initialization_with_boolean_velocity_and_position_should_call_the_parent_constructor(self, mock_particle):
        velocity_strategy = BooleanPSOStandardVelocityStrategy()
        position_update_strategy = BooleanPSOStandardPositionUpdateStrategy()
        parent_pop = Mock()
        problem = Mock()
        decoder = Mock()
        pso_params = Mock()
        particle = BooleanPSOParticle(parent_pop, problem, decoder, pso_params, velocity_strategy, position_update_strategy)
        assert mock_particle.called

