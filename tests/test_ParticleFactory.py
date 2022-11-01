from unittest import TestCase
from unittest.mock import patch, Mock

from particle_factory.ParticleFactory import BooleanPSOParticleFactory, RealPSOParticleFactory


class TestBooleanPSOParticleFactory(TestCase):

    @patch("Particle.BooleanPSOParticle.__init__", return_value=None)
    def test_make_particle_should_call_boolean_pso_constructor(self, mock_particle_init):
        velocity_strategy = Mock()
        position_update_strategy = Mock()
        parent_pop = Mock()
        problem = Mock()
        decoder = Mock()
        pso_params = Mock()
        particle_factory = BooleanPSOParticleFactory()
        particle_factory.make_particle(parent_pop, problem, decoder, pso_params, velocity_strategy, position_update_strategy)
        assert mock_particle_init.called


class TestRealPSOParticleFactory(TestCase):

    @patch("Particle.RealPSOParticle.__init__", return_value=None)
    def test_make_particle_should_call_real_pso_constructor(self, mock_particle_init):
        velocity_strategy = Mock()
        position_update_strategy = Mock()
        parent_pop = Mock()
        problem = Mock()
        decoder = Mock()
        pso_params = Mock()
        particle_factory = RealPSOParticleFactory()
        particle_factory.make_particle(parent_pop, problem, decoder, pso_params, velocity_strategy, position_update_strategy)
        assert mock_particle_init.called
