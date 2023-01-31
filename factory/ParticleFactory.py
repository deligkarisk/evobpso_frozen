import abc

from particle.BooleanPSOParticle import BooleanPSOParticle
from particle.NeuralBooleanPSOParticle import NeuralBooleanPSOParticle
from particle.NeuralRealPSOParticle import NeuralRealPSOParticle
from particle.RealPSOParticle import RealPSOParticle
from position_update_strategy.NeuralRealPSOPositionUpdateStrategy import NeuralRealPSOPositionUpdateStrategy
from pso_params.PsoParams import PsoParams, NeuralBooleanPSOParams, NeuralRealPSOParams
from position_update_strategy.BooleanPSOPositionUpdateStrategy import BooleanPSOPositionUpdateStrategy
from position_update_strategy.NeuralBooleanPSOPositionUpdateStrategy import NeuralBooleanPSOPositionUpdateStrategy
from position_update_strategy.PositionUpdateStrategy import PositionUpdateStrategy
from position_update_strategy.RealPSOPositionUpdateStrategy import RealPSOPositionUpdateStrategy
from velocity_update_strategy.BooleanPSOVelocityUpdateStrategy import BooleanPSOVelocityStrategy
from velocity_update_strategy.NeuralBooleanPSOVelocityUpdateStrategy import NeuralBooleanPSOVelocityUpdateStrategy
from velocity_update_strategy.NeuralRealPSOVelocityUpdateStrategy import NeuralRealPSOVelocityUpdateStrategy
from velocity_update_strategy.RealPSOVelocityUpdateStrategy import RealPSOVelocityStrategy
from velocity_update_strategy.VelocityUpdateStrategy import VelocityStrategy


class ParticleFactory(abc.ABC):

    def make_particle(self, parent_pop, problem, decoder, pso_params: PsoParams, velocity_strategy: VelocityStrategy,
                      position_update_strategy: PositionUpdateStrategy):
        raise NotImplementedError


class BooleanPSOParticleFactory(ParticleFactory):

    def make_particle(self, parent_pop, problem, decoder, pso_params: PsoParams, velocity_strategy: BooleanPSOVelocityStrategy,
                      position_update_strategy: BooleanPSOPositionUpdateStrategy):
        return BooleanPSOParticle(parent_pop, problem, decoder, pso_params, velocity_strategy, position_update_strategy)


class RealPSOParticleFactory(ParticleFactory):

    def make_particle(self, parent_pop, problem, decoder, pso_params: PsoParams, velocity_strategy: RealPSOVelocityStrategy,
                      position_update_strategy: RealPSOPositionUpdateStrategy):
        return RealPSOParticle(parent_pop, problem, decoder, pso_params, velocity_strategy, position_update_strategy)


class NeuralBooleanPSOParticleFactory(ParticleFactory):
    def make_particle(self, parent_pop, problem, decoder, pso_params: NeuralBooleanPSOParams, velocity_strategy: NeuralBooleanPSOVelocityUpdateStrategy,
                      position_update_strategy: NeuralBooleanPSOPositionUpdateStrategy):
        return NeuralBooleanPSOParticle(parent_pop, problem, decoder, pso_params, velocity_strategy, position_update_strategy)


class NeuralRealPSOParticleFactory(ParticleFactory):
    def make_particle(self, parent_pop, problem, decoder, pso_params: NeuralRealPSOParams, velocity_strategy: NeuralRealPSOVelocityUpdateStrategy,
                      position_update_strategy: NeuralRealPSOPositionUpdateStrategy):
        return NeuralRealPSOParticle(parent_pop, problem, decoder, pso_params, velocity_strategy, position_update_strategy)
