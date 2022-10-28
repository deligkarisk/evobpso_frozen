import abc

from Particle import BooleanPSOParticle, RealPSOParticle
from PsoParams import PsoParams
from position_update_strategy.BooleanPSOPositionUpdateStrategy import BooleanPSOPositionUpdateStrategy
from position_update_strategy.PositionUpdateStrategy import PositionUpdateStrategy
from position_update_strategy.RealPSOPositionUpdateStrategy import RealPSOPositionUpdateStrategy
from velocity_strategy.BooleanPSOVelocityStrategy import BooleanPSOVelocityStrategy
from velocity_strategy.RealPSOVelocityStrategy import RealPSOVelocityStrategy
from velocity_strategy.VelocityStrategy import VelocityStrategy


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
