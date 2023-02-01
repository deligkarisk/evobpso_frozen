import abc

from position_update_strategy.PositionUpdateStrategy import PositionUpdateStrategy
from pso_params.PsoParams import PsoParams
from velocity_update_strategy.VelocityUpdateStrategy import VelocityUpdateStrategy


class ParticleFactory(abc.ABC):

    def make_particle(self, parent_pop, problem, decoder, pso_params: PsoParams, velocity_strategy: VelocityUpdateStrategy,
                      position_update_strategy: PositionUpdateStrategy):
        raise NotImplementedError
