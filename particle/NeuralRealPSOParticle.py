from particle.Particle import Particle
from position_update_strategy.NeuralRealPSOPositionUpdateStrategy import NeuralRealPSOPositionUpdateStrategy
from pso_params.PsoParams import NeuralRealPSOParams
from velocity_update_strategy.NeuralRealPSOVelocityUpdateStrategy import NeuralRealPSOVelocityUpdateStrategy


class NeuralRealPSOParticle(Particle):

    def __init__(self, parent_pop, problem, decoder, pso_params: NeuralRealPSOParams, velocity_strategy: NeuralRealPSOVelocityUpdateStrategy,
                 position_update_strategy: NeuralRealPSOPositionUpdateStrategy):

        if not isinstance(position_update_strategy, NeuralRealPSOPositionUpdateStrategy):
            raise ValueError("Position update strategy in Neural BPSO must be " + str(NeuralRealPSOPositionUpdateStrategy.__name__))

        if not isinstance(velocity_strategy, NeuralRealPSOVelocityUpdateStrategy):
            raise ValueError("Velocity update strategy in Neural BPSO must be " + str(NeuralRealPSOVelocityUpdateStrategy.__name__))

        Particle.__init__(self, parent_pop, problem, decoder, pso_params, velocity_strategy, position_update_strategy)

    def get_initial_positions(self):
        pass

    def get_initial_velocity(self):
        pass