from evobpso.velocity_update_strategy.VelocityUpdateStrategy import VelocityUpdateStrategy


class ReturnZeroVelocityUpdateStrategy(VelocityUpdateStrategy):

    def __init__(self):
        pass

    def get_new_velocity(self, current_position, pbest_position, gbest_position):
        return 0



