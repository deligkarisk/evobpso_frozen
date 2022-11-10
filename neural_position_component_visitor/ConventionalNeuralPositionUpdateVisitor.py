from velocity_component.VelocityComponent import VelocityComponentAdd, VelocityComponentEvolve, VelocityComponentRemove


class ConventionalNeuralPositionUpdateVisitor:

    def do_for_component_add(self, component: VelocityComponentAdd):
        return component.data

    def do_for_component_evolve(self, component: VelocityComponentEvolve, current_position):
        return current_position ^ component.data

    def do_for_component_remove(self, component: VelocityComponentRemove):
        return None


