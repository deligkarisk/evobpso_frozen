from velocity_component.VelocityComponent import VelocityComponentAdd, VelocityComponentEvolve, VelocityComponentRemove

# The ComponentToPositionVisitor class implements the transitions from a component to a position.
# For example, for the standard approach, an XOR operation is applied to the evolve component.


class ComponentToPositionStandardVisitor:

    def do_for_component_add(self, component: VelocityComponentAdd):
        return component.data

    def do_for_component_evolve(self, component: VelocityComponentEvolve, current_position):
        return current_position ^ component.data

    def do_for_component_remove(self, component: VelocityComponentRemove):
        return None


