from evobpso.velocity_factor.VelocityFactor import VelocityFactorRemove


def equalize_sizes(personal_component, global_component):
    # this method equalizes the sizes of the two lists of components, by adding Remove components in the shortest list
    p_size = len(personal_component)
    g_size = len(global_component)

    # if the sizes are equal, nothing to do
    if p_size == g_size:
        return personal_component, global_component

    if p_size < g_size:
        diff = g_size - p_size
        for i in range(0, diff):
            personal_component.append(VelocityFactorRemove())
    else:
        diff = p_size - g_size
        for i in range(0, diff):
            global_component.append(VelocityFactorRemove())

    return personal_component, global_component
