from evobpso.utils.utils import create_rnd_binary_vector
from evobpso.velocity_update_extension.VelocityUpdateExtension import VelocityUpdateExtension



class BooleanVmutExtension(VelocityUpdateExtension):
    def get_new_velocity(self, velocity):
        mut_prob = self.params.pso_params.mutation_prob
        n_bits = self.params.pso_params.n_bits
        for layer in velocity:
            if layer.__class__.__name__ != "VelocityFactorRemove":
                new_data = layer.data
                random_bitstring = create_rnd_binary_vector(mut_prob, n_bits)
                new_data = new_data | random_bitstring
                layer.data = new_data
        return velocity

