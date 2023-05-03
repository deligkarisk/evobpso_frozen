import random

from evobpso.initializer.Initializer import Initializer


class RealInitializer(Initializer):
    def get_initial_position(self):

        # get random number to see how many layers this position would have
        num_layers = random.randint(self.architecture.min_layers, self.architecture.max_layers)
        position = []
        for i in range(0, num_layers):
            kernel_size = random.randint(self.encoding.min_kernel_size, self.encoding.max_kernel_size)
            conv_filters = random.randint(self.encoding.min_conv_filters, self.encoding.max_conv_filters)
            pooling = random.random()
            layer = {'kernel_size': kernel_size, 'conv_filters': conv_filters, 'pooling': pooling}
            position.append(layer)
        return position

