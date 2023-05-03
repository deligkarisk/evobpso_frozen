from unittest import TestCase
from unittest.mock import patch, Mock

from evobpso.encoding.RealEncoding import RealEncoding
from evobpso.initializer.RealInitializer import RealInitializer
from evobpso.params.NeuralArchitectureParams import NeuralArchitectureParams
from evobpso.params.OptimizationParams import OptimizationParams
from evobpso.params.PsoParams import RealPSOParams


class TestRealInitializer(TestCase):

    def test_get_initial_position(self):
        pso_params = RealPSOParams(pop_size=2, iters=2, c1=0, c2=0, omega=0, k=1)
        encoding = RealEncoding(min_conv_filters=2, max_conv_filters=10, min_kernel_size=2, max_kernel_size=4)
        architecture = NeuralArchitectureParams(min_layers=2, max_layers=4, max_pooling_layers=2)
        training_params = Mock()
        params = OptimizationParams(pso_params=pso_params, neural_architecture_params=architecture, training_params=training_params)
        initializer = RealInitializer(params, encoding=encoding)

        kernel_sizes = []
        conv_filters = []
        layers = []

        for i in range(0, 100):
            new_position = initializer.get_initial_position()
            layers.append(len(new_position))
            kernel_sizes.extend([x['kernel_size'] for x in new_position])
            conv_filters.extend([x['conv_filters'] for x in new_position])
        assert min(kernel_sizes) == encoding.min_kernel_size
        assert max(kernel_sizes) == encoding.max_kernel_size
        assert min(conv_filters) == encoding.min_conv_filters
        assert max(conv_filters) == encoding.max_conv_filters
        assert min(layers) == architecture.min_layers
        assert max(layers) == architecture.max_layers




