from unittest import TestCase

import utils.data_load_utils
from component_creator.StandardBooleanComponentCreator import StandardBooleanComponentCreator
from component_data_calculator.StandardBoolenComponentDataCalculator import StandardBooleanComponentDataCalculator
from architecture_decoder.StandardArchitectureDecoder import StandardArchitectureDecoder
from evaluator.StandardNNEvaluator import StandardNNEvaluator
from initializer.BinaryInitializer import BinaryInitializer
from model_creator.TensorflowModelCreator import TensorflowModelCreator
from params.FixedArchitectureParams import FixedArchitectureParams
from params.NeuralArchitectureParams import NeuralArchitectureParams
from params.OptimizationParams import OptimizationParams
from params.PsoParams import PsoParams, BooleanPSOParams
from population.Population import Population
from position_update_strategy.StandardPositionUpdateStrategy import StandardPositionUpdateStrategy
from position_validator.StandardBooleanPSOPositionValidator import StandardBooleanPSOPositionValidator
from velocity_update_strategy.StandardVelocityUpdateStrategy import StandardVelocityUpdateStrategy


class TestBooleanPSOIntegrationTest(TestCase):

    def test_runs_without_errors(self):

        num_of_classes = 10
        image_input_shape = (28, 28, 1)

        data_loader = utils.data_load_utils.load_mnist_data

        pso_params = BooleanPSOParams(c1=0.5, c2=0.5, n_bits=15, k=0.5)
        optimizable_architecture_params = NeuralArchitectureParams(min_out_conv=8, max_out_conv=64, min_kernel_conv=2, max_kernel_conv=8, min_layers=8, max_layers=32)
        all_params = OptimizationParams(pso_params=pso_params, architecture_params=optimizable_architecture_params)
        fixed_architecture_params = FixedArchitectureParams(input_shape=image_input_shape, conv_stride=1, activation_function='relu',
                                                            pool_layer_kernel_size=2, pool_layer_stride=2,
                                                            padding='same', dense_layer_units=num_of_classes)
        data_calculator = StandardBooleanComponentDataCalculator(params=all_params)
        component_creator = StandardBooleanComponentCreator(data_calculator=data_calculator)
        velocity_strategy = StandardVelocityUpdateStrategy(component_creator=component_creator, params=all_params)
        position_update_strategy = StandardPositionUpdateStrategy()
        decoder = StandardArchitectureDecoder()
        model_creator = TensorflowModelCreator(fixed_architecture_params=fixed_architecture_params)
        evaluator = StandardNNEvaluator(architecture_decoder=decoder, model_creator=model_creator, data_loader= data_loader)
        validator = StandardBooleanPSOPositionValidator()
        initializer = BinaryInitializer(params=all_params)
        parent_pop = Population(pop_size=20, params=all_params, validator=validator, initializer=initializer,
                                evaluator=evaluator, velocity_update_strategy=velocity_strategy, position_update_strategy=position_update_strategy)

        for i in range(0, 1000):
            parent_pop.iterate()

