import os
from unittest import TestCase

from evobpso.architecture_decoder.StandardArchitectureDecoder import StandardArchitectureDecoder
from evobpso.component_creator.VariableLengthComponentCreator import VariableLengthComponentCreator
from evobpso.component_creator_data_calculator.BoolenComponentCreatorDataCalculator import BooleanComponentCreatorDataCalculator
from evobpso.evaluator.StandardNNEvaluator import StandardNNEvaluator
from evobpso.initializer.BinaryInitializer import BinaryInitializer
from evobpso.model_creator.TensorflowModelCreator import TensorflowModelCreator
from evobpso.params.FixedArchitectureProperties import FixedArchitectureProperties
from evobpso.params.NeuralArchitectureParams import NeuralArchitectureParams
from evobpso.params.OptimizationParams import OptimizationParams
from evobpso.params.PsoParams import BooleanPSOParams
from evobpso.params.TrainingParams import TrainingParams
from evobpso.population.Population import Population
from evobpso.position_update_strategy.StandardPositionUpdateStrategy import StandardPositionUpdateStrategy
from evobpso.position_validator.DoNothingPositionValidator import DoNothingPositionValidator
from evobpso.position_validator.ValidatePoolingLayers import ValidatePoolingLayers
from evobpso.utils import data_load_utils
from evobpso.velocity_update_strategy.StandardVelocityUpdateStrategy import StandardVelocityUpdateStrategy
from evobpso.velocity_update_strategy.component_merge_strategy.VariableLengthCalculateDataComponentMergeStrategy import VariableLengthCalculateDataComponentMergeStrategy


class TestBooleanPSOIntegrationTest(TestCase):

    def test_runs_without_errors(self):

        num_of_classes = 10
        image_input_shape = (28, 28, 1)
        results_folder = os.path.join(data_load_utils.get_project_root(), 'test_tmp_results')

        data_loader = data_load_utils.load_mnist_data

        pso_params = BooleanPSOParams(c1=0.5, c2=0.5, n_bits=14, k=0.5, iters=2, mutation_prob=0.001, pop_size=1)
        optimizable_architecture_params = NeuralArchitectureParams(min_layers=3, max_layers=8, max_pooling_layers=2)
        training_params = TrainingParams(batch_size=64, train_eval_epochs=1, best_solution_training_epochs=1, loss='categorical_crossentropy',
                                         optimizer='rmsprop', metrics=['accuracy'])
        optimization_params = OptimizationParams(pso_params=pso_params, architecture_params=optimizable_architecture_params, training_params=training_params)
        fixed_architecture_params = FixedArchitectureProperties(input_shape=image_input_shape, conv_stride=1, activation_function='relu',
                                                                pool_layer_kernel_size=2, pool_layer_stride=2,
                                                                padding='same', dense_layer_units=num_of_classes)

        data_calculator = BooleanComponentCreatorDataCalculator(params=optimization_params)
        component_creator = VariableLengthComponentCreator(data_calculator=data_calculator)
        component_merger = VariableLengthCalculateDataComponentMergeStrategy()
        velocity_strategy = StandardVelocityUpdateStrategy(component_creator=component_creator, params=optimization_params, component_merger=component_merger)
        position_update_strategy = StandardPositionUpdateStrategy(optimization_params=optimization_params)
        decoder = StandardArchitectureDecoder()
        model_creator = TensorflowModelCreator(fixed_architecture_properties=fixed_architecture_params)
        evaluator = StandardNNEvaluator(architecture_decoder=decoder, model_creator=model_creator,
                                        training_params=training_params, data_loader= data_loader)
        validator = DoNothingPositionValidator()
        initializer = BinaryInitializer(params=optimization_params)
        parent_pop = Population(params=optimization_params, validator=validator, initializer=initializer,
                                evaluator=evaluator, velocity_update_strategy=velocity_strategy, position_update_strategy=position_update_strategy)

        aggregated_history = []

        results = parent_pop.iterate(first_iter=True)
        aggregated_history.append(results)
        for i in range(0, 2):
            results = parent_pop.iterate(first_iter=False)
            aggregated_history.append(results)
        print("Integrated test finished")



