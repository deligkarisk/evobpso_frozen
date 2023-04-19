import os
from unittest import TestCase

from evobpso.configuration.BPSOWithVmaxAndVmutConfiguration import BPSOWithVmaxAndVmutConfiguration
from evobpso.configuration.BPSOWithVmaxAndVmutConfigurationNoMaxPoolLimit import BPSOWithVmaxAndVmutConfigurationNoMaxPoolLimit
from evobpso.configuration.StandardBPSOConfiguration import StandardBPSOConfiguration
from evobpso.configuration.StandardBPSOConfigurationNoMaxPoolLimit import StandardBPSOConfigurationNoMaxPoolLimit
from evobpso.initializer.MockInitializer import MockInitializer
from evobpso.params.FixedArchitectureProperties import FixedArchitectureProperties
from evobpso.params.NeuralArchitectureParams import NeuralArchitectureParams
from evobpso.params.OptimizationParams import OptimizationParams
from evobpso.params.PsoParams import BooleanPSOParams
from evobpso.params.TrainingParams import TrainingParams
from evobpso.runner.OptimizationRunner import OptimizationRunner
from evobpso.utils import data_load_utils


class TestComparePerformance(TestCase):

    def test_runs_without_errors(self):
        num_of_classes = 10
        image_input_shape = (28, 28, 1)
        results_folder = os.path.join(data_load_utils.get_project_root(), 'test_tmp_results')

        data_loader = data_load_utils.load_mnist_data

        pso_params = BooleanPSOParams(c1=0.5, c2=0.5, n_bits=15, k=0.5, iters=5, mutation_prob=0.001, pop_size=5, vmax=2)
        optimizable_architecture_params = NeuralArchitectureParams(min_layers=3, max_layers=8, max_pooling_layers=2)
        training_params = TrainingParams(batch_size=128, train_eval_epochs=1, best_solution_training_epochs=1,
                                         loss='categorical_crossentropy',
                                         optimizer='rmsprop', metrics=['accuracy'])
        optimization_params = OptimizationParams(pso_params=pso_params, architecture_params=optimizable_architecture_params,
                                                 training_params=training_params)
        fixed_architecture_params = FixedArchitectureProperties(input_shape=image_input_shape, conv_stride=1, activation_function='relu',
                                                                pool_layer_kernel_size=2, pool_layer_stride=2,
                                                                padding='same', dense_layer_units=num_of_classes)

        initializer = MockInitializer(params=optimization_params)  # replace initializer with dummy so we always begin with same positions

        # BPSOVmaxVmut
        configuration = BPSOWithVmaxAndVmutConfigurationNoMaxPoolLimit(optimization_params=optimization_params,
                                                                       architecture_properties=fixed_architecture_params,
                                                                       data_loader=data_loader,
                                                                       initializer=initializer,
                                                                       results_folder=results_folder)

        runner = OptimizationRunner(configuration=configuration)
        results_1 = runner.run()

        # BPSOVmaxVumMaxPool
        configuration = BPSOWithVmaxAndVmutConfiguration(optimization_params=optimization_params,
                                                                   architecture_properties=fixed_architecture_params,
                                                                   data_loader=data_loader,
                                                                   initializer=initializer,
                                                                   results_folder=results_folder)

        runner = OptimizationRunner(configuration=configuration)
        results_2 = runner.run()


        # No Vmax no Vmut no max pooling layers
        configuration = StandardBPSOConfigurationNoMaxPoolLimit(optimization_params=optimization_params,
                                                         architecture_properties=fixed_architecture_params,
                                                         data_loader=data_loader,
                                                         initializer=initializer,
                                                         results_folder=results_folder)

        runner = OptimizationRunner(configuration=configuration)
        results_3 = runner.run()



        print(results_1['optimization_elapsed_time'])
        print(results_2['optimization_elapsed_time'])
        print(results_3['optimization_elapsed_time'])


