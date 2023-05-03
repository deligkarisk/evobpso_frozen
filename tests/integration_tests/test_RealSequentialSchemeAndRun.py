from unittest import TestCase

import tests.test_functions.test_functions
from evobpso.encoding.RealEncoding import RealEncoding
from evobpso.params.FixedArchitectureProperties import FixedArchitectureProperties
from evobpso.params.NeuralArchitectureParams import NeuralArchitectureParams
from evobpso.params.OptimizationParams import OptimizationParams
from evobpso.params.PsoParams import RealPSOParams
from evobpso.params.TrainingParams import TrainingParams
from evobpso.population.Population import Population
from evobpso.runner.OptimizationRunner import OptimizationRunner
from evobpso.scheme.SequentialScheme import SequentialScheme


class TestRealSequentialSchemeAndRun(TestCase):

    def test_real_sequential_scheme_and_run(self):
        encoding = RealEncoding(min_conv_filters=2, max_conv_filters=12, min_kernel_size=2, max_kernel_size=4)

        pso_params = RealPSOParams(pop_size=2, iters=2, c1=0.5, c2=0.5, k=1)
        neural_architecture_params = NeuralArchitectureParams(min_layers=2, max_layers=4, max_pooling_layers=2)
        fixed_architecture_properties = FixedArchitectureProperties(input_shape=(28, 28, 1),
                                                                    conv_stride=2,
                                                                    activation_function='relu',
                                                                    pool_layer_kernel_size=2,
                                                                    pool_layer_stride=2,
                                                                    padding='same',
                                                                    dense_layer_units=10)

        training_params = TrainingParams(batch_size=2,
                                         train_eval_epochs=1,
                                         best_solution_training_epochs=1,
                                         loss='categorical_crossentropy',
                                         optimizer='adam',
                                         metrics=['accuracy'])

        data_loader = tests.test_functions.test_functions.testdata_loader

        optimization_params = OptimizationParams(pso_params=pso_params,
                                                 neural_architecture_params=neural_architecture_params,
                                                 training_params=training_params)

        scheme = SequentialScheme(version='real', variable_length=True, vmax=False, vmut=False, optimization_params=optimization_params,
                                  encoding=encoding)
        scheme.compile(fixed_architecture_properties=fixed_architecture_properties, data_loader=data_loader, results_folder='temp')
        population = Population(scheme)
        runner = OptimizationRunner(population=population)
        runner.run()
