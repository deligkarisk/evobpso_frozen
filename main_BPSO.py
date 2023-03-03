import os
import pickle
import time
from pathlib import Path
import tensorflow as tf

import utils.data_load_utils
from architecture_decoder.StandardArchitectureDecoder import StandardArchitectureDecoder
from component_creator.StandardBooleanComponentCreator import StandardBooleanComponentCreator
from component_data_calculator.StandardBoolenComponentDataCalculator import StandardBooleanComponentDataCalculator
from evaluator.StandardNNEvaluator import StandardNNEvaluator
from initializer.BinaryInitializer import BinaryInitializer
from model_creator.TensorflowModelCreator import TensorflowModelCreator
from params.FixedArchitectureParams import FixedArchitectureParams
from params.NeuralArchitectureParams import NeuralArchitectureParams
from params.OptimizationParams import OptimizationParams
from params.PsoParams import BooleanPSOParams
from params.TrainingParams import TrainingParams
from population.Population import Population
from position_update_strategy.StandardPositionUpdateStrategy import StandardPositionUpdateStrategy
from position_validator.DoNothingPositionValidator import DoNothingPositionValidator
from velocity_update_strategy.StandardVelocityUpdateStrategy import StandardVelocityUpdateStrategy


def test_runs_without_errors():
    start_time = time.time()
    num_of_classes = 10
    image_input_shape = (28, 28, 1)
    population_size = 20
    iterations = 10
    results_folder = os.path.join(utils.data_load_utils.get_project_root(), 'test_tmp_results')
    pso_params = BooleanPSOParams(c1=0.5, c2=0.5, n_bits=14, k=0.5)
    optimizable_architecture_params = NeuralArchitectureParams(min_layers=5, max_layers=5)
    all_params = OptimizationParams(pso_params=pso_params, architecture_params=optimizable_architecture_params)
    fixed_architecture_params = FixedArchitectureParams(input_shape=image_input_shape, conv_stride=1, activation_function='relu',
                                                        pool_layer_kernel_size=2, pool_layer_stride=2,
                                                        padding='same', dense_layer_units=num_of_classes)
    training_params = TrainingParams(batch_size=32, epochs=1, loss='sparse_categorical_crossentropy',
                                     optimizer='rmsprop', metrics=['accuracy'])
    data_loader = utils.data_load_utils.load_mnist_data
    data_calculator = StandardBooleanComponentDataCalculator(params=all_params)
    component_creator = StandardBooleanComponentCreator(data_calculator=data_calculator)
    velocity_strategy = StandardVelocityUpdateStrategy(component_creator=component_creator, params=all_params)
    position_update_strategy = StandardPositionUpdateStrategy()
    decoder = StandardArchitectureDecoder()
    model_creator = TensorflowModelCreator(fixed_architecture_params=fixed_architecture_params)
    evaluator = StandardNNEvaluator(architecture_decoder=decoder, model_creator=model_creator,
                                    training_params=training_params, data_loader=data_loader)
    validator = DoNothingPositionValidator()
    initializer = BinaryInitializer(params=all_params)
    population = Population(pop_size=population_size, params=all_params, validator=validator, initializer=initializer,
                            evaluator=evaluator, velocity_update_strategy=velocity_strategy,
                            position_update_strategy=position_update_strategy)

    aggregated_history = []

    try:
        results = population.iterate(first_iter=True)
        aggregated_history.append(results)
        for i in range(0, iterations):
            print('starting iteration: ' + str(i))
            results = population.iterate(first_iter=False)
            aggregated_history.append(results)
        print("OK")
    except Exception as e:
        print("problem")

    end_time = time.time()
    elapsed_time = ((end_time - start_time) / 60) / 60
    print("Runtime: " + str(elapsed_time) + " hours.")

    Path(results_folder).mkdir(parents=True, exist_ok=True)
    filename = os.path.join(results_folder, 'aggregated_history.pickle')
    with open(filename, 'wb') as handle:
        pickle.dump(aggregated_history, handle, protocol=pickle.HIGHEST_PROTOCOL)

    params_to_save = {'all_params': all_params, 'fixed_architecture_params': fixed_architecture_params, 'training_params': training_params}
    filename = os.path.join(results_folder, 'params.pickle')
    with open(filename, 'wb') as handle:
        pickle.dump(params_to_save, handle, protocol=pickle.HIGHEST_PROTOCOL)

    filename = os.path.join(results_folder, 'population.pickle')
    with open(filename, 'wb') as handle:
        pickle.dump(population, handle, protocol=pickle.HIGHEST_PROTOCOL)



physical_devices = tf.config.list_physical_devices('GPU')
print(physical_devices)
tf.config.set_visible_devices(physical_devices[0], 'GPU')

gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        # Currently, memory growth needs to be the same across GPUs
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        logical_gpus = tf.config.list_logical_devices('GPU')
        print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
    except RuntimeError as e:
        # Memory growth must be set before GPUs have been initialized
        print(e)


test_runs_without_errors()
