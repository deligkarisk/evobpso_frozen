import os
import pickle
from pathlib import Path

from configuration.StandardBPSORunner import StandardBPSORunner
from params.FixedArchitectureProperties import FixedArchitectureProperties
from params.NeuralArchitectureParams import NeuralArchitectureParams
from params.OptimizationParams import OptimizationParams
from params.PsoParams import BooleanPSOParams
from params.TrainingParams import TrainingParams
from utils.data_load_utils import load_mnist_data, load_mnist_background_images, load_mnist_background_random, load_mnist_rotation, \
    load_mnist_rotation_background, load_rectangles, load_rectangles_images, load_convex_data
from utils.filesystem_utils import save_object

# import common config parameters
from config import *

# Experiment name
experiment_name = 'first_run'
dataset_names = ['mnist', 'mnist_bg', 'mnist_bg_rnd', 'mnist_rot', 'mnist_bg_rot', 'rectangle', 'rect_images',
                    'convex']  # Give a meaningful name, it will be used as a sub-folder for saving data
data_loaders = [load_mnist_data, load_mnist_background_images, load_mnist_background_random, load_mnist_rotation,
                load_mnist_rotation_background, load_rectangles, load_rectangles_images, load_convex_data]

# Output folder
results_folder = '/home/kosmas-deligkaris/DeepBPSOResults'

for i in range(0, len(dataset_names)):
    print('analyzing dataset: ' + dataset_names[i])
    results_folder = os.path.join(results_folder, experiment_name, dataset_names[i])
    data_loader = data_loaders[i]

    Path(results_folder).mkdir(parents=True, exist_ok=True)

    pso_params = BooleanPSOParams(pop_size=population_size, iters=iterations, c1=c1, c2=c2, n_bits=n_bits, k=k, mutation_prob=0)
    optimizable_architecture_params = NeuralArchitectureParams(min_layers=min_layers, max_layers=max_layers)
    training_params = TrainingParams(batch_size=batch_size, epochs=train_eval_epochs, loss=loss,
                                     optimizer=optimizer, metrics=metrics)

    optimization_params = OptimizationParams(pso_params=pso_params, architecture_params=optimizable_architecture_params,
                                             training_params=training_params)
    fixed_architecture_properties = FixedArchitectureProperties(input_shape=image_input_shape, conv_stride=conv_stride,
                                                                activation_function=activation_function,
                                                                pool_layer_kernel_size=pool_layer_kernel_size,
                                                                pool_layer_stride=pool_layers_stride,
                                                                padding=padding,
                                                                dense_layer_units=num_classes)

    params_to_save = {'optimization_params': optimization_params, 'fixed_architecture_properties': fixed_architecture_properties,
                      'training_params': training_params}
    save_object(params_to_save, os.path.join(results_folder, 'params.pickle'))

    runner = StandardBPSORunner(optimization_params=optimization_params,
                                architecture_properties=fixed_architecture_properties,
                                data_loader=data_loader,
                                results_folder=results_folder)

    runner.run()
