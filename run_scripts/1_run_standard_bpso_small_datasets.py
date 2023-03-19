import copy
import os
from pathlib import Path

from configuration.StandardBPSOConfiguration import StandardBPSOConfiguration
from run_scripts.param_loaders.standard_bpso_param_loader import load_standard_bpso_params
from utils.algorithm_utils import train_and_evaluate_global_best
from utils.data_load_utils import load_mnist_data, load_mnist_background_images, load_mnist_background_random, load_mnist_rotation, \
    load_mnist_rotation_background, load_rectangles, load_rectangles_images, load_convex_data
from utils.filesystem_utils import save_object, save_params

# import common config parameters
from utils.utils import set_seed

# set the param loaders and configuration class.
# If you would like to change the behavior of the algorithm, e.g. to add mutation, create new config loaders or new classes and
# change the settings here.
param_loader_partial = load_standard_bpso_params
config_class = StandardBPSOConfiguration

run_times = 10
experiment_name = 'standard_bpso_small_datasets'
dataset_names = ['mnist', 'mnist_bg', 'mnist_bg_rnd', 'mnist_rot', 'mnist_bg_rot', 'rectangle', 'rect_images',
                 'convex']  # Give a meaningful name, it will be used as a sub-folder for saving data
data_loaders = [load_mnist_data, load_mnist_background_images, load_mnist_background_random, load_mnist_rotation,
                load_mnist_rotation_background, load_rectangles, load_rectangles_images, load_convex_data]
num_classes = [10, 10, 10, 10, 10, 2, 2, 2]

# Output folder
results_base_folder = '/home/kosmas-deligkaris/DeepBPSOResults'

# set random seed for reproducible results
set_seed(82)

for k in range(0, run_times):
    results_base_folder_run = os.path.join(results_base_folder, experiment_name, 'run_' + str(k))
    for i in range(0, len(dataset_names)):
        print('analyzing dataset: ' + dataset_names[i])
        results_folder = os.path.join(results_base_folder_run, dataset_names[i])
        data_loader = data_loaders[i]
        num_classes_in_dataset = num_classes[i]
        Path(results_folder).mkdir(parents=True, exist_ok=True)

        pso_params, optimizable_architecture_params, training_params, optimization_params, fixed_architecture_properties = \
            param_loader_partial(num_classes_in_dataset)

        runner = config_class(optimization_params=optimization_params,
                              architecture_properties=fixed_architecture_properties,
                              data_loader=data_loader,
                              results_folder=results_folder)

        # Run initial optimization
        run_results = runner.run()

        # Based on the initial optimization results, evaluate the global best solution for a large number of epochs
        run_results['best_position_test_results'] = train_and_evaluate_global_best(run_results, runner, training_params, results_folder)

        # Save all associated parameters and results
        save_object(run_results, os.path.join(results_folder, 'run_results.pickle'))
        save_params(optimization_params, fixed_architecture_properties, training_params, os.path.join(results_folder, 'params.pickle'))

        print('Finished analyzing dataset ' + dataset_names[i])
    print('Finished run #' + str(k))
