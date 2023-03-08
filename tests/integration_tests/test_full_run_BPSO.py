import os
import pickle
from pathlib import Path

from configuration.StandardBPSORunner import StandardBPSORunner
from params.FixedArchitectureProperties import FixedArchitectureProperties
from params.NeuralArchitectureParams import NeuralArchitectureParams
from params.OptimizationParams import OptimizationParams
from params.PsoParams import BooleanPSOParams
from params.TrainingParams import TrainingParams
from utils.data_load_utils import load_mnist_data
from utils.filesystem_utils import save_object

# Experiment name
experiment_name = 'FirstRun'  # Give a meaningful name, it will be used as a sub-folder for saving data

# Basic PSO parameters
population_size = 3
iterations = 10
c1 = 0.5
c2 = 0.5
n_bits = 14
k = 0.5

# Particle initialization parameters
min_layers = 1
max_layers = 2

# Training parameters
train_eval_epochs = 1  # Number of epochs to run during the optimization procedure
best_solution_training_epochs = 100  # Number of epochs to run for the global best position
batch_size = 64
optimizer = 'adam'
loss = 'sparse_categorical_crossentropy'
metrics = ['accuracy']
batch_normalization = True
dropout = False

# Fixed architecture properties
conv_stride = 1
activation_function = 'relu'
pool_layer_kernel_size = 2
pool_layers_stride = 2
padding = 'same'

# Dataset parameters
num_classes = 10
image_input_shape = (28, 28, 1)

# Output folder
results_folder = '/home/kosmas-deligkaris/DeepBPSOResults'

results_folder = os.path.join(results_folder, experiment_name)
Path(results_folder).mkdir(parents=True, exist_ok=True)


pso_params = BooleanPSOParams(pop_size=population_size, iters=iterations, c1=c1, c2=c2, n_bits=n_bits, k=k)
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

data_loader = load_mnist_data

params_to_save = {'optimization_params': optimization_params, 'fixed_architecture_properties': fixed_architecture_properties,
                  'training_params': training_params}
save_object(params_to_save, os.path.join(results_folder, 'params.pickle'))


runner = StandardBPSORunner(optimization_params=optimization_params,
                            architecture_properties=fixed_architecture_properties,
                            data_loader=data_loader,
                            results_folder=results_folder)

runner.run()


