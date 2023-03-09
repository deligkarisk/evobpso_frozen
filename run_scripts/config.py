
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