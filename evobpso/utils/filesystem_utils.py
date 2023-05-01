import os
import pickle


def save_params(optimization_params, fixed_architecture_properties, encoding, filename):
    params_to_save = {'optimization_params': optimization_params, 'fixed_architecture_properties': fixed_architecture_properties,
                      'encoding': encoding}
    save_object(params_to_save, filename)


def save_object(object, filename):
    with open(filename, 'wb') as handle:
        pickle.dump(object, handle, protocol=pickle.HIGHEST_PROTOCOL)


def get_save_folder(results_folder, iter_number, particle_index):
    if results_folder is not None:
        save_folder = os.path.join(results_folder, 'iter_' + str(iter_number), 'particle_' + str(particle_index))
    else:
        save_folder = None
    return save_folder

