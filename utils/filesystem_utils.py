import os
import pickle


def save_object(object, filename):
    with open(filename, 'wb') as handle:
        pickle.dump(object, handle, protocol=pickle.HIGHEST_PROTOCOL)


def get_save_folder(results_folder, iter_number, particle_index):
    if results_folder is not None:
        save_folder = os.path.join(results_folder, 'iter_' + str(iter_number), 'particle_' + str(particle_index))
    else:
        save_folder = None
    return save_folder

