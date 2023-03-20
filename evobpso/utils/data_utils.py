import random
from pathlib import Path

from data_load_utils import load_mnist_data, load_convex_data, get_project_root, load_mnist_background_images, load_mnist_background_random, \
    load_mnist_rotation, load_mnist_rotation_background, load_rectangles, load_rectangles_images
import os
from matplotlib import pyplot as plt


def extract_samples_from_datasets():
    dataloaders = [load_mnist_data, load_mnist_background_images,
                   load_mnist_background_random, load_mnist_rotation, load_mnist_rotation_background,
                   load_rectangles, load_rectangles_images, load_convex_data]
    num_figs = 30

    for dataloader in dataloaders:
        save_folder_location = os.path.join(get_project_root(), 'data_samples', dataloader.__name__)
        Path(save_folder_location).mkdir(parents=True, exist_ok=True)
        (x_train, y_train), (x_valid, y_valid), (x_test, y_test) = dataloader()

        for k in range(0, num_figs):
            im_num = random.randint(0, len(x_train))
            image = x_train[im_num, :]
            label = y_train[im_num]
            plt.imshow(image, cmap=plt.get_cmap('gray'))
            plt.savefig(os.path.join(save_folder_location, 'train_' + str(im_num) + '_' + str(label) + '.tiff'))

        for k in range(0, num_figs):
            im_num = random.randint(0, len(x_valid))
            image = x_valid[im_num, :]
            label = y_valid[im_num]
            plt.imshow(image, cmap=plt.get_cmap('gray'))
            plt.savefig(os.path.join(save_folder_location, 'valid_' + str(im_num) + '_' + str(label) + '.tiff'))

        for k in range(0, num_figs):
            im_num = random.randint(0, len(x_test))
            image = x_test[im_num, :]
            label = y_test[im_num]
            plt.imshow(image, cmap=plt.get_cmap('gray'))
            plt.savefig(os.path.join(save_folder_location, 'test_' + str(im_num) + '_' + str(label) + '.tiff'))


extract_samples_from_datasets()
