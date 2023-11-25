from pathlib import Path
from os import environ
environ['OMP_NUM_THREADS'] = '16'
import argparse
from math import floor
import pickle

import numpy as np
from tqdm import tqdm

from test import process
import constants
from plots_util import plots_directory, title_map, title_directory_map, plot_accuracy, plot_accuracy_all_objects, plot_mean_error, plot_mean_error_all_objects, plot_error_histogram

logs_directory = 'logs/coil-' + str(constants.NUM_OBJECTS)

def update_constants_pca_threshold(pca_threshold):
    constants.PCA_THRESHOLD = pca_threshold

def update_constants_training_split(training_data_split):
    constants.TRAINING_PERCENTAGE = training_data_split
    constants.TESTING_PERCENTAGE = 1 - constants.TRAINING_PERCENTAGE
    constants.NUM_TRAINING_IMAGES = floor(constants.NUM_IMAGES * constants.TRAINING_PERCENTAGE)
    constants.NUM_TESTING_IMAGES = constants.NUM_IMAGES - constants.NUM_TRAINING_IMAGES

def update_constants_combined(pca_threshold, training_data_split):
    update_constants_pca_threshold(pca_threshold)
    update_constants_training_split(training_data_split)

update_constants = {0 : update_constants_pca_threshold, 1 : update_constants_training_split, 2 : update_constants_combined}

pca_thresholds = np.arange(0.05,1,0.05)
training_data_splits = np.arange(0.05,1,0.05)
num_testing_images = np.zeros(len(training_data_splits))
variable_map = {0: pca_thresholds, 1: training_data_splits}

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--regenerate-data", action = 'store_true')
    arguments = parser.parse_args()
    regenerate_data = bool(arguments.regenerate_data)

    print("Number of objects", constants.NUM_OBJECTS)
    for i in range(2):
        if regenerate_data:
            accuracy_object = np.zeros((len(variable_map[i]), constants.NUM_OBJECTS))
            accuracy_pose = np.zeros((len(variable_map[i]), constants.NUM_OBJECTS))
            mean_error = [None] * len(variable_map[i])
            distances = [None] * len(variable_map[i])
            for idx, variable in tqdm(list(enumerate(variable_map[i])), desc="Generating data by varying " + title_map[i]):
                update_constants[i](variable)
                num_testing_images[idx] = constants.NUM_TESTING_IMAGES
                accuracy_object[idx], accuracy_pose[idx], mean_error[idx], distances[idx] = process(False)
        else:
            print("Loading data with varying " + title_map[i] + "...")
            filename = logs_directory + '/' + str(title_directory_map[i]) + '.pkl'
            with open(filename, 'rb') as file:
                accuracy_object, accuracy_pose, mean_error, distances = pickle.load(file)
            for idx, variable in enumerate(variable_map[i]):
                update_constants[i](variable)
                num_testing_images[idx] = constants.NUM_TESTING_IMAGES

        Path(plots_directory + title_directory_map[i]).mkdir(parents=True, exist_ok=True)
        print("Generating accuracy plots...")
        plot_accuracy(variable_map[i], np.mean(accuracy_object, axis=1)/num_testing_images, np.mean(accuracy_pose, axis=1)/num_testing_images, -1, i)
        plot_accuracy_all_objects(training_data_splits, accuracy_object/num_testing_images.reshape(len(num_testing_images), 1), 0, i)
        plot_accuracy_all_objects(training_data_splits, accuracy_pose/num_testing_images.reshape(len(num_testing_images), 1), 1, i)
        for object_id in range(constants.NUM_OBJECTS):
            plot_accuracy(variable_map[i], accuracy_object[:,object_id]/num_testing_images, accuracy_pose[:,object_id]/(num_testing_images), object_id, i)

        print("Generating error plots...")
        plot_mean_error(variable_map[i], [np.mean(data) for data in mean_error], -1, i)
        plot_mean_error_all_objects(variable_map[i], [data.mean(axis=1) for data in mean_error] , i)
        for object_id in range(constants.NUM_OBJECTS):
            plot_mean_error(variable_map[i], [np.mean(data[object_id]) for data in mean_error], object_id, i)

        print("Generating histograms...")
        x = np.concatenate([np.repeat(variable, np.product(mean_error[idx].shape)) for idx, variable in enumerate(variable_map[i])])
        y = np.concatenate([np.reshape(data, np.product(data.shape)) for idx, data in enumerate(mean_error)])
        plot_error_histogram(x, y, -1, i)
        for object_id in range(constants.NUM_OBJECTS):
            x = np.concatenate([np.repeat(variable, np.product(mean_error[idx].shape)/constants.NUM_OBJECTS) for idx, variable in enumerate(variable_map[i])])
            y = np.concatenate([np.reshape(data[object_id], np.product(data[object_id].shape)) for idx, data in enumerate(mean_error)])
            plot_error_histogram(x, y, object_id, i)

        if regenerate_data:
            Path(logs_directory).mkdir(parents=True, exist_ok=True)
            filename = logs_directory + '/' + str(title_directory_map[i]) + '.pkl'
            with open(filename, 'wb') as file:
                pickle.dump([accuracy_object, accuracy_pose, mean_error, distances], file)