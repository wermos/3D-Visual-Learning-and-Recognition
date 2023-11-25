# import os
from os import environ
environ['OMP_NUM_THREADS'] = '16'

# import sys
from math import floor
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

from test import process
import constants
from plots_util import plot_directory, plot_accuracy, plot_accuracy_all_objects, plot_mean_error, plot_mean_error_all_objects, plot_error_histogram

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

# pca_thresholds = np.arange(0.05,1,0.25)
# training_data_splits = np.arange(0.05,1,0.05)
training_data_splits = np.arange(0.05,1,0.25)
testing_images = np.zeros(len(training_data_splits))

if __name__ == "__main__":
    accuracy_object = np.zeros((len(training_data_splits), constants.NUM_OBJECTS))
    accuracy_pose = np.zeros((len(training_data_splits), constants.NUM_OBJECTS))
    mean_error = [None] * len(training_data_splits)
    distances = [None] * len(training_data_splits)
    for idx, variable in tqdm(list(enumerate(training_data_splits)), desc="Generating data"):
        update_constants[1](variable)
        testing_images[idx] = constants.NUM_TESTING_IMAGES
        accuracy_object[idx], accuracy_pose[idx], mean_error[idx], distances[idx] = process(False)

    plot_accuracy(training_data_splits, np.mean(accuracy_object, axis=1)/testing_images, np.mean(accuracy_pose, axis=1)/testing_images, -1, 1)
    plot_accuracy_all_objects(training_data_splits, accuracy_object/testing_images.reshape(len(testing_images), 1), 0, 1)
    plot_accuracy_all_objects(training_data_splits, accuracy_pose/testing_images.reshape(len(testing_images), 1), 1, 1)
    for object_id in range(constants.NUM_OBJECTS):
        plot_accuracy(training_data_splits, accuracy_object[:,object_id]/testing_images, accuracy_pose[:,object_id]/testing_images, object_id, 1)

    plot_mean_error(training_data_splits, [np.mean(data) for data in mean_error] , -1, 1)#[sum([np.sum(data[idx]) for data in mean_error])/sum([len(data[idx]) for data in mean_error]) for idx in range(len(training_data_splits))]
    plot_mean_error_all_objects(training_data_splits, [data.mean(axis=1) for data in mean_error], 1)#[[np.sum(data[idx])/len(data[0]) for data in mean_error] for idx in range(len(training_data_splits))]
    for object_id in range(constants.NUM_OBJECTS):
        plot_mean_error(training_data_splits, [np.sum(data[object_id])/len(data[object_id]) for data in mean_error], object_id, 1)

    # print(testing_images)
    # print(training_data_splits.shape)
    # print([mean_error[i].shape for i in range(4)])
    plot_error_histogram(np.concatenate([np.repeat(training_data_split, np.product(mean_error[idx].shape)) for idx, training_data_split in enumerate(training_data_splits)]), np.concatenate([np.reshape(data, np.product(data.shape)) for idx, data in enumerate(mean_error)]), -1, 1)
    for object_id in range(constants.NUM_OBJECTS):
        plot_error_histogram(np.concatenate([np.repeat(training_data_split, np.product(mean_error[idx].shape)/constants.NUM_OBJECTS) for idx, training_data_split in enumerate(training_data_splits)]), np.concatenate([np.reshape(data[object_id], np.product(data[object_id].shape)) for idx, data in enumerate(mean_error)]), object_id, 1)