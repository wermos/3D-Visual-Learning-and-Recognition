# import os
from os import environ
environ['OMP_NUM_THREADS'] = '16'

# import sys
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

from test import process
import constants
from plots_util import plot_accuracy, plot_accuracy_all_objects, plot_mean_error, plot_mean_error_all_objects, plot_error_histogram

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

if __name__ == "__main__":
    # sys.stdout = open('outputs/pca.txt','w')
    # pca_thresholds = np.arange(0.05,1,0.25)
    pca_thresholds = np.arange(0.05,1,0.05)
    accuracy_object = np.zeros((len(pca_thresholds), constants.NUM_OBJECTS))
    accuracy_pose = np.zeros((len(pca_thresholds), constants.NUM_OBJECTS))
    mean_error = np.zeros((len(pca_thresholds), constants.NUM_OBJECTS, constants.NUM_TESTING_IMAGES))
    distances = np.zeros((len(pca_thresholds), constants.NUM_OBJECTS, constants.NUM_TESTING_IMAGES, 2))
    for idx, pca_threshold in tqdm(list(enumerate(pca_thresholds)), desc="Generating data"):
        update_constants[0](pca_threshold)
        accuracy_object[idx], accuracy_pose[idx], mean_error[idx], distances[idx] = process(False)
        # print(format(pca_threshold, ".2f"), format(accuracy_object[idx], ".3%"), format(accuracy_pose[idx], ".3%"), format(mean_error[idx], ".3f") + "\u00b0")

    plot_accuracy(pca_thresholds, np.mean(accuracy_object, axis=1)/constants.NUM_TESTING_IMAGES, np.mean(accuracy_pose, axis=1)/constants.NUM_TESTING_IMAGES, -1, 0)
    plot_accuracy_all_objects(pca_thresholds, accuracy_object/constants.NUM_TESTING_IMAGES, 0, 0)
    plot_accuracy_all_objects(pca_thresholds, accuracy_pose/constants.NUM_TESTING_IMAGES, 1, 0)
    for object_id in range(constants.NUM_OBJECTS):
        plot_accuracy(pca_thresholds, accuracy_object[:,object_id]/constants.NUM_TESTING_IMAGES, accuracy_pose[:,object_id]/(constants.NUM_TESTING_IMAGES), object_id, 0)

    plot_mean_error(pca_thresholds, np.mean(mean_error, axis=(1,2)), -1, 0)
    plot_mean_error_all_objects(pca_thresholds, np.mean(mean_error, axis=2), 0)
    for object_id in range(constants.NUM_OBJECTS):
        plot_mean_error(pca_thresholds, np.mean(mean_error[:,object_id], axis=1), object_id, 0)

    plot_error_histogram(np.repeat(pca_thresholds, constants.NUM_OBJECTS*constants.NUM_TESTING_IMAGES), mean_error.reshape(len(pca_thresholds)*constants.NUM_OBJECTS*constants.NUM_TESTING_IMAGES) , -1, 0)
    for object_id in range(constants.NUM_OBJECTS):
        plot_error_histogram(np.repeat(pca_thresholds, constants.NUM_TESTING_IMAGES), mean_error[:,object_id].reshape(len(pca_thresholds)*constants.NUM_TESTING_IMAGES), object_id, 0)