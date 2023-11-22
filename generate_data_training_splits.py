import os
import sys
from math import floor
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

from test import process
import constants

def update_constants(training_data_split):
    constants.TRAINING_PERCENTAGE = training_data_split
    constants.TESTING_PERCENTAGE = 1 - constants.TRAINING_PERCENTAGE
    constants.NUM_TRAINING_IMAGES = floor(constants.NUM_IMAGES * constants.TRAINING_PERCENTAGE)
    constants.NUM_TESTING_IMAGES = constants.NUM_IMAGES - constants.NUM_TRAINING_IMAGES

if __name__ == "__main__":
    sys.stdout = open('outputs/training_splits.txt','w')
    training_data_splits = np.arange(0.05,1,0.05)
    accuracy_object = np.zeros(len(training_data_splits))
    accuracy_pose = np.zeros(len(training_data_splits))
    mean_error = np.zeros(len(training_data_splits))
    for idx, training_data_split in tqdm(list(enumerate(training_data_splits)), desc="Generating data"):
        update_constants(training_data_split)
        accuracy_object[idx], accuracy_pose[idx], mean_error[idx] = process(False)
        print(format(training_data_split, ".2f"), format(accuracy_object[idx], ".3%"), format(accuracy_pose[idx], ".3%"), format(mean_error[idx], ".3f") + "\u00b0")

    plots_directory = 'plots/training_data_split/'
    fig_1, ax_1 = plt.subplots()
    ax_1.set_xlabel('Training Data Split')
    ax_1.set_xlim([0,1])
    ax_1.set_xticks(np.linspace(0,1,11))
    ax_1.set_ylabel('Accuracy')
    ax_1.set_title('Accuracy with varying Training Data Split')
    ax_1.plot(training_data_splits, accuracy_object, '-o', markersize=5)
    ax_1.plot(training_data_splits, accuracy_pose, '-o', markersize=5)
    ax_1.set_ylim(bottom=0)
    ax_1.legend(["Object Accuracy", "Pose Accuracy"])
    fig_1.savefig(plots_directory+'accuracy.pdf', dpi=200)
    fig_1.savefig(plots_directory+'accuracy.png', dpi=200)

    fig_2, ax_2 = plt.subplots()
    ax_2.set_xlabel('Training Data Split')
    ax_2.set_xlim([0,1])
    ax_2.set_xticks(np.linspace(0,1,11))
    ax_2.set_ylabel('Average pose error (in degrees)')
    ax_2.set_title('Average pose error with varying Training Data Split')
    ax_2.plot(training_data_splits, mean_error, '-o', markersize=5)
    fig_2.savefig(plots_directory+'mean_error.pdf', dpi=200)
    fig_2.savefig(plots_directory+'mean_error.png', dpi=200)