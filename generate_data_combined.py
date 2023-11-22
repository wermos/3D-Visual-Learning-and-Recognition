import sys
from math import floor
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

from test import process
import constants

def update_constants(pca_threshold, training_data_split):
    constants.PCA_THRESHOLD = pca_threshold
    constants.TRAINING_PERCENTAGE = training_data_split
    constants.TESTING_PERCENTAGE = 1 - constants.TRAINING_PERCENTAGE
    constants.NUM_TRAINING_IMAGES = floor(constants.NUM_IMAGES * constants.TRAINING_PERCENTAGE)
    constants.NUM_TESTING_IMAGES = constants.NUM_IMAGES - constants.NUM_TRAINING_IMAGES

if __name__ == "__main__":
    sys.stdout = open('outputs/combined.txt','w')
    pca_thresholds = np.arange(0.1,1,0.05)
    training_data_splits = np.arange(0.1,1,0.05)
    
    accuracy_object = np.zeros((len(pca_thresholds), len(training_data_splits)))
    accuracy_pose = np.zeros((len(pca_thresholds), len(training_data_splits)))
    mean_error = np.zeros((len(pca_thresholds), len(training_data_splits)))
    
    for idx, pca_threshold in tqdm(list(enumerate(pca_thresholds)), desc="Generating data"):
        for jdx, training_data_split in enumerate(training_data_splits):
            update_constants(pca_threshold, training_data_split)
            
            accuracy_object[idx][jdx], accuracy_pose[idx][jdx], mean_error[idx][jdx] = process(False)
            print(format(pca_threshold, ".2f"), format(training_data_split, ".2f"), format(accuracy_object[idx][jdx], ".3%"), format(accuracy_pose[idx][jdx], ".3%"), format(mean_error[idx][jdx], ".3f") + "\u00b0")

    pca_thresholds, training_data_splits = np.meshgrid(pca_thresholds, training_data_splits)
    plots_directory = 'plots/combined/'
    fig_1 = plt.figure()
    ax_1 = plt.axes(projection ='3d')
    ax_1.set_xlabel('PCA threshold')
    ax_1.set_ylabel('Training Data Split')
    ax_1.set_zlabel('Object Recognition Accuracy')
    ax_1.set_xticks(np.linspace(0,1,6))
    ax_1.set_yticks(np.linspace(0,1,6))
    ax_1.set_zlim([0,1])
    ax_1.set_title('Object Accuracy with varying parameters')
    ax_1.plot_wireframe(pca_thresholds, training_data_splits, accuracy_object)
    fig_1.savefig(plots_directory+'accuracy_object.pdf', dpi=200)
    fig_1.savefig(plots_directory+'accuracy_object.png')

    fig_2 = plt.figure()
    ax_2 = plt.axes(projection ='3d')
    ax_2.set_xlabel('PCA threshold')
    ax_2.set_ylabel('Training Data Split')
    ax_2.set_zlabel('Pose Recognition Accuracy')
    ax_2.set_xticks(np.linspace(0,1,6))
    ax_2.set_yticks(np.linspace(0,1,6))
    ax_2.set_zlim([0,1])
    ax_2.set_title('Pose Accuracy with varying parameters')
    ax_2.plot_wireframe(pca_thresholds, training_data_splits, accuracy_pose)
    fig_2.savefig(plots_directory+'accuracy_pose.pdf', dpi=200)
    fig_2.savefig(plots_directory+'accuracy_pose.png')

    fig_3 = plt.figure()
    ax_3 = plt.axes(projection ='3d')
    ax_3.set_xlabel('PCA threshold')
    ax_3.set_ylabel('Training Data Split')
    ax_3.set_zlabel('Average pose error (in degrees)')
    ax_3.set_xticks(np.linspace(0,1,6))
    ax_3.set_yticks(np.linspace(0,1,6))
    ax_3.set_title('Average pose error with varying parameters')
    ax_3.plot_wireframe(pca_thresholds, training_data_splits, mean_error)
    fig_3.savefig(plots_directory+'mean_error.pdf', dpi=200)
    fig_2.savefig(plots_directory+'mean_error.png')