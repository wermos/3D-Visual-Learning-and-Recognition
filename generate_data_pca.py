import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

from test import process
import constants

if __name__ == "__main__":
    sys.stdout = open('outputs/pca.txt','w')
    pca_thresholds = np.arange(0.1,1,0.05)
    accuracy_object = np.zeros(len(pca_thresholds))
    accuracy_pose = np.zeros(len(pca_thresholds))
    mean_error = np.zeros(len(pca_thresholds))
    for idx, pca_threshold in tqdm(enumerate(pca_thresholds), desc="generating data..."):
        constants.PCA_THRESHOLD = pca_threshold
        accuracy_object[idx], accuracy_pose[idx], mean_error[idx] = process(False)
        print(format(pca_threshold, ".2f"), format(accuracy_object[idx], ".3%"), format(accuracy_pose[idx], ".3%"), format(mean_error[idx], ".3f") + "\u00b0")

    plots_directory = 'plots/pca_threshold/'
    fig_1, ax_1 = plt.subplots()
    ax_1.set_xlabel('PCA threshold')
    ax_1.set_xlim([0,1])
    ax_1.set_xticks(np.linspace(0,1,11))
    ax_1.set_ylabel('Accuracy')
    ax_1.set_title('Accuracy with varying PCA Threshold')
    ax_1.plot(pca_thresholds, accuracy_object, '-o', markersize=5)
    ax_1.plot(pca_thresholds, accuracy_pose, '-o', markersize=5)
    ax_1.set_ylim(bottom=0)
    ax_1.legend(["Object Accuracy", "Pose Accuracy"])
    fig_1.savefig(plots_directory+'accuracy.pdf', dpi=200)
    fig_1.savefig(plots_directory+'accuracy.png')

    fig_2, ax_2 = plt.subplots()
    ax_2.set_xlabel('PCA threshold')
    ax_2.set_xlim([0,1])
    ax_2.set_xticks(np.linspace(0,1,11))
    ax_2.set_ylabel('Average pose error (in degrees)')
    ax_2.set_title('Average pose error with varying PCA Threshold')
    ax_2.plot(pca_thresholds, mean_error, '-o', markersize=5)
    fig_2.savefig(plots_directory+'mean_error.pdf', dpi=200)
    fig_2.savefig(plots_directory+'mean_error.png')