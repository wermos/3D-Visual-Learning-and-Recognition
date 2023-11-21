import os
import sys
import numpy as np
from tqdm import tqdm

from test import process
import constants

if __name__ == "__main__":
    sys.stdout = open('output.txt','w')
    pca_thresholds = np.arange(0.1,1,0.05)
    # training_data_splits = np.arange(0.1,1,0.1)
    accuracy_object = np.zeros(len(pca_thresholds))
    accuracy_pose = np.zeros(len(pca_thresholds))
    mean_error = np.zeros(len(pca_thresholds))
    for idx, pca_threshold in enumerate(pca_thresholds):
        # for training_data_split in training_data_splits:
            # Set relevant values in constants library
            constants.PCA_THRESHOLD = pca_threshold
            accuracy_object[idx], accuracy_pose[idx], mean_error[idx] = process(False)
            print(format(pca_threshold, ".2f"), format(accuracy_object[idx], ".3%"), format(accuracy_pose[idx], ".3%"), format(mean_error[idx], ".3f") + "\u00b0")

