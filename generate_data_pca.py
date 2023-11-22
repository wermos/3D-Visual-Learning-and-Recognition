import numpy as np
from tqdm import tqdm

from plot_util import generate_pca_threshold_constants_list, plot_pca_graphs
from test import process

def process_and_write(constants_tuple):
    idx, constants = constants_tuple
    accuracy_object[idx], accuracy_pose[idx], mean_error[idx] = process(constants)
    
    f = open('outputs/pca.txt','a')
    f.write(f"{constants.PCA_THRESHOLD:.2f} {accuracy_object[idx]:.3%} {accuracy_pose[idx]:.3%} {mean_error[idx]:.3f}\u00b0\n")
    f.close()

pca_thresholds = np.arange(0.1,1,0.05)
accuracy_object = np.zeros(len(pca_thresholds))
accuracy_pose = np.zeros(len(pca_thresholds))
mean_error = np.zeros(len(pca_thresholds))

constants_list = generate_pca_threshold_constants_list(pca_thresholds)

# Clearing file content
open('outputs/pca.txt','w').close()

for constant in tqdm(constants_list, desc="Generating data"):
    process_and_write(constant)

plot_pca_graphs(pca_thresholds, accuracy_object, accuracy_pose, mean_error)
