import numpy as np
from tqdm import tqdm

from plot_util import generate_constants_list, plot_combined_graphs
from test import process

def process_and_write(constants_tuple):
    idx, jdx, constants = constants_tuple
    
    accuracy_object[idx][jdx], accuracy_pose[idx][jdx], mean_error[idx][jdx] = process(constants)   
    
    f = open('outputs/combined.txt', 'a')
    f.write(f"{constants.PCA_THRESHOLD:.2f} {constants.TRAINING_PERCENTAGE:.2f} {accuracy_object[idx][jdx]:.3%} {accuracy_pose[idx][jdx]:.3%} {mean_error[idx][jdx]:.3f}\u00b0\n")
    f.close()

pca_thresholds = np.arange(0.1,1,0.05)
training_data_splits = np.arange(0.1,1,0.05)

accuracy_object = np.zeros((len(pca_thresholds), len(training_data_splits)))
accuracy_pose = np.zeros((len(pca_thresholds), len(training_data_splits)))
mean_error = np.zeros((len(pca_thresholds), len(training_data_splits)))

constants_list = generate_constants_list(pca_thresholds, training_data_splits)

# Clearing file content
open('outputs/combined.txt','w').close()
for constant in tqdm(constants_list, desc="Generating data"):
    process_and_write(constant)

print("Plotting and saving the plots...")
plot_combined_graphs(pca_thresholds, training_data_splits, accuracy_object, accuracy_pose, mean_error)
print("Finished.")
