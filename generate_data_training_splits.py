from os import environ
environ['OMP_NUM_THREADS'] = '16'
import numpy as np
from tqdm import tqdm

from test import process
from plot_util import generate_training_constants_list, plot_training_graphs

def process_and_write(constants_tuple):
    idx, constants = constants_tuple
    
    accuracy_object[idx], accuracy_pose[idx], mean_error[idx] = process(constants)
    
    f = open('outputs/training_splits.txt','a')
    f.write(f"{constants.TRAINING_PERCENTAGE:.2f} {accuracy_object[idx]:.3%} {accuracy_pose[idx]:.3%} {mean_error[idx]:.3f}\u00b0\n")
    f.close()

training_data_splits = np.arange(0.05,1,0.05)

accuracy_object = np.zeros(len(training_data_splits))
accuracy_pose = np.zeros(len(training_data_splits))
mean_error = np.zeros(len(training_data_splits))

constants_list = generate_training_constants_list(training_data_splits, coil20=False)

# Clearing file content
open('outputs/training_splits.txt','w').close()

for constant in tqdm(constants_list, desc="Generating data"):
    process_and_write(constant)

plot_training_graphs(training_data_splits, accuracy_object, accuracy_pose, mean_error)
