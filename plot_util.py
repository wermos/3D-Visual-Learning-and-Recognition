import itertools

import matplotlib.pyplot as plt
import numpy as np

import constants

# Combined
def update_combined_constants(num_objs, pca_threshold, training_data_split):
    # Constants like the number of images, image width, and image height do not
    # change across the datasets.
    consts = constants.Constants(num_objs, constants.NUM_IMAGES,
                                 constants.IMAGE_HEIGHT, constants.IMAGE_WIDTH,
                                 training_data_split, pca_threshold)
    return consts

def update_combined_constants_coil20(pca_threshold, training_data_split):
    return update_combined_constants(20, pca_threshold, training_data_split)

def update_combined_constants_coil100(pca_threshold, training_data_split):
    return update_combined_constants(100, pca_threshold, training_data_split)

def generate_combined_constants_list_helper_coil20(data):
    pca_enum, training_data_enum = data
    
    idx, pca_threshold = pca_enum
    jdx, training_data_split = training_data_enum
    const = update_combined_constants_coil20(pca_threshold, training_data_split)
    
    return (idx, jdx, const)

def generate_combined_constants_list_helper_coil100(data):
    pca_enum, training_data_enum = data
    
    idx, pca_threshold = pca_enum
    jdx, training_data_split = training_data_enum
    const = update_combined_constants_coil100(pca_threshold, training_data_split)
    
    return (idx, jdx, const)

def generate_combined_constants_list(pca_thresholds, training_data_splits, coil20=True):
    constants = itertools.product(enumerate(pca_thresholds), enumerate(training_data_splits))
    
    if coil20:
        return list(map(generate_combined_constants_list_helper_coil20, constants))
    else:
        return list(map(generate_combined_constants_list_helper_coil100, constants))

# Training
def update_training_constants(num_objs, training_data_split):
    # Constants like the number of images, image width, and image height do not
    # change across the datasets.
    consts = constants.Constants(num_objs, constants.NUM_IMAGES,
                                 constants.IMAGE_HEIGHT, constants.IMAGE_WIDTH,
                                 training_data_split, constants.PCA_THRESHOLD)
    return consts

def update_training_constants_coil20(training_data_split):
    return update_training_constants(20, training_data_split)

def update_training_constants_coil100(training_data_split):
    return update_training_constants(100, training_data_split)

def generate_training_constants_list_helper_coil20(data):
    idx, training_data_split = data
    const = update_training_constants_coil20(training_data_split)
    
    return (idx, const)

def generate_training_constants_list_helper_coil100(data):
    idx, training_data_split = data    
    const = update_training_constants_coil100(training_data_split)
    
    return (idx, const)

def generate_training_constants_list(training_data_splits, coil20=True):
    if coil20:
        return list(map(generate_training_constants_list_helper_coil20, enumerate(training_data_splits)))
    else:
        return list(map(generate_training_constants_list_helper_coil100, enumerate(training_data_splits)))

# PCA Thresholds
def update_pca_threshold_constants(num_objs, pca_threshold):
    # Constants like the number of images, image width, and image height do not
    # change across the datasets.
    consts = constants.Constants(num_objs, constants.NUM_IMAGES,
                                 constants.IMAGE_HEIGHT, constants.IMAGE_WIDTH,
                                 constants.TRAINING_PERCENTAGE, pca_threshold)
    return consts

def update_pca_threshold_constants_coil20(pca_threshold):
    return update_pca_threshold_constants(20, pca_threshold)

def update_pca_threshold_constants_coil100(pca_threshold):
    return update_pca_threshold_constants(100, pca_threshold)

def generate_pca_threshold_constants_list_helper_coil20(data):
    idx, pca_threshold = data
    const = update_pca_threshold_constants_coil20(pca_threshold)
    
    return (idx, const)

def generate_pca_threshold_constants_list_helper_coil100(data):
    idx, pca_threshold = data
    const = update_pca_threshold_constants_coil100(pca_threshold)
    
    return (idx, const)

def generate_pca_threshold_constants_list(pca_thresholds, coil20=True):    
    if coil20:
        return list(map(generate_pca_threshold_constants_list_helper_coil20, enumerate(pca_thresholds)))
    else:
        return list(map(generate_pca_threshold_constants_list_helper_coil100, enumerate(pca_thresholds)))

def plot_combined_graphs(pca_thresholds, training_data_splits, accuracy_object, accuracy_pose, mean_error):
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
    fig_1.savefig(plots_directory+'accuracy_object.png', dpi=200)

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
    fig_2.savefig(plots_directory+'accuracy_pose.png', dpi=200)

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
    fig_2.savefig(plots_directory+'mean_error.png', dpi=200)

def plot_pca_graphs(pca_thresholds, accuracy_object, accuracy_pose, mean_error):
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
    fig_1.savefig(plots_directory+'accuracy.png', dpi=200)

    fig_2, ax_2 = plt.subplots()
    ax_2.set_xlabel('PCA threshold')
    ax_2.set_xlim([0,1])
    ax_2.set_xticks(np.linspace(0,1,11))
    ax_2.set_ylabel('Average pose error (in degrees)')
    ax_2.set_title('Average pose error with varying PCA Threshold')
    ax_2.plot(pca_thresholds, mean_error, '-o', markersize=5)
    fig_2.savefig(plots_directory+'mean_error.pdf', dpi=200)
    fig_2.savefig(plots_directory+'mean_error.png', dpi=200)

def plot_training_graphs(training_data_splits, accuracy_object, accuracy_pose, mean_error):
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