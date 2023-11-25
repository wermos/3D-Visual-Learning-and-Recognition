import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

import constants
plots_directory = 'plots/coil-' + str(constants.NUM_OBJECTS) + '/'

title_map = {0: 'PCA Threshold', 1: 'Training Data Split', 2: 'PCA Threshold and Training Data Split'}
title_directory_map = {0: 'pca_threshold', 1: 'training_data_split',  2: 'combined'}
accuracy_map = {0: 'Object', 1: 'Pose'}
accuracy_directory_map = {0: 'object', 1: 'pose'}
sub_directories = [['accuracy', 'mean_error', 'error_histogram'], ['accuracy', 'mean_error', 'error_histogram'], ['object_accuracy', 'pose_accuracy', 'mean_error']]

def save_fig(figure, filename):
    figure.savefig(plots_directory + filename + '.pdf', dpi=200)
    figure.savefig(plots_directory + filename + '.png', dpi=200)

def object_directory_name(object_id):
    return '_overall' if object_id == -1 else '/' + str(object_id) #'object_' + 

def object_name(object_id):
    return 'overall' if object_id == -1 else 'object$_{' + str(object_id) + '}$'

def plot_accuracy(x_axis, y_axis_1, y_axis_2, object_id, plot_type):
    fig, ax = plt.subplots()
    ax.set_xlabel(title_map[plot_type])
    ax.set_xlim([0,1])
    ax.set_xticks(np.linspace(0,1,11))
    ax.set_ylabel('Accuracy')
    ax.set_title('Accuracy ' + object_name(object_id) + ' with varying ' + title_map[plot_type])
    ax.plot(x_axis, y_axis_1, '-o', markersize=5)
    ax.plot(x_axis, y_axis_2, '-o', markersize=5)
    ax.set_ylim(bottom=0)
    ax.legend(["Object Accuracy", "Pose Accuracy"])
    save_fig(fig, title_directory_map[plot_type] + '/' +'accuracy' + object_directory_name(object_id))
    plt.close()

def plot_accuracy_all_objects(x_axis, y_axis, accuracy_type, plot_type):
    fig, ax = plt.subplots()
    ax.set_xlabel(title_map[plot_type])
    ax.set_xlim([0,1])
    ax.set_xticks(np.linspace(0,1,11))
    ax.set_ylabel(accuracy_map[accuracy_type] + ' Accuracy')
    ax.set_title(accuracy_map[accuracy_type] + ' Accuracy for all objects with varying ' + title_map[plot_type])
    ax.plot(x_axis, y_axis, '-o', markersize=3, linewidth=0.2)
    ax.set_ylim(bottom=0)
    ax.legend([str(object_id) for object_id in range(constants.NUM_OBJECTS)], prop={'size': 1.7})
    save_fig(fig, title_directory_map[plot_type] + '/' +accuracy_directory_map[accuracy_type] + '_accuracy_all_objects')
    plt.close()

def plot_mean_error(x_axis, y_axis, object_id, plot_type):
    fig, ax = plt.subplots()
    ax.set_xlabel(title_map[plot_type])
    ax.set_xlim([0,1])
    ax.set_xticks(np.linspace(0,1,11))
    ax.set_ylabel('Average pose error (in degrees)')
    ax.set_title('Average pose error ' + object_name(object_id) + ' with varying ' + title_map[plot_type])
    ax.plot(x_axis, y_axis, '-o', markersize=5)
    ax.set_ylim(bottom=0)
    save_fig(fig, title_directory_map[plot_type] + '/' +'mean_error' + object_directory_name(object_id))
    plt.close()

def plot_mean_error_all_objects(x_axis, y_axis, plot_type):
    fig, ax = plt.subplots()
    ax.set_xlabel(title_map[plot_type])
    ax.set_xlim([0,1])
    ax.set_xticks(np.linspace(0,1,11))
    ax.set_ylabel('Average pose error (in degrees)')
    ax.set_title('Average pose error for all objects with varying ' + title_map[plot_type])
    ax.plot(x_axis, y_axis, '-o', markersize=3, linewidth=0.2)
    ax.set_ylim(bottom=0)
    ax.legend([str(object_id) for object_id in range(constants.NUM_OBJECTS)], prop={'size': 1.7})
    save_fig(fig, title_directory_map[plot_type]  + '/' + 'mean_error_all_objects')
    plt.close()

def plot_error_histogram(x_axis, y_axis, object_id, plot_type):
    fig, ax = plt.subplots()
    ax.set_xlabel(title_map[plot_type])
    ax.set_ylabel('Pose error (in degrees)')
    ax.set_title('Pose error ' + object_name(object_id) + ' with varying ' + title_map[plot_type])
    _, _, _, image = ax.hist2d(x_axis, y_axis, norm=LogNorm(), weights=x_axis)
    fig.colorbar(image)
    ax.set_ylim(bottom=0)
    save_fig(fig, title_directory_map[plot_type] + '/' +'error_histogram' + object_directory_name(object_id))
    plt.close()

def plot_accuracy_wireframe(x_axis, y_axis, z_axis, accuracy_type, object_id, plot_type):
    fig = plt.figure()
    ax = plt.axes(projection ='3d')
    ax.set_xlabel(title_map[0])
    ax.set_ylabel(title_map[1])
    ax.set_zlabel('Accuracy')
    ax.set_zlim([0,1])
    ax.set_xlim([1,0])
    ax.set_title(accuracy_map[accuracy_type] + ' recognition accuracy ' + object_name(object_id) + ' with varying ' + title_map[plot_type], fontsize='small')
    ax.plot_wireframe(x_axis, y_axis, z_axis)
    save_fig(fig, title_directory_map[plot_type] + '/' + accuracy_directory_map[accuracy_type] +'_accuracy' + object_directory_name(object_id))
    plt.close()

def plot_mean_error_wireframe(x_axis, y_axis, z_axis, object_id, plot_type):
    fig = plt.figure()
    ax = plt.axes(projection ='3d')
    ax.set_xlabel(title_map[0])
    ax.set_ylabel(title_map[1])
    ax.set_ylim([1,0])
    ax.set_zlabel('Average pose error (in degrees)')
    ax.set_title('Average pose error ' + object_name(object_id) + ' with varying ' + title_map[plot_type], fontsize='small')
    ax.plot_wireframe(x_axis, y_axis, z_axis)
    save_fig(fig, title_directory_map[plot_type] + '/' +'mean_error' + object_directory_name(object_id))
    plt.close()
