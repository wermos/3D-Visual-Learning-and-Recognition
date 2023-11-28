from os import environ
environ['OMP_NUM_THREADS'] = '16'

import numpy as np
from scipy.interpolate import CubicSpline
import pickle
import argparse
import constants
from data_loader import data_loader
from pca import PCA
from util import normalize, cubic_splines_to_vector, append_intial_element
from plots_util import logs_directory, plots_directory, title_directory_map, plot_manifolds, plot_type_map
from pathlib import Path

def compute_manifold_for_object(eigenvectors, object_vectors, object_angles, mean):
    num_components = (eigenvectors.shape)[1]
    eigencoefficients = np.zeros((num_components, constants.NUM_TRAINING_IMAGES))
    for idx, image in enumerate(object_vectors.T):
        eigencoefficients[:, idx] = np.dot(image - mean, eigenvectors)
    return [CubicSpline(np.append(object_angles, 360+object_angles[0]), np.append(eigencoefficients[component_id], eigencoefficients[component_id][0]), bc_type = 'periodic') for component_id in range(num_components)]

def evaluate_cubic_splines_for_angles(manifolds_universal, manifolds_object, angle_values):
    manifold_points_universal = np.array([cubic_splines_to_vector(manifolds_universal[object_id], angle_values) for object_id in range(constants.NUM_OBJECTS)])
    manifold_points_object = [np.array(cubic_splines_to_vector(manifolds_object[object_id], angle_values)) for object_id in range(constants.NUM_OBJECTS)]
    return manifold_points_universal, manifold_points_object

def train_model(training_data):
    # processing training data
    training_data.sort()
    vectors_universal = np.zeros((constants.IMAGE_SIZE, constants.NUM_OBJECTS*constants.NUM_TRAINING_IMAGES))
    vectors_object = np.zeros((constants.NUM_OBJECTS, constants.IMAGE_SIZE, constants.NUM_TRAINING_IMAGES))
    object_angles = np.zeros((constants.NUM_OBJECTS, constants.NUM_TRAINING_IMAGES))
    for idx, (object_id, angle, image)  in enumerate(training_data):
        image = normalize(image)
        vectors_universal[:, idx] = image
        vectors_object[object_id, :, idx % constants.NUM_TRAINING_IMAGES] = image
        object_angles[object_id][idx % constants.NUM_TRAINING_IMAGES] = angle

    # computing eigenspaces
    _, eigenvectors_universal, mean_universal = PCA(vectors_universal)

    eigenvectors_object = [None] * constants.NUM_OBJECTS
    mean_object = np.zeros((constants.IMAGE_SIZE, constants.NUM_OBJECTS))
    for object_id in range(constants.NUM_OBJECTS):
        _, eigenvectors_object[object_id], mean_object[:, object_id] = PCA(vectors_object[object_id])

    # computing manifolds (parametric appearance representation)
    manifolds_universal = [compute_manifold_for_object(eigenvectors_universal, vectors_object[object_id], object_angles[object_id], mean_universal) for object_id in range(constants.NUM_OBJECTS)]
    manifolds_object = [compute_manifold_for_object(eigenvectors_object[object_id], vectors_object[object_id], object_angles[object_id], mean_object[:, object_id]) for object_id in range(constants.NUM_OBJECTS)]

    return mean_universal, mean_object, eigenvectors_universal, eigenvectors_object, manifolds_universal, manifolds_object

def generate_manifolds_plots(manifolds_universal, manifolds_object, angle_values):
    _, manifold_points_object = evaluate_cubic_splines_for_angles(manifolds_universal, manifolds_object, angle_values)
    Path(plots_directory + title_directory_map[-1] + plot_type_map[0]).mkdir(parents=True, exist_ok=True)
    Path(plots_directory + title_directory_map[-1] + plot_type_map[1]).mkdir(parents=True, exist_ok=True)
    for object_id, manifolds in enumerate(manifold_points_object):
        arguments = []
        x = append_intial_element(manifolds[0])
        y = append_intial_element(manifolds[1]) if len(manifolds) >= 2 else None
        z = append_intial_element(manifolds[2]) if len(manifolds) >= 3 else None
        for axes in [x,y,z]:
            if axes is not None:
                arguments.append(axes)
        plot_manifolds(object_id, 0, *arguments)
        plot_manifolds(object_id, 1, *arguments)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--restrict-plot-generation", action = 'store_true')
    arguments = parser.parse_args()
    restrict_plot_generation = bool(arguments.restrict_plot_generation)

    print("Number of objects", constants.NUM_OBJECTS)
    print("data loading initiated...")
    training, testing = data_loader()
    print("data loading completed\ntraining initiated...")
    angle_values = np.arange(0,360,5)
    mean_universal, mean_object, eigenvectors_universal, eigenvectors_object, manifolds_universal, manifolds_object = train_model(training)
    manifold_points_universal, manifold_points_object = evaluate_cubic_splines_for_angles(manifolds_universal, manifolds_object, angle_values)
    print("training completed\nstoring data...")
    filename = logs_directory + '/' + 'training_data' + '.pkl'
    with open(filename, 'wb') as file:
        pickle.dump([mean_universal, mean_object, eigenvectors_universal, eigenvectors_object, manifold_points_universal, manifold_points_object, angle_values], file)
    print("data stored")

    if not restrict_plot_generation:
        print("generating parametric representation plots...")
        generate_manifolds_plots(manifolds_universal, manifolds_object, angle_values)