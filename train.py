import numpy as np
from scipy.interpolate import CubicSpline

import constants
from pca import PCA
from util import normalize

def compute_manifold_for_object(eigenvectors, object_vectors, object_angles, mean):
    num_components = (eigenvectors.shape)[1]
    eigencoefficients = np.zeros((num_components, constants.NUM_TRAINING_IMAGES))
    for idx, image in enumerate(object_vectors.T):
        eigencoefficients[:, idx] = np.dot(image - mean, eigenvectors)
    return [CubicSpline(np.append(object_angles, 360+object_angles[0]), np.append(eigencoefficients[component_id], eigencoefficients[component_id][0]), bc_type = 'periodic') for component_id in range(num_components)]

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