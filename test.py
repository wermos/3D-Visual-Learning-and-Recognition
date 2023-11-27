from os import environ
environ['OMP_NUM_THREADS'] = '16'

import numpy as np
from scipy.spatial.distance import cdist
import argparse
import pickle

import constants
from data_loader import data_loader
from train import train_model, evaluate_cubic_splines_for_angles
from util import normalize, load_grayscale_image, load_color_image
from plots_util import logs_directory, images_directory

def test_image(image, mean_universal, mean_object, eigenvectors_universal, eigenvectors_object, manifold_points_universal, manifold_points_object, angle_values):
    image = normalize(image)
    num_components_universal = (eigenvectors_universal.shape)[1]
    projection = np.dot(image-mean_universal, eigenvectors_universal)
    distances = np.array([cdist(manifold_points_universal[object_id].T, projection.reshape(1,num_components_universal), 'euclidean') for object_id in range(constants.NUM_OBJECTS)])
    distances_minimum = np.min(distances, axis = 1)
    object_id = np.argmin(distances_minimum)
    distance = [np.min(distances[object_id])]

    num_components_object = (eigenvectors_object[object_id].shape)[1]
    projection = np.dot(image-mean_object[:, object_id], eigenvectors_object[object_id])
    distances = np.array(cdist(manifold_points_object[object_id].T, projection.reshape(1,num_components_object), 'euclidean'))
    angle_id = np.argmin(distances)
    distance.append(distances[angle_id][0])
    return object_id, angle_values[angle_id], distance

def process(DEBUGGING = False, precision = 5):
    print("data loading initiated...") if DEBUGGING else None
    training, testing = data_loader()
    print("data loading completed\ntraining initiated...") if DEBUGGING else None
    mean_universal, mean_object, eigenvectors_universal, eigenvectors_object, manifolds_universal, manifolds_object = train_model(training)
    print("training completed\ntesting initiated...") if DEBUGGING else None
    angle_values = np.arange(0,360,precision)
    manifold_points_universal, manifold_points_object = evaluate_cubic_splines_for_angles(manifolds_universal, manifolds_object, angle_values)
    accurate_count = np.zeros((constants.NUM_OBJECTS, 2))
    error = np.zeros((constants.NUM_OBJECTS, constants.NUM_TESTING_IMAGES))
    distances = np.zeros((constants.NUM_OBJECTS, constants.NUM_TESTING_IMAGES, 2))
    # print("Actual vs Estimated") if DEBUGGING else None
    for idx, (object_id_true, angle_true, image) in enumerate(testing):
        object_id, angle, distances[object_id_true][idx % constants.NUM_TESTING_IMAGES] = test_image(image, mean_universal, mean_object, eigenvectors_universal, eigenvectors_object, manifold_points_universal, manifold_points_object, angle_values)
        # print("Object:", [object_id_true, object_id], "Angle:", [angle_true, angle]) if DEBUGGING else None
        accurate_count[object_id_true][0] += (object_id_true == object_id)
        accurate_count[object_id_true][1] += (angle_true == angle)#*(object_id_true == object_id)
        error[object_id_true][idx % constants.NUM_TESTING_IMAGES] = min(abs(angle_true - angle), 360-abs(angle_true - angle))#*(object_id_true == object_id)
    print("testing completed...generating stats\n") if DEBUGGING else None
    return accurate_count[:,0], accurate_count[:,1], error, distances

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--test-single-image", action = 'store_true')
    parser.add_argument("--object-id")
    parser.add_argument("--angle")
    arguments = parser.parse_args()
    test_single_image = bool(arguments.test_single_image)
    object_id = int(arguments.object_id)
    angle = int(arguments.angle)
    if test_single_image:
        if constants.NUM_OBJECTS == 20:
            image = load_grayscale_image(images_directory, object_id, angle//5)
        else:
            image = load_color_image(images_directory, object_id, angle)
        filename = logs_directory + '/' + 'training_data' + '.pkl'
        with open(filename, 'rb') as file:
            print(test_image(image, *pickle.load(file)))

    else:
        print("Number of objects", constants.NUM_OBJECTS)
        accuracy_object, accuracy_pose, mean_error, _ = process(True)
        # print(accuracy_object, accuracy_pose, mean_error)
        print("Object Recognition accuracy: ", format(np.sum(accuracy_object)/(constants.NUM_OBJECTS*constants.NUM_TESTING_IMAGES), ".3%"))
        print("Pose Estimation accuracy:", format(np.sum(accuracy_pose)/(constants.NUM_OBJECTS*constants.NUM_TESTING_IMAGES), ".3%"))
        print("Mean Pose error:", format(np.sum(mean_error)/(constants.NUM_OBJECTS*constants.NUM_TESTING_IMAGES), ".3f") + "\u00b0")