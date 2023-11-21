import numpy as np
from scipy.interpolate import CubicSpline
from scipy.spatial.distance import cdist

from constants import IMAGE_SIZE, NUM_OBJECTS, NUM_TRAINING_IMAGES
from data_loader import coil_20_data_loader
from pca import PCA
from util import normalize, cubic_splines_to_vector

def train_model(training_data):
    # processing training data
    training_data.sort()
    vectors_universal = np.zeros((IMAGE_SIZE, NUM_OBJECTS*NUM_TRAINING_IMAGES))
    vectors_object = np.zeros((NUM_OBJECTS, IMAGE_SIZE, NUM_TRAINING_IMAGES))
    object_angles = np.zeros((NUM_OBJECTS, NUM_TRAINING_IMAGES))
    for idx, (object_id, angle, image)  in enumerate(training_data):
        image = normalize(image)
        vectors_universal[:, idx] = image
        vectors_object[object_id, :, idx % NUM_TRAINING_IMAGES] = image
        object_angles[object_id][idx % NUM_TRAINING_IMAGES] = angle

    # computing eigenspaces
    mean_universal = np.mean(vectors_universal, axis = 1)
    X_universal = vectors_universal - mean_universal.reshape((IMAGE_SIZE,1))

    mean_object = np.zeros((IMAGE_SIZE, NUM_OBJECTS))
    for object_id, object_vectors in enumerate(vectors_object):
        mean_object[:, object_id] = np.mean(object_vectors, axis = 1)
    X_object = vectors_object - (mean_object.T).reshape(NUM_OBJECTS,IMAGE_SIZE,1)

    # print("size of X:", X_universal.shape, X_object[0].shape)
    
    _, eigenvectors_universal, num_components_universal = PCA(X_universal)

    num_components_object = np.zeros(NUM_OBJECTS, dtype = int)
    eigenvectors_object = [None] * NUM_OBJECTS
    for object_id in range(NUM_OBJECTS):
        _, eigenvectors_object[object_id], num_components_object[object_id] = PCA(X_object[object_id])

    # print("size of eignevectors:", eigenvectors_universal.shape, eigenvectors_object[0].shape)
    # print("number of components:", num_components_universal, num_components_object)

    # computing manifolds (parametric appearance representation)
    eigencoefficients_universal = [None] * NUM_OBJECTS
    eigencoefficients_object = [None] * NUM_OBJECTS
    for object_id in range(NUM_OBJECTS):
        eigencoefficients_universal[object_id] = np.zeros((num_components_universal, NUM_TRAINING_IMAGES))
        eigencoefficients_object[object_id] = np.zeros((num_components_object[object_id], NUM_TRAINING_IMAGES))

    for object_id, object_vectors in enumerate(vectors_object):
        for angle_id, image in enumerate(object_vectors.T):
            eigencoefficients_universal[object_id][:, angle_id] = np.dot(image - mean_universal, eigenvectors_universal)
            eigencoefficients_object[object_id][:, angle_id] = np.dot(image - mean_object[:, object_id], eigenvectors_object[object_id])

    # print("size of eigencoefficients:", eigencoefficients_universal[0].shape, [eigencoefficients_object[idx].shape for idx in range(NUM_OBJECTS)])

    manifolds_universal = [None] * NUM_OBJECTS
    manifolds_object = [None] * NUM_OBJECTS
    for object_id in range(NUM_OBJECTS):
        manifolds_universal[object_id] = [CubicSpline(np.append(object_angles[object_id], 360+object_angles[object_id][0]), np.append(eigencoefficients_universal[object_id][component_id], eigencoefficients_universal[object_id][component_id][0]), bc_type = 'periodic') for component_id in range(num_components_universal)]
        manifolds_object[object_id] = [CubicSpline(np.append(object_angles[object_id], 360+object_angles[object_id][0]), np.append(eigencoefficients_object[object_id][component_id], eigencoefficients_object[object_id][component_id][0]), bc_type = 'periodic') for component_id in range(num_components_object[object_id])]

    # print("number of manifolds:", len(manifolds_universal), len(manifolds_object))
    return mean_universal, mean_object, eigenvectors_universal, eigenvectors_object, manifolds_universal, manifolds_object

def evaluate_cubic_splines_for_angles(manifolds_universal, manifolds_object, angle_values):
    points_universal = np.array([cubic_splines_to_vector(manifolds_universal[object_id], angle_values) for object_id in range(NUM_OBJECTS)])
    points_object = [np.array(cubic_splines_to_vector(manifolds_object[object_id], angle_values)) for object_id in range(NUM_OBJECTS)]
    return points_universal, points_object

def test_image(image, mean_universal, mean_object, eigenvectors_universal, eigenvectors_object, points_universal, points_object, angle_values):
    image = normalize(image)
    num_components_universal = (eigenvectors_universal.shape)[1]
    projection = np.dot(image-mean_universal, eigenvectors_universal)
    distances = np.array([cdist(points_universal[object_id].T, projection.reshape(1,num_components_universal), 'euclidean') for object_id in range(NUM_OBJECTS)])
    distances_minimum = np.min(distances, axis = 1)
    object_id = np.argmin(distances_minimum)
    distance = [np.min(distances[object_id])]

    num_components_object = (eigenvectors_object[object_id].shape)[1]
    projection = np.dot(image-mean_object[:, object_id], eigenvectors_object[object_id])
    distances = np.array(cdist(points_object[object_id].T, projection.reshape(1,num_components_object), 'euclidean'))
    angle_id = np.argmin(distances)
    distance.append(distances[angle_id][0])
    return object_id, angle_values[angle_id], distance

if __name__ == "__main__":
    training, testing = coil_20_data_loader()
    print("Training Phase")
    mean_universal, mean_object, eigenvectors_universal, eigenvectors_object, manifolds_universal, manifolds_object = train_model(training)

    print("Testing Phase")
    precision = 5
    angle_values = np.arange(0,360,precision)
    points_universal, points_object = evaluate_cubic_splines_for_angles(manifolds_universal, manifolds_object, angle_values)
    num_tests = len(testing)
    accurate_count = np.zeros(2)
    error = 0
    # print("Actual vs Estimated")
    for object_id_true, angle_true , image in testing:
        object_id, angle, _ = test_image(image, mean_universal, mean_object, eigenvectors_universal, eigenvectors_object, points_universal, points_object, angle_values)
        # print("Object:", [object_id_true, object_id], "Angle:", [angle_true, angle])
        accurate_count[0] += (object_id_true == object_id)
        accurate_count[1] += (angle_true == angle)&(object_id_true == object_id)
        error += abs(angle_true - angle)
    print("Object Recognition: ", 100*accurate_count[0]/num_tests)
    print("Pose Estimation:", 100*accurate_count[1]/num_tests)
    print("Mean Pose error:", error/num_tests)