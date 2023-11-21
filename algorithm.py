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
    _, eigenvectors_universal, mean_universal = PCA(vectors_universal)

    eigenvectors_object = [None] * NUM_OBJECTS
    mean_object = np.zeros((IMAGE_SIZE, NUM_OBJECTS))
    for object_id in range(NUM_OBJECTS):
        _, eigenvectors_object[object_id], mean_object[:, object_id] = PCA(vectors_object[object_id])

    # computing manifolds (parametric appearance representation)
    manifolds_universal = [compute_manifold_for_object(eigenvectors_universal, vectors_object[object_id], object_angles[object_id], mean_universal) for object_id in range(NUM_OBJECTS)]
    manifolds_object = [compute_manifold_for_object(eigenvectors_object[object_id], vectors_object[object_id], object_angles[object_id], mean_object[:, object_id]) for object_id in range(NUM_OBJECTS)]

    return mean_universal, mean_object, eigenvectors_universal, eigenvectors_object, manifolds_universal, manifolds_object

def compute_manifold_for_object(eigenvectors, object_vectors, object_angles, mean):
    num_components = (eigenvectors.shape)[1]
    eigencoefficients = np.zeros((num_components, NUM_TRAINING_IMAGES))
    for idx, image in enumerate(object_vectors.T):
        eigencoefficients[:, idx] = np.dot(image - mean, eigenvectors)
    return [CubicSpline(np.append(object_angles, 360+object_angles[0]), np.append(eigencoefficients[component_id], eigencoefficients[component_id][0]), bc_type = 'periodic') for component_id in range(num_components)]

def evaluate_cubic_splines_for_angles(manifolds_universal, manifolds_object, angle_values):
    manifold_points_universal = np.array([cubic_splines_to_vector(manifolds_universal[object_id], angle_values) for object_id in range(NUM_OBJECTS)])
    manifold_points_object = [np.array(cubic_splines_to_vector(manifolds_object[object_id], angle_values)) for object_id in range(NUM_OBJECTS)]
    return manifold_points_universal, manifold_points_object

def test_image(image, mean_universal, mean_object, eigenvectors_universal, eigenvectors_object, manifold_points_universal, manifold_points_object, angle_values):
    image = normalize(image)
    num_components_universal = (eigenvectors_universal.shape)[1]
    projection = np.dot(image-mean_universal, eigenvectors_universal)
    distances = np.array([cdist(manifold_points_universal[object_id].T, projection.reshape(1,num_components_universal), 'euclidean') for object_id in range(NUM_OBJECTS)])
    distances_minimum = np.min(distances, axis = 1)
    object_id = np.argmin(distances_minimum)
    distance = [np.min(distances[object_id])]

    num_components_object = (eigenvectors_object[object_id].shape)[1]
    projection = np.dot(image-mean_object[:, object_id], eigenvectors_object[object_id])
    distances = np.array(cdist(manifold_points_object[object_id].T, projection.reshape(1,num_components_object), 'euclidean'))
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
    manifold_points_universal, manifold_points_object = evaluate_cubic_splines_for_angles(manifolds_universal, manifolds_object, angle_values)
    num_tests = len(testing)
    accurate_count = np.zeros(2)
    error = 0
    # print("Actual vs Estimated")
    for object_id_true, angle_true , image in testing:
        object_id, angle, _ = test_image(image, mean_universal, mean_object, eigenvectors_universal, eigenvectors_object, manifold_points_universal, manifold_points_object, angle_values)
        # print("Object:", [object_id_true, object_id], "Angle:", [angle_true, angle])
        accurate_count[0] += (object_id_true == object_id)
        accurate_count[1] += (angle_true == angle)&(object_id_true == object_id)
        error += abs(angle_true - angle)
    print("Object Recognition: ", 100*accurate_count[0]/num_tests)
    print("Pose Estimation:", 100*accurate_count[1]/num_tests)
    print("Mean Pose error:", error/num_tests)