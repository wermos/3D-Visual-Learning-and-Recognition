import numpy as np

from constants import IMAGE_SIZE, NUM_OBJECTS, NUM_IMAGES, NUM_TESTING_IMAGES, NUM_TRAINING_IMAGES
from data_loader import coil_20_data_loader
from pca import PCA

if __name__ == "__main__":
    training, testing = coil_20_data_loader()
    training.sort()

    vectors_universal = np.zeros((IMAGE_SIZE, NUM_OBJECTS*NUM_TRAINING_IMAGES))
    vectors_object = np.zeros((NUM_OBJECTS, IMAGE_SIZE, NUM_TRAINING_IMAGES))
    object_angles = np.zeros((NUM_OBJECTS, NUM_TRAINING_IMAGES))
    for idx, (object_id, angle, image)  in enumerate(training):
        image = image / np.linalg.norm(image)
        image = np.squeeze(image)
        vectors_universal[:, idx] = image
        vectors_object[object_id, :, idx % NUM_TRAINING_IMAGES] = image
        object_angles[object_id][idx % NUM_TRAINING_IMAGES] = angle

    mean_universal = np.mean(vectors_universal, axis = 1)
    X_universal = vectors_universal - mean_universal.reshape((IMAGE_SIZE,1))

    mean_object = np.zeros((IMAGE_SIZE, NUM_OBJECTS))
    for idx, object_vectors in enumerate(vectors_object):
        mean_object[:, idx] = np.mean(object_vectors, axis = 1)
    X_object = vectors_object - (mean_object.T).reshape(NUM_OBJECTS,IMAGE_SIZE,1)

    print(X_universal.shape, X_object[0].shape)
    
    eigenvalues_universal, eigenvectors_universal, num_components_universal = PCA(X_universal)

    num_components_object = np.zeros(NUM_OBJECTS, dtype = int)
    eigenvalues_object = [None] * NUM_OBJECTS
    eigenvectors_object = [None] * NUM_OBJECTS
    for i in range(NUM_OBJECTS):
        eigenvalues_object[i], eigenvectors_object[i], num_components_object[i] = PCA(X_object[i])

    print(num_components_universal, eigenvectors_universal.shape, num_components_object[0], eigenvectors_object[0].shape)
