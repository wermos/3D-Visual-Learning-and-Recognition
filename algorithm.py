import numpy as np

from constants import IMAGE_SIZE, NUM_OBJECTS, NUM_IMAGES, NUM_TESTING_IMAGES, NUM_TRAINING_IMAGES
from data_loader import coil_20_data_loader

if __name__ == "__main__":
    training, testing = coil_20_data_loader()

    vectors_universal = [[] for i in range(NUM_OBJECTS*NUM_TRAINING_IMAGES)]
    vectors_object = [[] for i in range(NUM_OBJECTS)]
    object_angles = [[] for i in range(NUM_OBJECTS)]
    for object_id, angle, image  in training:
        vectors_object[object_id].append(image)
        object_angles[object_id].append(angle)

    for i in range(NUM_OBJECTS):
        vectors_universal[NUM_TRAINING_IMAGES*i:NUM_TRAINING_IMAGES*i+NUM_TRAINING_IMAGES] = vectors_object[i]

    vectors_universal = np.squeeze(np.array(vectors_universal))
    vectors_object = np.squeeze(np.array(vectors_object))
    object_angles = np.squeeze(np.array(object_angles))

    mean_universal = np.mean(vectors_universal, axis = 0)
    mean_object = np.empty(NUM_OBJECTS, dtype = object)
    for i, vectors in enumerate(vectors_object):
        mean_object[i] = np.mean(vectors, axis = 0)

    X_universal = vectors_universal - mean_universal
    X_object = np.empty(NUM_OBJECTS, dtype = object)
    for i in range(NUM_OBJECTS):
        X_object[i] = vectors_object[i] - mean_object[i]

    X_universal = np.matrix(X_universal)
    for i in range(20):
        X_object[i] = np.matrix(X_object[i])

    print(X_universal.shape, X_object[0].shape)