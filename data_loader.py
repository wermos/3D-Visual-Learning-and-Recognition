import itertools
import numpy as np
import random

import constants
from util import load_grayscale_image, load_color_image

# Define the data type for the tuple elements
dtype = np.dtype([('object number', np.ubyte), ('angle', np.ushort), ('image data', (np.double, (constants.IMAGE_SIZE, 1)))])

def classifier(num_objects):
    # Decides which angles go into the training set and the testing set, for
    # each object.

    # We store the list of testing angles for each object, since that's fewer
    # things to store in memory
    testing_list = []
    
    random.seed(constants.RANDOM_SEED)
    
    for _ in range(num_objects):
        testing_list.append(random.sample(range(constants.NUM_IMAGES), k=constants.NUM_TESTING_IMAGES))

    return testing_list

def coil_20_data_loader():
    dir_name = "./data/coil-20"

    training = np.ndarray(20 * constants.NUM_TRAINING_IMAGES, dtype=dtype)
    testing = np.ndarray(20 * constants.NUM_TESTING_IMAGES, dtype=dtype)

    testing_list = classifier(20)

    training_idx = 0
    testing_idx = 0

    for obj_num_idx, angle_idx in itertools.product(range(20), range(constants.NUM_IMAGES)):
        # There's 71 images, and each image's rotation is 5 times the index value.
        # For example, image number 5 has a rotation of 5 * 5 = 25 degrees.
        #
        # The `angle_idx` is a proxy for the actual angle of rotation of the image.
        # Similarly, `obj_num_idx` is a proxy for the actual object number.
        obj_num = obj_num_idx + 1

        image = load_grayscale_image(dir_name, obj_num, angle_idx)
        datum = (obj_num_idx, angle_idx * 5, image)

        if angle_idx in testing_list[obj_num_idx]:
            testing[testing_idx] = datum
            testing_idx += 1
        else:
            training[training_idx] = datum
            training_idx += 1

    return training, testing

def coil_100_data_loader():
    dir_name = "./data/coil-100"

    training = np.ndarray(100 * constants.NUM_TRAINING_IMAGES, dtype=dtype)
    testing = np.ndarray(100 * constants.NUM_TESTING_IMAGES, dtype=dtype)

    testing_list = classifier(100)

    training_idx = 0
    testing_idx = 0

    for obj_num_idx, angle_idx in itertools.product(range(100), range(constants.NUM_IMAGES)):
        # There's 71 images, and each image's rotation is 5 times the index value.
        # For example, image number 5 has a rotation of 5 * 5 = 25 degrees.
        #
        # The `angle_idx` is a proxy for the actual angle of rotation of the image.
        angle = angle_idx * 5
        # Similarly, `obj_num_idx` is a proxy for the actual object number.
        obj_num = obj_num_idx + 1

        image = load_color_image(dir_name, obj_num, angle)
        datum = (obj_num_idx, angle, image)

        if angle_idx in testing_list[obj_num_idx]:
            testing[testing_idx] = datum
            testing_idx += 1
        else:
            training[training_idx] = datum
            training_idx += 1

    return training, testing

if __name__ == "__main__":
    # training, testing = coil_20_data_loader()
    training, testing = coil_100_data_loader()
