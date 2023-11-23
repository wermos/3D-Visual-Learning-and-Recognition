import itertools
from os import environ
environ['OMP_NUM_THREADS'] = '16'
import numpy as np
import random

import constants
from util import load_grayscale_image, load_color_image

# Define the data type for the tuple elements
dtype = np.dtype([('object number', np.ubyte), ('angle', np.ushort), ('image data', (np.double, (constants.IMAGE_SIZE, 1)))])

def classifier(consts):
    # Decides which angles go into the training set and the testing set, for
    # each object.

    # We store the list of testing angles for each object, since that's fewer
    # things to store in memory
    
    random.seed(constants.RANDOM_SEED)
    
    testing_list = [random.sample(range(consts.NUM_IMAGES), k=consts.NUM_TESTING_IMAGES)
                    for _ in range(consts.NUM_OBJECTS)]

    return testing_list

def coil_20_data_loader(consts):
    dir_name = "./data/coil-20"

    training = np.ndarray(20 * consts.NUM_TRAINING_IMAGES, dtype=dtype)
    testing = np.ndarray(20 * consts.NUM_TESTING_IMAGES, dtype=dtype)

    testing_list = classifier(consts)

    training_idx = 0
    testing_idx = 0

    for obj_num_idx, angle_idx in itertools.product(range(20), range(consts.NUM_IMAGES)):
        # There's 72 images, and each image's rotation is 5 times the index value.
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

def coil_100_data_loader(consts):
    dir_name = "./data/coil-100"

    training = np.ndarray(100 * consts.NUM_TRAINING_IMAGES, dtype=dtype)
    testing = np.ndarray(100 * consts.NUM_TESTING_IMAGES, dtype=dtype)


    testing_list = classifier(consts)

    training_idx = 0
    testing_idx = 0

    for obj_num_idx, angle_idx in itertools.product(range(100), range(consts.NUM_IMAGES)):
        # There's 72 images, and each image's rotation is 5 times the index value.
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

def data_loader(consts):
    if consts.NUM_OBJECTS == 20:
        return coil_20_data_loader(consts)
    elif consts.NUM_OBJECTS== 100:
        return coil_100_data_loader(consts)
    else:
        raise NotImplementedError(f"There is no dataset that we support that contains {consts.NUM_OBJECTS} images.")

if __name__ == "__main__":
    # training, testing = data_loader(constants.COIL20_CONSTS)
    training, testing = data_loader(constants.Constants(20, 72, 128, 128, 0.1, 0.1))
    # training, testing = data_loader(constants.COIL100_CONSTS)
