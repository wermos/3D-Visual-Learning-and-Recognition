import cv2
import numpy as np
import os

import constants

def construct_filename(parent_dir, obj_num, angle):
    # We construct the file name from `obj_num` and `angle`
    filename = "".join(["obj", str(obj_num), "__", str(angle), ".png"])
    return os.path.join(parent_dir, filename)

def load_grayscale_image(dir_name, obj_num, angle):
    filename = construct_filename(dir_name, obj_num, angle)

    # `img` is a matrix of dimension (`IMAGE_HEIGHT`, `IMAGE_WIDTH`)
    img = cv2.imread(filename, cv2.IMREAD_GRAYSCALE)

    if img is not None:
        return np.reshape(img, (constants.IMAGE_SIZE, 1))
    else:
        print(f"Failed to load image: `{filename}`")

def load_color_image(dir_name, obj_num, angle):
    filename = construct_filename(dir_name, obj_num, angle)

    # `img` is a matrix of dimension (`IMAGE_HEIGHT`, `IMAGE_WIDTH`)
    img = cv2.cvtColor(cv2.imread(filename), cv2.COLOR_BGR2GRAY)

    if img is not None:
        return np.reshape(img, (constants.IMAGE_SIZE, 1))
    else:
        print(f"Failed to load image: `{filename}`")

def cubic_splines_to_vector(manifolds, vector):
    return [manifold(vector) for manifold in manifolds]

def normalize(image):
    return np.squeeze(image / np.linalg.norm(image))

def reshape_to_square_matrix(array, length):
    return array.reshape(length, length)

def append_intial_element(array):
    return np.append(array, array[0])