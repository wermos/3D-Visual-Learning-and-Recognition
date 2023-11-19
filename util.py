import cv2
import numpy as np
import os

from constants import IMAGE_SIZE

def construct_filename(parent_dir, obj_num, angle):
    # We construct the file name from `obj_num` and `angle`
    filename = "".join(["obj", str(obj_num), "__", str(angle), ".png"])
    return os.path.join(parent_dir, filename)

def load_image(dir_name, obj_num, angle):
    filename = construct_filename(dir_name, obj_num, angle)

    # `img` is a matrix of dimension (`IMAGE_HEIGHT`, `IMAGE_WIDTH`)
    img = cv2.imread(filename, cv2.IMREAD_GRAYSCALE)

    if img is not None:
        return np.reshape(img, (IMAGE_SIZE, 1))
    else:
        print(f"Failed to load image: `{filename}`")
