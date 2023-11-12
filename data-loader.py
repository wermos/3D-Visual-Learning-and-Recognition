import os
import cv2
import numpy as np

# We will designate approximately 60% of the dataset to training and the
# remaining 40% to testing.

def coil_20_processed_path_generator():
    # There's 20 objects, and each object has 72 pictures for it.

    # Since 0.6 * 72 = 43.2, we'll use 43 images for training and the remaining
    # 29 images for testing.

    # All in all, there will be 20 * 43 = 860 training images and 20 * 29 = 580
    # testing images.
    training = []
    testing = []

    dir_name = './data/coil-20-processed'

    for file_name in os.listdir(dir_name):
        if file_name.endswith(".png"):
            _, y = map(int, file_name.replace("obj", "").replace(".png", "").split("__"))
            if y <= 42:
                training.append(os.path.join(dir_name, file_name))
            else:
                testing.append(os.path.join(dir_name, file_name))

    return training, testing

def coil_20_unprocessed_path_generator():
    # There's 5 objects, and each object has 72 pictures for it.

    # Since 0.6 * 72 = 43.2, we'll use 43 images for training and the remaining
    # 29 images for testing.

    # All in all, there will be 5 * 43 = 215 training images, and 5 * 29 = 145
    # testing images.
    training = []
    testing = []

    dir_name = './data/coil-20-unprocessed'

    for file_name in os.listdir(dir_name):
        if file_name.endswith(".png"):
            _, y = map(int, file_name.replace("obj", "").replace(".png", "").split("__"))
            if y <= 42:
                training.append(os.path.join(dir_name, file_name))
            else:
                testing.append(os.path.join(dir_name, file_name))

    return training, testing

def coil_100_path_generator():
    # There's 100 objects, and each object has 72 pictures for it.

    # Since 0.6 * 72 = 43.2, we'll use 43 images for training and the remaining
    # 29 images for testing.

    # All in all, there will be 100 * 43 = 4300 training images, and 100 * 29 =
    # 2900 testing images.
    training = []
    testing = []

    dir_name = './data/coil-100'

    for file_name in os.listdir(dir_name):
        if file_name.endswith(".png"):
            _, y = map(int, file_name.replace("obj", "").replace(".png", "").split("__"))
            if y <= 210:
                training.append(os.path.join(dir_name, file_name))
            else:
                testing.append(os.path.join(dir_name, file_name))

    return training, testing

def image_loader(file_list):
    image_data = []
    for file in file_list:
        img = cv2.imread(file, cv2.IMREAD_GRAYSCALE)

        if img is not None:
            image_data.append(img)
        else:
            print(f"Failed to load image: `{file}`")

    return image_data

def coil_20_processed_data_loader():
    # COIL 20 Processed images are 128 x 128
    training_data, _ = coil_20_processed_path_generator()

    training_data = image_loader(training_data)

    training_matrix = np.ndarray(shape=(128 * 128, len(training_data)), dtype=np.float64)

    for idx, image in enumerate(training_data):
        training_matrix[:, idx] = np.reshape(image, (128 * 128, 1))

    return training_matrix

def coil_20_unprocessed_training_data_loader():
    # COIL 20 Unprocessed images are 448 x 416
    training_data, _ = coil_20_unprocessed_path_generator()

    training_data = image_loader(training_data)

    training_matrix = np.ndarray(shape=(448 * 416, len(training_data)), dtype=np.float64)

    for idx, image in enumerate(training_data):
        training_matrix[:, idx] = np.reshape(image, (448 * 416, 1))

    return training_matrix

def coil_100_training_data_loader():
    # COIL 100 images are 128 x 128
    training_data, _ = coil_100_path_generator()

    training_data = image_loader(training_data)

    training_matrix = np.ndarray(shape=(128 * 128, len(training_data)), dtype=np.float64)

    for idx, image in enumerate(training_data):
        training_matrix[:, idx] = np.reshape(image, (128 * 128, 1))

    return training_matrix

if __name__ == "__main__":
    training, testing = coil_20_processed_path_generator()
    print("Training:")
    print('\n'.join(map(str, training)))
    print("Testing:")
    print('\n'.join(map(str, testing)))
    # print(len(training))
    # print(len(testing))