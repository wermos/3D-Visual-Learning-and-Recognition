from math import floor

RANDOM_SEED = 7 # Set this to None for truly (pseudo)random behavior.

# The number of objects we have
NUM_OBJECTS = 20
# The number of images we have of each object
NUM_IMAGES = 72

# Number of pixels in one image
IMAGE_HEIGHT = 128
IMAGE_WIDTH = 128
IMAGE_SIZE = IMAGE_HEIGHT * IMAGE_WIDTH

TRAINING_PERCENTAGE = 0.6

# This value is set according to the value of `TRAINING_PERCENTAGE` in case it is needed.
TESTING_PERCENTAGE = 1 - TRAINING_PERCENTAGE

# We will designate approximately `TRAINING_PERCENTAGE` of the dataset to training and the
# remaining `TESTING_PERCENTAGE` to testing.

# Since there are `NUM_IMAGES` images of each object, and we use `floor(72 * TRAINING_PERCENTAGE)` images
# for training, and the remaining images for testing.
#
# The training and testing images will be randomly chosen to provide more
# variance in the angle and orientation of the testing images.

NUM_TRAINING_IMAGES = floor(NUM_IMAGES * TRAINING_PERCENTAGE)

# Once again, this value is set according to the value of `NUM_TRAINING_IMAGES` in case it is needed.
NUM_TESTING_IMAGES = NUM_IMAGES - NUM_TRAINING_IMAGES

PCA_THRESHOLD = 0.8

DEBUGGING = True

class Constants:
    def __init__(self, NUM_OBJECTS, NUM_IMAGES, IMAGE_HEIGHT, IMAGE_WIDTH,
                 TRAINING_PERCENTAGE, PCA_THRESHOLD):
        # The number of objects we have
        self.NUM_OBJECTS = NUM_OBJECTS
        # The number of images we have of each object
        self.NUM_IMAGES = NUM_IMAGES

        # Number of pixels in one image
        self.IMAGE_HEIGHT = IMAGE_HEIGHT
        self.IMAGE_WIDTH = IMAGE_WIDTH

        self.TRAINING_PERCENTAGE = TRAINING_PERCENTAGE
        
        self.PCA_THRESHOLD = PCA_THRESHOLD

        # Everything below this is calculated from the above values
        self.IMAGE_SIZE = IMAGE_HEIGHT * IMAGE_WIDTH
        
        # This value is set according to the value of `TRAINING_PERCENTAGE` in case it is needed.
        self.TESTING_PERCENTAGE = 1 - TRAINING_PERCENTAGE

        # We will designate approximately `TRAINING_PERCENTAGE` of the dataset to training and the
        # remaining `TESTING_PERCENTAGE` to testing.

        # Since there are `NUM_IMAGES` images of each object, and we use `floor(NUM_IMAGES * TRAINING_PERCENTAGE)` images
        # for training, and the remaining images for testing.
        #
        # The training and testing images will be randomly chosen to provide more
        # variance in the angle and orientation of the testing images.

        self.NUM_TRAINING_IMAGES = floor(NUM_IMAGES * TRAINING_PERCENTAGE)

        # Once again, this value is set according to the value of `NUM_TRAINING_IMAGES` in case it is needed.
        self.NUM_TESTING_IMAGES = NUM_IMAGES - NUM_TRAINING_IMAGES

COIL20_CONSTS = Constants(20, 72, 128, 128, 0.6, 0.8)
COIL100_CONSTS = Constants(100, 72, 128, 128, 0.6, 0.8)