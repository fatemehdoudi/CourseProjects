import numpy as np

""" This script implements the functions for data augmentation and preprocessing.
"""

def parse_record(record, training):
    """ Parse a record to an image and perform data preprocessing.

    Args:
        record: An array of shape [3072,]. One row of the x_* matrix.
        training: A boolean. Determine whether it is in training mode.

    Returns:
        image: An array of shape [3, 32, 32].
    """
    # Reshape from [depth * height * width] to [depth, height, width].
    depth_major = record.reshape((3, 32, 32))

    # Convert from [depth, height, width] to [height, width, depth]
    image = np.transpose(depth_major, [1, 2, 0])

    image = preprocess_image(image, training)

    # Convert from [height, width, depth] to [depth, height, width]
    image = np.transpose(image, [2, 0, 1])

    return image

def preprocess_image(image, training):
    """ Preprocess a single image of shape [height, width, depth].

    Args:
        image: An array of shape [32, 32, 3].
        training: A boolean. Determine whether it is in training mode.
    
    Returns:
        image: An array of shape [32, 32, 3].
    """
    if training:
        ### YOUR CODE HERE
        # Resize the image to add four extra pixels on each side.
        ### YOUR CODE HERE
        pad = 4
        image = np.pad(image, ((pad, pad), (pad, pad), (0, 0)), mode='constant')
        ### YOUR CODE HERE
        # Randomly crop a [32, 32] section of the image.
        # HINT: randomly generate the upper left point of the image
        coordinates = np.random.randint(0 , 2*pad+1 , size=2)
        sub_image = image[coordinates[0]:coordinates[0]+32 , coordinates[1]:coordinates[1]+32 , :]
        ### YOUR CODE HERE

        ### YOUR CODE HERE
        image = np.fliplr(sub_image) if (np.random.rand() > 0.5) else sub_image
        ### YOUR CODE HERE

    ### YOUR CODE HERE
    mean = np.mean(image , axis=(0, 1))
    std = np.std(image , axis=(0, 1))
    image = (image - mean) / std
    ### YOUR CODE HERE

    return image