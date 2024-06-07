import numpy as np
def parse_record(record, training):
    """ Parse a record to an image and perform data preprocessing.

    Args:
        record: An array of shape [3072,]. One row of the x_* matrix.
        training: A boolean. Determine whether it is in training mode.

    Returns:
        image: An array of shape [3, 32, 32].
    """
    depth_major = record.reshape((3, 32, 32))
    image = np.transpose(depth_major, [1, 2, 0]) #H x W x d
    image = preprocess_image(image, training)
    image = np.transpose(image, [2, 0, 1]) #d x H x W
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
        pad = 4
        image = np.pad(image, ((pad, pad), (pad, pad), (0, 0)), mode='constant')
        coordinates = np.random.randint(0 , 2*pad+1 , size=2)
        sub_image = image[coordinates[0]:coordinates[0]+32 , coordinates[1]:coordinates[1]+32 , :]
        image = np.fliplr(sub_image) if (np.random.rand() > 0.5) else sub_image
        
    mean = np.mean(image , axis=(0, 1))
    std = np.std(image , axis=(0, 1))
    image = (image - mean) / std
    return image