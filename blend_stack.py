import numpy as np


# Blends image stack into a single artifact using median blend
def median_blend(images):
    images = np.array(images);
    image = np.median(images, axis=0)
    return image
