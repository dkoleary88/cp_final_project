import errno
import os
import sys
from glob import glob

import numpy as np
import cv2

from align_stack import align_and_crop
from blend_stack import median_blend

# Alignment transformation method { ecc , feature }
TRANS_METHOD = 'ecc'


# Read image files from specified directory within `source` directory
def read_files(dir):
    extensions = ['bmp', 'pbm', 'pgm', 'ppm', 'sr', 'ras', 'jpeg',
                  'jpg', 'jpe', 'jp2', 'tiff', 'tif', 'png', 'JPG']

    search_paths = [os.path.join(dir, '*.' + ext) for ext in extensions]
    image_files = sorted(reduce(list.__add__, map(glob, search_paths)))

    images = [cv2.imread(f, cv2.IMREAD_UNCHANGED | cv2.IMREAD_COLOR) for f in image_files]

    bad_read = any([img is None for img in images])
    if bad_read:
        raise RuntimeError(
            "Reading one or more files in {} failed - aborting"
            .format(dir))

    if len(images) < 5:
        raise RuntimeError('Not enough images in the directory - aborting')

    return images


# Write image file to specified directory within `output` directory
def write_file(dir, image):
    cv2.imwrite(os.path.join(dir, 'median_blend.jpg'), image)


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print 'Please specify directory'
        sys.exit()

    collection_name = str(sys.argv[1])

    src_dir = os.path.join('source', collection_name)
    out_dir = os.path.join('output', collection_name)

    try:
        os.makedirs(out_dir)
    except OSError as exception:
        if exception.errno != errno.EEXIST:
            raise

    print 'Reading images..'
    images = read_files(src_dir)

    print 'Aligning image stack'
    aligned = align_and_crop(images, TRANS_METHOD)

    print 'Creating artifact..'
    output_image = median_blend(aligned)

    print 'Writing artifact to file..'
    write_file(out_dir, output_image)

    print 'Finished.'
