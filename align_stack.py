import cv2
import numpy as np
import sys

import ecc
import feature

align_methods = {
    'ecc': ecc,
    'feature': feature
}


# Warp an image into correct canvas size using homography matrix
def warp_image(ref_image, image, M_hom):
    image_aligned = cv2.warpPerspective(image, M_hom, (ref_image.shape[1], ref_image.shape[0]), flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP)

    return image_aligned


# Returns an image's corner coordinates
def get_image_corners(image):
    corners = np.zeros((4, 1, 2), dtype=np.float32)
    y, x = image.shape[0], image.shape[1]
    corners[1,0] = [0, y]
    corners[2,0] = [x, 0]
    corners[3,0] = [x, y]

    return corners


# Converts image corners into bounds
def convert_corners_to_bounds(corners):
    corners_reshaped = corners.reshape((2,2,2))

    left   = corners_reshaped[0, :, 0].max()
    right  = corners_reshaped[1, :, 0].min()
    top    = corners_reshaped[:, 0, 1].max()
    bottom = corners_reshaped[:, 1, 1].min()

    return np.array([left, right, top, bottom])


# Returns the new bounds which include pixels from all images
def get_bounds(current_bounds, image, M_hom):
    left, right, top, bottom = current_bounds

    corners = get_image_corners(image)
    corners_trans = cv2.perspectiveTransform(corners, M_hom)
    tleft, tright, ttop, tbottom = convert_corners_to_bounds(corners_trans)

    left   = np.ceil(max(left, tleft))
    right  = np.floor(min(right, tright))
    top    = np.ceil(max(top, ttop))
    bottom = np.floor(min(bottom, tbottom))

    return np.array([left, right, top, bottom], dtype=int)


# Crops entire image stack based on bounds
def crop_stack(images, bounds):
    left, right, top, bottom = bounds
    return [image[top:bottom, left:right] for image in images]


# Aligns image stack based on ECC metrics
def align_and_crop(images, align_method):
    ref_image = images[0]
    tail_images = images[1:]

    ref_corners = get_image_corners(ref_image)
    bounds = convert_corners_to_bounds(ref_corners)
    aligned_stack = []

    for i, image in enumerate(tail_images):
        M_hom = align_methods[align_method].find_M_hom(ref_image, image)
        bounds = get_bounds(bounds, image, M_hom)
        image_aligned = warp_image(ref_image, image, M_hom)
        aligned_stack.append(image_aligned)

    cropped_stack = crop_stack(aligned_stack, bounds)
    return cropped_stack
