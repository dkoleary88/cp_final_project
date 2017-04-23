import cv2
import numpy as np


# Finds homography matrix between two images for ECC transform
def find_M_hom(ref_image, image):
    # Convert images to grayscale
    ref_gray = cv2.cvtColor(ref_image,cv2.COLOR_BGR2GRAY)
    image_gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)

    # Define termination criteria
    criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 50,  1e-5)

    # Run the ECC algorithm. The results are stored in warp_matrix.
    M_hom = np.eye(3, 3, dtype=np.float32)
    _, M_hom = cv2.findTransformECC(ref_gray, image_gray, M_hom, cv2.MOTION_HOMOGRAPHY, criteria)

    return M_hom
