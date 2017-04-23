import cv2
import numpy as np


# Finds feature matches between two images
def find_matches (image_1, image_2, num_matches):
    orb = cv2.ORB_create()

    image_1_kp, image_1_desc = orb.detectAndCompute(image_1, None)
    image_2_kp, image_2_desc = orb.detectAndCompute(image_2, None)

    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches_unsorted = bf.match(image_1_desc, image_2_desc)

    matches = sorted(matches_unsorted, key=lambda x: x.distance)[:num_matches]

    return image_1_kp, image_2_kp, matches


# Finds homography matrix between matching keypoints
def find_M_hom(image_1, image_2):
    NUM_MATCHES = 1000

    kp_1, kp_2, matches = find_matches(image_1, image_2, NUM_MATCHES)

    image_1_points = np.zeros((len(matches), 1, 2), dtype=np.float32)
    image_2_points = np.zeros((len(matches), 1, 2), dtype=np.float32)

    for i, match in enumerate(matches):
        image_1_points[i, 0] = kp_1[match.queryIdx].pt
        image_2_points[i, 0] = kp_2[match.trainIdx].pt

    M_hom, _ = cv2.findHomography(image_1_points, image_2_points, method=cv2.RANSAC, ransacReprojThreshold=5.0)
    return M_hom


