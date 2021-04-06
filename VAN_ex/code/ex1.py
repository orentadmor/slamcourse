import os
from typing import Tuple
from typing import Optional

import cv2
import numpy as np
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

DATA_PATH = '../data/dataset05/sequences/05/'
FEATURES = dict(orb=cv2.ORB_create, akaze=cv2.AKAZE_create, sift=cv2.SIFT_create)
DISTANCES = dict(orb=cv2.NORM_HAMMING, akaze=cv2.NORM_HAMMING, sift=cv2.NORM_L2)


def read_pair(image_index: int) -> Tuple[np.ndarray, np.ndarray]:
    left_image = cv2.imread(os.path.join(DATA_PATH, f'image_0/{image_index:06d}.png'), cv2.IMREAD_GRAYSCALE)
    right_image = cv2.imread(os.path.join(DATA_PATH, f'image_1/{image_index:06d}.png'), cv2.IMREAD_GRAYSCALE)
    return left_image, right_image


def read_cameras():
    with open(DATA_PATH + 'calib.txt') as f:
        l1 = f.readline().split()[1:]  # skip first token
        l2 = f.readline().split()[1:]  # skip first token
    l1 = [float(i) for i in l1]
    m1 = np.array(l1).reshape(3, 4)
    l2 = [float(i) for i in l2]
    m2 = np.array(l2).reshape(3, 4)
    k = m1[:, :3]
    m1 = np.linalg.inv(k) @ m1
    m2 = np.linalg.inv(k) @ m2
    return k, m1, m2


def detect_and_compute(image: np.ndarray,
                       detector: cv2.Feature2D,
                       ) -> Tuple[np.ndarray, np.ndarray]:
    keypoints_cv, descriptors = detector.detectAndCompute(image, None)
    keypoints = np.vstack([kp.pt for kp in keypoints_cv])
    return keypoints, descriptors


def pair_imshow(left_image: np.ndarray,
                right_image: np.ndarray,
                left_keypoints: Optional[np.ndarray] = None,
                right_keypoints: Optional[np.ndarray] = None,
                match_indices: Optional[np.ndarray] = None,
                show_lines: bool = True):
    stacked_image = np.vstack([left_image, right_image])
    plt.imshow(stacked_image, cmap='gray')
    if left_keypoints is not None:
        left_colors = np.tile([0, 1, 1], (len(left_keypoints), 1)).astype('float')  # cyan
        if match_indices is not None:
            left_colors[match_indices[:, 0]] = (1, 0.3, 0)  # orange
        plt.scatter(left_keypoints[:, 0], left_keypoints[:, 1], s=10, c=left_colors)
    if right_keypoints is not None:
        right_colors = np.tile([0, 1, 1], (len(right_keypoints), 1)).astype('float')  # cyan
        if match_indices is not None:
            right_colors[match_indices[:, 1]] = (1, 0.3, 0)  # orange
        plt.scatter(right_keypoints[:, 0], right_keypoints[:, 1] + right_image.shape[0], s=10, c=right_colors)
    if match_indices is not None and show_lines:
        leftx, lefty = left_keypoints[match_indices[:, 0]].T
        rightx, righty = right_keypoints[match_indices[:, 1]].T
        plt.plot(np.vstack([leftx, rightx]), np.vstack([lefty, righty + left_image.shape[0]]))
    plt.show()


def compute_distmat(left_descriptors: np.ndarray, right_descriptors: np.ndarray) -> np.ndarray:
    diffmat = left_descriptors[..., np.newaxis] - right_descriptors.T[np.newaxis]
    distmat = np.sqrt(np.sum(np.square(diffmat), axis=1))
    return distmat


def find_closest(distmat: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    match_indices = np.vstack([np.arange(len(distmat)), distmat.argmin(axis=1)]).T
    match_scores = distmat.min(axis=1)
    return match_indices, match_scores


def compute_significance(distmat: np.ndarray) -> np.ndarray:
    match_distance = distmat.min(axis=1)
    significance = match_distance / np.partition(distmat, 1, axis=1)[:, 1]
    return significance


def main(feature_type: str = 'orb'):
    # Read images
    left_image, right_image = read_pair(0)
    # Extract features
    detector = FEATURES[feature_type]()
    left_keypoints, left_descriptors = detect_and_compute(left_image, detector)
    right_keypoints, right_descriptors = detect_and_compute(right_image, detector)
    # 1.1
    pair_imshow(left_image,
                right_image,
                left_keypoints=left_keypoints,
                right_keypoints=right_keypoints)
    # 1.2
    print('first descriptors')
    print(left_descriptors[0])
    print(right_descriptors[0])
    # 1.3
    distmat = compute_distmat(left_descriptors, right_descriptors)
    match_indices, match_scores = find_closest(distmat)
    pair_imshow(left_image,
                right_image,
                left_keypoints=left_keypoints,
                right_keypoints=right_keypoints,
                match_indices=match_indices)
    # 1.4
    match_significance = compute_significance(distmat)
    match_filter = match_significance <= 0.8
    pair_imshow(left_image,
                right_image,
                left_keypoints=left_keypoints,
                right_keypoints=right_keypoints,
                match_indices=match_indices[match_filter])
    # 1.5
    # Looking at the camera matrices we see the cameras are stereo-normal,
    # meaning that they are both facing the same direction and are shifted only on the x-axis.
    # So matched keypoints should have approximately the same y-coordinates.
    ydiff = np.abs((left_keypoints[match_indices[:, 0], 1] - right_keypoints[match_indices[:, 1], 1]))
    match_filter = np.logical_and(ydiff <= 5, match_significance <= 0.8)
    pair_imshow(left_image,
                right_image,
                left_keypoints=left_keypoints,
                right_keypoints=right_keypoints,
                match_indices=match_indices[match_filter])
    pair_imshow(left_image,
                right_image,
                left_keypoints=left_keypoints,
                right_keypoints=right_keypoints,
                match_indices=match_indices[match_filter],
                show_lines=False)
    # 1.6
    # Using the fact that x=PX, we conclude that cross(x, PX) = 0. This trick ges rid of any scaling factor.
    # Computing the cross product we get two equations for each point (+ a redundant equation):
    # [y*P[2] - P[1] ; P[0] - x*P[2]] @ X = 0
    # with two 2d points we get an overdetermined system, and solve it using SVD.
    # http://www.cs.cmu.edu/~16385/s17/Slides/11.4_Triangulation.pdf
    k, m1, m2 = read_cameras()
    P0 = k @ m1
    P1 = k @ m2
    filtered_matched_indices = match_indices[match_filter]
    X = np.zeros((len(filtered_matched_indices), 3))
    err = np.zeros((len(filtered_matched_indices), 2))
    for idx, (lidx, ridx) in enumerate(filtered_matched_indices):
        x0, y0 = left_keypoints[lidx]
        x1, y1 = right_keypoints[ridx]
        A = np.vstack([y0*P0[2] - P0[1],
                       P0[0] - x0*P0[2],
                       y1*P1[2] - P1[1],
                       P1[0] - x1*P1[2]])
        U, sigma, VT = np.linalg.svd(A)
        X[idx] = VT[-1, :3] / VT[-1, -1]
        # Reprojection error
        xh0 = P0 @ np.hstack([X[idx], 1])
        xh0 = xh0[:2] / xh0[2]
        xh1 = P1 @ np.hstack([X[idx], 1])
        xh1 = xh1[:2] / xh1[2]
        err[idx] = np.sqrt([np.square(xh0 - [x0, y0]).sum(), np.square(xh1 - [x1, y1]).sum()])
    # plot 3d keypoints
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.scatter(X[:, 0], X[:, 1], X[:, 2])
    ax.view_init(azim=-90, elev=-70)
    plt.show()
    # Compare to using opencv
    lkp = left_keypoints[filtered_matched_indices[:, 0]]
    rkp = right_keypoints[filtered_matched_indices[:, 1]]
    Xcv = cv2.triangulatePoints(P0[:3], P1[:3], lkp.T, rkp.T).T
    Xcv = Xcv[:, :3] / Xcv[:, 3:]
    print(f'max error with opencv {np.max(np.abs(X - Xcv))}')

    # 1.7
    # We can further filter points by looking at the parallax of matching points between the two images (larget=better)
