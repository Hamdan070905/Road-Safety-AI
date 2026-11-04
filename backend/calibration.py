"""Ground-plane calibration helpers. Coordinates are metres in a user-defined plane."""
import cv2
import numpy as np


def estimate_homography(image_points, ground_points):
    image = np.asarray(image_points, dtype=float)
    ground = np.asarray(ground_points, dtype=float)
    if image.ndim != 2 or ground.ndim != 2 or image.shape != ground.shape or image.shape[1:] != (2,):
        raise ValueError('Provide matching image and ground point pairs')
    if len(image) < 4:
        raise ValueError('At least four point pairs are required')
    if not np.isfinite(image).all() or not np.isfinite(ground).all():
        raise ValueError('Coordinates must be finite')
    if np.linalg.matrix_rank(np.column_stack([image, np.ones(len(image))])) < 3 or np.linalg.matrix_rank(np.column_stack([ground, np.ones(len(ground))])) < 3:
        raise ValueError('Point arrangement is collinear or degenerate')
    h, mask = cv2.findHomography(image.astype(np.float64), ground.astype(np.float64), cv2.RANSAC, 0.05)
    if h is None or mask is None or int(mask.sum()) < 4 or abs(np.linalg.det(h)) < 1e-12:
        raise ValueError('Could not estimate a stable homography; spread points across the road plane')
    predicted = cv2.perspectiveTransform(image.astype(np.float64)[None], h)[0]
    error = np.linalg.norm(predicted-ground, axis=1)
    return {'homography': h.tolist(), 'inliers': mask.reshape(-1).astype(bool).tolist(), 'rms_error_m': float(np.sqrt(np.mean(error[mask.reshape(-1).astype(bool)]**2)))}


def transform_point(homography, point):
    p = np.asarray(point, dtype=float).reshape(1, 1, 2)
    out = cv2.perspectiveTransform(p, np.asarray(homography, dtype=float))[0, 0]
    return [float(out[0]), float(out[1])]


def inside_polygon(point, polygon):
    return cv2.pointPolygonTest(np.asarray(polygon, dtype=np.float32), tuple(map(float, point)), False) >= 0
