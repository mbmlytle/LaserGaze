# -----------------------------------------------------------------------------------
# Company: TensorSense
# Project: LaserGaze
# File: AffineTransformer.py
# Description: This class is designed to calculate and apply affine transformations
#              between two sets of 3D points, typically used for mapping points from
#              one coordinate system to another. It supports scaling based on defined
#              horizontal and vertical points, estimating an affine transformation matrix,
#              and converting points between the two coordinate systems using the matrix.
#              It is especially useful in projects involving facial recognition,
#              augmented reality, or any application where precise spatial transformations
#              are needed between different 3D models.
# Author: Sergey Kuldin
# -----------------------------------------------------------------------------------

import numpy as np
import cv2
from dataclasses import dataclass

import screeninfo

class AffineTransformer:
    """
    A class to calculate and manage affine transformations between two 3D point sets.
    This class allows for precise spatial alignment of 3D models based on calculated
    scale factors and transformation matrices, facilitating tasks such as facial
    landmark transformations or other applications requiring model alignment.
    """

    def __init__(self, m1_points, m2_points, m1_hor_points, m1_ver_points, m2_hor_points, m2_ver_points):
        """
        Initializes the transformer by calculating the scale factor and the affine transformation matrix.

        Args:
        - m1_points (np.array): Numpy array of the first set of 3D points.
        - m2_points (np.array): Numpy array of the second set of 3D points to which the first set is aligned.
        - m1_hor_points (np.array): Horizontal reference points from the first model used to calculate scaling.
        - m1_ver_points (np.array): Vertical reference points from the first model used to calculate scaling.
        - m2_hor_points (np.array): Horizontal reference points from the second model used to calculate scaling.
        - m2_ver_points (np.array): Vertical reference points from the second model used to calculate scaling.
        """
        self.scale_factor = self._get_scale_factor(
            np.array(m1_hor_points),
            np.array(m1_ver_points),
            np.array(m2_hor_points),
            np.array(m2_ver_points)
        )

        scaled_m2_points = m2_points * self.scale_factor

        retval, M, inliers = cv2.estimateAffine3D(m1_points, scaled_m2_points)
        if retval:
            self.success = True
            self.transform_matrix = M
        else:
            self.success = False
            self.transform_matrix = None

    def _get_scale_factor(self, m1_hor_points, m1_ver_points, m2_hor_points, m2_ver_points):
        """
        Calculates the scale factor between two sets of reference points (horizontal and vertical).

        Args:
        - m1_hor_points (np.array): Horizontal reference points from the first model.
        - m1_ver_points (np.array): Vertical reference points from the first model.
        - m2_hor_points (np.array): Horizontal reference points from the second model.
        - m2_ver_points (np.array): Vertical reference points from the second model.

        Returns:
        - float: The calculated uniform scale factor to apply.
        """
        m1_width = np.linalg.norm(m1_hor_points[0] - m1_hor_points[1])
        m1_height = np.linalg.norm(m1_ver_points[0] - m1_ver_points[1])
        m2_width = np.linalg.norm(m2_hor_points[0] - m2_hor_points[1])
        m2_height = np.linalg.norm(m2_ver_points[0] - m2_ver_points[1])
        scale_width = m1_width / m2_width
        scale_height = m1_height / m2_height
        return (scale_width + scale_height) / 2

    def to_m2(self, m1_point):
        """
        Transforms a point from the first model space to the second model space using the affine transformation matrix.

        Args:
        - m1_point (np.array): The point in the first model's coordinate space.

        Returns:
        - np.array or None: Transformed point in the second model's space if the transformation was successful; otherwise None.
        """
        if self.success:
            m1_point_homogeneous = np.append(m1_point, 1)  # Convert to homogeneous coordinates
            return np.dot(self.transform_matrix, m1_point_homogeneous) / self.scale_factor
        else:
            return None

    def to_m1(self, m2_point):
        """
        Transforms a point from the second model space back to the first model space using the inverse of the affine transformation matrix.

        Args:
        - m2_point (np.array): The point in the second model's coordinate space.

        Returns:
        - np.array or None: Transformed point back in the first model's space if the transformation was successful; otherwise None.
        """
        if self.success:
            affine_transform_4x4 = np.vstack([self.transform_matrix, [0, 0, 0, 1]])
            inverse_affine_transform = np.linalg.inv(affine_transform_4x4)
            m2_point_homogeneous = np.append(m2_point * self.scale_factor, 1)  # Convert to homogeneous coordinates
            m1_point_homogeneous = np.dot(inverse_affine_transform, m2_point_homogeneous)

            # Convert back to non-homogeneous coordinates
            return (m1_point_homogeneous[:3] / m1_point_homogeneous[3])
        else:
            return None


class IrisAffineTransformer:
    def __init__(self, left_iris, right_iris, frame_width, frame_height):
        """
        Initialize the transformer with iris landmarks and frame dimensions.

        Args:
            left_iris (np.array): Normalized left iris landmarks from MediaPipe
            right_iris (np.array): Normalized right iris landmarks from MediaPipe
            frame_width (float): Width of the frame in pixels
            frame_height (float): Height of the frame in pixels
        """
        self.camera = CameraParams(frame_width, frame_height)

        # Calculate depth (Z) based on iris size
        self.left_z, self.right_z = self._get_iris_depth(
            np.array(left_iris),
            np.array(right_iris),
            frame_width,
            frame_height
        )

        # Store iris centers in normalized coordinates
        self.left_center = self._get_iris_center(left_iris)
        self.right_center = self._get_iris_center(right_iris)

    def _get_iris_center(self, iris_points):
        """Calculate the center point of iris from its landmarks."""
        return np.mean(iris_points, axis=0)

    def _get_iris_depth(self, left_iris, right_iris, frame_width, frame_height):
        """
        Calculate depth based on iris diameter in image vs known real diameter.
        Returns depth in millimeters.
        """
        # Get horizontal diameter (points 0 and 2)
        left_hor_diameter = abs(
            (left_iris[0][0] - left_iris[2][0]) * self.camera.frame_width
        )
        right_hor_diameter = abs(
            (right_iris[0][0] - right_iris[2][0]) * self.camera.frame_width
        )

        # Get vertical diameter (points 1 and 3)
        left_ver_diameter = abs(
            (left_iris[1][1] - left_iris[3][1]) * self.camera.frame_height
        )
        right_ver_diameter = abs(
            (right_iris[1][1] - right_iris[3][1]) * self.camera.frame_height
        )

        # Calculate Z for each eye using both diameters
        # Use the known iris diameter (11.7mm) and camera focal length
        left_z = (
                         (self.camera.IRIS_DIAMETER_MM / left_hor_diameter) * self.camera.fx +
                         (self.camera.IRIS_DIAMETER_MM / left_ver_diameter) * self.camera.fy
                 ) / 2

        right_z = (
                          (self.camera.IRIS_DIAMETER_MM / right_hor_diameter) * self.camera.fx +
                          (self.camera.IRIS_DIAMETER_MM / right_ver_diameter) * self.camera.fy
                  ) / 2

        return left_z, right_z

    def get_iris_world_coordinates(self, iris_points, is_left=True):
        """
        Convert normalized iris coordinates to world coordinates in millimeters.

        Args:
            iris_points (np.array): Normalized iris landmarks from MediaPipe
            is_left (bool): Whether these points are for the left iris

        Returns:
            np.array: Array of 3D points in world coordinates (millimeters)
        """
        # Get Z depth for the appropriate eye
        z = self.left_z if is_left else self.right_z

        # Convert normalized coordinates to pixel coordinates
        points_px = np.zeros_like(iris_points)
        points_px[:, 0] = iris_points[:, 0] * self.camera.frame_width
        points_px[:, 1] = iris_points[:, 1] * self.camera.frame_height

        # Convert to world coordinates using perspective projection
        points_world = np.zeros((len(points_px), 3))
        for i, point in enumerate(points_px):
            # X coordinate
            points_world[i, 0] = (point[0] - self.camera.cx) * z / self.camera.fx
            # Y coordinate (note the sign flip due to y-axis direction)
            points_world[i, 1] = -(point[1] - self.camera.cy) * z / self.camera.fy
            # Z coordinate
            points_world[i, 2] = z

        return points_world


    # Calculate iris centers and normal vectors
    def get_normal_and_center(self, iris_world_coords):
        # Get vectors across the iris
        vector1 = iris_world_coords[2] - iris_world_coords[0]  # right - left
        vector2 = iris_world_coords[1] - iris_world_coords[3]  # top - bottom

        # Calculate normal using cross product
        normal = np.cross(vector1, vector2)
        normal = normal / np.linalg.norm(normal)  # normalize

        # Calculate iris center
        center = np.mean(iris_world_coords, axis=0)

        return normal, center


    # Calculate line-plane intersection at z=0
    def get_intersection(self, normal, center):
        # If normal[2] is 0, line is parallel to plane
        if abs(normal[2]) < 1e-6:
            return None

        # Calculate t where line intersects z=0 plane
        t = -center[2] / normal[2]

        # Calculate intersection point
        intersection = center + t * normal

        return intersection[:2]  # return only x,y coordinates


def get_monitor_size():
    for m in screeninfo.get_monitors():
        if m.is_primary:
            primary_monitor = m
    return primary_monitor

class CameraParams:
    """Logitech C920 specific parameters"""
    FOCAL_LENGTH_MM: float = 3.67
    HORIZONTAL_FOV: float = 70.42
    VERTICAL_FOV: float = 43.3
    MAX_RESOLUTION: tuple = (1920, 1080)
    IRIS_DIAMETER_MM = 11.7

    def __init__(self, frame_width, frame_height):
        self.frame_width = frame_width
        self.frame_height = frame_height

        # Calculate sensor dimensions
        self.sensor_width = 2 * self.FOCAL_LENGTH_MM * np.tan(np.radians(self.HORIZONTAL_FOV / 2))
        self.sensor_height = 2 * self.FOCAL_LENGTH_MM * np.tan(np.radians(self.VERTICAL_FOV / 2))

        # Calculate focal length in pixels
        self.fx = self.frame_width * self.FOCAL_LENGTH_MM / self.sensor_width
        self.fy = self.frame_height * self.FOCAL_LENGTH_MM / self.sensor_height

        # Principal point (usually at image center)
        self.cx = self.frame_width / 2
        self.cy = self.frame_height / 2

        # Pixel size in mm
        self.pixel_size_x = self.sensor_width / self.frame_width
        self.pixel_size_y = self.sensor_height / self.frame_height