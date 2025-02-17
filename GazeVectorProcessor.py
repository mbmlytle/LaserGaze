import numpy as np
from screeninfo import get_monitors


class GazeVectorProcessor:
    def __init__(self):
        """Initialize with monitor information"""
        self.monitor = self._get_monitor_size()

    def _get_monitor_size(self):
        """Get primary monitor information"""
        for m in get_monitors():
            if m.is_primary:
                return m
        raise RuntimeError("No primary monitor found")

    def calculate_gaze_normal(self, iris_points_world):
        """
        Calculate normal vector from iris points in real world coordinates.
        Uses cross product of two vectors formed by iris points.

        Args:
            iris_points_world: Array of 4 iris points in real world coordinates [mm]
                             [left, top, right, bottom]
        Returns:
            numpy array: Normalized gaze vector
        """
        # Create vectors from opposite iris points
        vector1 = iris_points_world[2] - iris_points_world[0]  # right - left
        vector2 = iris_points_world[1] - iris_points_world[3]  # top - bottom

        # Calculate normal vector using cross product
        normal = np.cross(vector1, vector2)

        # Normalize the vector
        return normal / np.linalg.norm(normal)

    def find_plane_intersection(self, origin, direction, plane_z=0):
        """
        Find intersection of gaze vector with z=0 plane (monitor plane)

        Args:
            origin: 3D point where the vector originates (eye position) [mm]
            direction: Normalized direction vector
            plane_z: Z-coordinate of the plane (default 0)

        Returns:
            numpy array or None: Intersection point [x,y,z] if it exists
        """
        # Avoid division by zero
        if abs(direction[2]) < 1e-6:
            return None

        # Calculate t parameter for line-plane intersection
        t = (plane_z - origin[2]) / direction[2]

        # If t is negative, the intersection is behind the eye
        if t < 0:
            return None

        # Calculate intersection point
        intersection = origin + t * direction
        return intersection

    def monitor_transform(self, world_point):
        """
        Transform world coordinates to monitor pixel coordinates
        Assumes camera is centered at top of monitor

        Args:
            world_point: [x,y,z] point in world coordinates [mm]

        Returns:
            tuple: (pixel_x, pixel_y) coordinates on monitor
        """
        point_x, point_y = world_point[0], world_point[1]

        # Convert world coordinates to pixel coordinates
        pixel_x = ((1000 * point_x) - (self.monitor.width_mm / 2)) * (self.monitor.width / self.monitor.width_mm)
        pixel_y = ((1000 * point_y) - self.monitor.height_mm) * (self.monitor.height / self.monitor.height_mm)

        # Convert to integers and ensure within monitor bounds
        pixel_x = max(0, min(int(pixel_x), self.monitor.width))
        pixel_y = max(0, min(int(pixel_y), self.monitor.height))

        return pixel_x, pixel_y

    def process_gaze(self, iris_points_world):
        """
        Process iris points to get screen coordinates

        Args:
            iris_points_world: Array of 4 iris points in world coordinates [mm]
            eye_position: 3D position of eye center [mm]

        Returns:
            tuple or None: (pixel_x, pixel_y) if gaze intersects monitor
        """
        # Calculate gaze direction
        gaze_vector = self.calculate_gaze_normal(iris_points_world)

        # Find intersection with monitor plane
        intersection = self.find_plane_intersection(gaze_vector)
        if intersection is None:
            return None

        # Convert to screen coordinates
        return self.monitor_transform(intersection)