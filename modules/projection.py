import numpy as np
import cv2
from typing import Tuple

def apply_inverse_projection(
    mask: np.ndarray,
    height_m: float,
    tilt_angle_deg: float,
    camera_intrinsics: np.ndarray,
    distortion_coefficients: np.ndarray,
) -> np.ndarray:
    """Apply the inverse projection to the rail mask.

    A detailed description of the mathematically approach can be found in"Tang, 2024: A Real-Time Method for Railway
    Track Detection and 3D Fitting Based on Camera and LiDAR Fusion Sensing".

    Args:
        mask (np.ndarray): The rail mask.
        height_m (float): The height of the camera in meters.
        tilt_angle_deg (float): The tilt angle of the camera in degrees.
        camera_intrinsics (np.ndarray): The camera intrinsics.
        distortion_coefficients (np.ndarray): The distortion coefficients.

    Returns:
        np.ndarray: The rail mask in camera coordinates (Nx3 matrix of X, Y, Z coordinates).
    """

    def preprocess_mask(mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Preprocess the mask and extract the (v,u) pixel coordinates of the rail points."""
        # Undistort the mask
        undistorted = cv2.undistort(mask, camera_intrinsics, distortion_coefficients)

        rail_pixels = np.where(undistorted > 0)
        return rail_pixels[0], rail_pixels[1]

    def get_homogeneous_coords(rail_pixels: tuple[np.ndarray, np.ndarray]) -> np.ndarray:
        """Get the homogeneous coordinates of the rail mask as a (3, N) array in the form (u, v, 1)."""
        return np.vstack((rail_pixels[1], rail_pixels[0], np.ones_like(rail_pixels[0])))

    def compute_angles(v: np.ndarray, tilt_angle_deg: float) -> tuple[np.ndarray, np.ndarray]:
        """Compute vertical angles and trigonometric terms."""
        theta_1 = np.radians(tilt_angle_deg)
        theta_2 = np.arctan(v)
        return np.sin(theta_1 + theta_2), np.cos(theta_2)

    def get_normalized_coords(homogeneous_coords: np.ndarray, camera_intrinsics: np.ndarray) -> np.ndarray:
        """Normalize the coordinates so that the principle point (u0, v0) becomes the origin (0,0)."""
        return np.linalg.inv(camera_intrinsics) @ homogeneous_coords

    def project_to_camera_coords(
        normalized_coords: np.ndarray, sin_term: np.ndarray, cos_term: np.ndarray, height: float
    ) -> np.ndarray:
        """Project pixel coordinates to RDF (Right-Down-Forward) camera coordinates (X-right, Y-down, Z-forward)."""
        scale = height / sin_term
        X = scale * cos_term * normalized_coords[0]
        Y = scale * cos_term * normalized_coords[1]
        Z = scale * cos_term * normalized_coords[2]
        return np.vstack((X, Y, Z))

    def convert_rdf_to_flu(points: np.ndarray) -> np.ndarray:
        """Convert points from the RDF (Right-Down-Forward) coordinate system to the FLU (Forward-Left-Up) system.

        - X (RDF) -> Z (FLU)
        - Y (RDF) -> -X (FLU)
        - Z (RDF) -> -Y (FLU)
        """
        return np.vstack((points[2], -points[0], -points[1]))

    if len(mask.shape) != 2:
        raise ValueError(f"Array shape is {mask.shape}, expected a 2D mask.")

    rail_pixels = preprocess_mask(mask)
    homogeneous_coords = get_homogeneous_coords(rail_pixels)
    normalized_coords = get_normalized_coords(homogeneous_coords, camera_intrinsics)
    v = normalized_coords[1]
    sin_term, cos_term = compute_angles(v, tilt_angle_deg)
    camera_coords_rdf = project_to_camera_coords(normalized_coords, sin_term, cos_term, height_m)
    
    # Apply Z adjustment as in the original implementation
    camera_coords_rdf[2] -= camera_coords_rdf[2].min()

    camera_coords_flu = convert_rdf_to_flu(camera_coords_rdf)
    
    # Return FLU coordinates
    return camera_coords_flu.T

def apply_transform_to_points(points: np.ndarray, transform_matrix: np.ndarray) -> np.ndarray:
    """
    Apply a 4x4 transformation matrix to 3D points.
    
    Args:
        points: Nx3 array of 3D points
        transform_matrix: 4x4 homogeneous transformation matrix
        
    Returns:
        Nx3 array of transformed points
    """
    # Convert to homogeneous coordinates
    homogeneous_points = np.hstack((points, np.ones((points.shape[0], 1))))
    
    # Apply transformation
    transformed_points = homogeneous_points @ transform_matrix.T
    # Zero out Z coordinates to match run_projection.py behavior
    transformed_points[:, 2] = transformed_points[:, 2] * 0
    
    # Convert back to 3D coordinates
    return transformed_points[:, :3]

def camera_to_lidar_transform() -> np.ndarray:
    """
    Create a transformation matrix to convert from camera to LiDAR coordinates.
    Returns the 4x4 homogeneous transformation matrix.
    """
    # Camera to world (LiDAR) transformation matrix
    # This is the inverse of the world to camera matrix from the calibration file
    camera_to_world = np.array([
        [0.994801848338136, 0.008530331054488222, -0.101471749739171, 0],
        [-0.008253767163616355, 0.9999609911792882, 0.00314506714465518, 0],
        [0.1014946199098106, -0.002291194412618652, 0.9948334496575426, 0],
        [0.05399349045425253, -0.2858909428211028, 3.506285581367224, 1]
    ])
    
    # Calculate the inverse transform
    # For a homogeneous transformation matrix [R|t; 0 1], the inverse is [R^T | -R^T*t; 0 1]
    R = camera_to_world[:3, :3]
    t = camera_to_world[:3, 3]
    
    R_transpose = R.T
    t_new = -R_transpose @ t
    
    # Construct the inverse transformation matrix
    transform_matrix = np.eye(4)
    transform_matrix[:3, :3] = R_transpose
    transform_matrix[:3, 3] = t_new
    
    return transform_matrix

def transform_camera_to_lidar_coords(camera_points: np.ndarray) -> np.ndarray:
    """
    Transform points from camera coordinates to LiDAR coordinates.
    
    Args:
        camera_points: Nx3 array of points in camera coordinates
        
    Returns:
        Nx3 array of points in LiDAR coordinates
    """
    transform = camera_to_lidar_transform()
    return apply_transform_to_points(camera_points, transform) 