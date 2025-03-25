import numpy as np
from scipy.spatial import KDTree
from scipy.interpolate import interp1d

def calculate_centerline_distance_error(centerline1, centerline2):
    """
    Calculate distance errors between two centerlines.
    
    Args:
        centerline1: Nx3 array of first centerline points
        centerline2: Mx3 array of second centerline points
        
    Returns:
        Dict with error metrics (mean, max, min, std)
    """
    # Build KD-Tree for efficient nearest neighbor search
    tree = KDTree(centerline2)
    
    # Find distances from each point in centerline1 to nearest point in centerline2
    distances = []
    for point in centerline1:
        dist, _ = tree.query(point)
        distances.append(dist)
    
    # Calculate error metrics
    distances = np.array(distances)
    error_metrics = {
        'mean': np.mean(distances),
        'max': np.max(distances),
        'min': np.min(distances),
        'std': np.std(distances)
    }
    
    return error_metrics, distances

def calculate_errors_at_distances(camera_centerline, lidar_centerline, distances_to_check=[0, 5, 10, 20, 30, 50]):
    """
    Calculate distance errors between camera and LiDAR centerlines at specific distances along the track.
    
    Args:
        camera_centerline: Nx3 array of camera-derived centerline points
        lidar_centerline: Mx3 array of LiDAR centerline points
        distances_to_check: List of distances along the track to check (in meters)
    
    Returns:
        Dictionary of errors at each distance
    """
    results = {}
    
    # Sort centerlines by X coordinate (driving direction)
    camera_centerline = camera_centerline[np.argsort(camera_centerline[:, 0])]
    lidar_centerline = lidar_centerline[np.argsort(lidar_centerline[:, 0])]
    
    # Build KD-Tree for efficient nearest neighbor search in lidar centerline
    lidar_tree = KDTree(lidar_centerline)
    
    # Find min and max x values to ensure we don't go out of bounds
    camera_min_x = camera_centerline[0, 0]
    camera_max_x = camera_centerline[-1, 0]
    
    # Create interpolation functions for camera centerline Y and Z
    camera_x = camera_centerline[:, 0]
    camera_y = camera_centerline[:, 1]
    camera_z = camera_centerline[:, 2]
    
    # Create interpolation functions (use 'linear' for simple interpolation)
    interp_camera_y = interp1d(camera_x, camera_y, kind='linear', bounds_error=False, fill_value='extrapolate')
    interp_camera_z = interp1d(camera_x, camera_z, kind='linear', bounds_error=False, fill_value='extrapolate')
    
    # Create interpolation functions for lidar centerline
    lidar_x = lidar_centerline[:, 0]
    lidar_y = lidar_centerline[:, 1]
    lidar_z = lidar_centerline[:, 2]
    
    interp_lidar_y = interp1d(lidar_x, lidar_y, kind='linear', bounds_error=False, fill_value='extrapolate')
    interp_lidar_z = interp1d(lidar_x, lidar_z, kind='linear', bounds_error=False, fill_value='extrapolate')
    
    # Calculate the reference distance (0 meters) as the starting position
    # Use the average of camera and lidar starting positions
    x_ref = max(min(camera_centerline[:, 0]), min(lidar_centerline[:, 0]))
    
    # Calculate errors at each specified distance
    for distance in distances_to_check:
        x_pos = x_ref + distance  # Position along track
        
        # Check if position is within bounds
        if x_pos < max(camera_min_x, min(lidar_x)) or x_pos > min(camera_max_x, max(lidar_x)):
            results[distance] = {
                'error': None,
                'camera_point': None,
                'lidar_point': None,
                'in_bounds': False
            }
            continue
        
        # Get interpolated points at this X position
        camera_point = np.array([x_pos, interp_camera_y(x_pos), interp_camera_z(x_pos)])
        lidar_point = np.array([x_pos, interp_lidar_y(x_pos), interp_lidar_z(x_pos)])
        
        # Calculate 3D Euclidean distance
        error = np.linalg.norm(camera_point - lidar_point)
        
        results[distance] = {
            'error': error,
            'camera_point': camera_point,
            'lidar_point': lidar_point,
            'in_bounds': True,
            'lateral_error': abs(camera_point[1] - lidar_point[1]),  # Y-axis (lateral) error
            'vertical_error': abs(camera_point[2] - lidar_point[2])  # Z-axis (vertical) error
        }
    
    return results

def print_error_report(error_metrics, error_results, distances_to_check):
    """
    Print a comprehensive error report to the console.
    
    Args:
        error_metrics: Dictionary with overall error metrics
        error_results: Dictionary with errors at specific distances
        distances_to_check: List of distances that were checked
    """
    print("\nCenterline Error Metrics:")
    print(f"Mean distance: {error_metrics['mean']:.3f} m")
    print(f"Max distance: {error_metrics['max']:.3f} m")
    print(f"Min distance: {error_metrics['min']:.3f} m")
    print(f"Standard deviation: {error_metrics['std']:.3f} m")
    
    # Display results in a table format
    print("\nError Measurements at Specific Distances:")
    print(f"{'Distance (m)':<12} {'Total Error (m)':<15} {'Lateral Error (m)':<18} {'Vertical Error (m)':<15} {'In Bounds':<10}")
    print("-" * 70)
    
    for distance in distances_to_check:
        result = error_results[distance]
        if result['in_bounds']:
            print(f"{distance:<12} {result['error']:<15.3f} {result['lateral_error']:<18.3f} {result['vertical_error']:<15.3f} {'Yes':<10}")
        else:
            print(f"{distance:<12} {'N/A':<15} {'N/A':<18} {'N/A':<15} {'No':<10}") 