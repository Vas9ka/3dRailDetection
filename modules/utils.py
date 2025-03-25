import numpy as np
from scipy.spatial import KDTree

def sort_points_by_driving_direction(points):
    """
    Sort 3D points along the driving direction (X-axis in world coordinates).
    
    Args:
        points: Array of shape (N, 3) containing 3D points
        
    Returns:
        Sorted array of points
    """
    # Sort points by X coordinate (forward direction)
    indices = np.argsort(points[:, 0])
    return points[indices]

def create_polyline_from_points(points, min_distance=0.1):
    """
    Create an ordered polyline from 3D points, maintaining a minimum distance between points.
    
    Args:
        points: Array of shape (N, 3) containing 3D points
        min_distance: Minimum distance between consecutive points in the polyline
        
    Returns:
        Array of shape (M, 3) containing ordered polyline points
    """
    # Ensure points are sorted
    sorted_points = sort_points_by_driving_direction(points)
    
    # Start with the first point
    polyline = [sorted_points[0]]
    last_point = sorted_points[0]
    
    # Add points that maintain the minimum distance
    for point in sorted_points[1:]:
        distance = np.linalg.norm(point - last_point)
        if distance >= min_distance:
            polyline.append(point)
            last_point = point
    
    return np.array(polyline)

def calculate_centerline(left_rail, right_rail):
    """
    Calculate the centerline between left and right rail polylines.
    
    Args:
        left_rail: Array of shape (N, 3) for left rail points
        right_rail: Array of shape (M, 3) for right rail points
        
    Returns:
        Array of shape (P, 3) containing centerline points
    """
    # Ensure rails are sorted
    left_rail_sorted = sort_points_by_driving_direction(left_rail)
    right_rail_sorted = sort_points_by_driving_direction(right_rail)
    
    # Resample both rails to have the same number of points
    # Use the rail with fewer points as reference
    if len(left_rail_sorted) <= len(right_rail_sorted):
        reference_rail = left_rail_sorted
        other_rail = right_rail_sorted
    else:
        reference_rail = right_rail_sorted
        other_rail = left_rail_sorted
    
    # Build KD-Tree for the other rail
    tree = KDTree(other_rail)
    
    # Calculate centerline by averaging corresponding points
    centerline = []
    for point in reference_rail:
        # Find closest point in the other rail
        distance, index = tree.query(point)
        other_point = other_rail[index]
        
        # Calculate midpoint
        centerline_point = (point + other_point) / 2
        centerline.append(centerline_point)
    
    return np.array(centerline)

def point_to_polyline_distance(point, polyline):
    """
    Calculate the minimum distance from a point to a polyline.
    
    Args:
        point: Array of shape (3,) representing the query point
        polyline: Array of shape (N, 3) representing the polyline
        
    Returns:
        Minimum distance from the point to the polyline
    """
    min_distance = float('inf')
    
    # Check distance to each line segment
    for i in range(len(polyline) - 1):
        segment_start = polyline[i]
        segment_end = polyline[i + 1]
        
        # Vector from segment start to end
        segment_vector = segment_end - segment_start
        segment_length = np.linalg.norm(segment_vector)
        
        # Normalize segment vector
        if segment_length > 0:
            segment_unit = segment_vector / segment_length
        else:
            continue  # Skip zero-length segments
        
        # Vector from segment start to point
        point_vector = point - segment_start
        
        # Project point vector onto segment
        projection = np.dot(point_vector, segment_unit)
        
        # Calculate closest point on segment
        if projection <= 0:
            closest_point = segment_start
        elif projection >= segment_length:
            closest_point = segment_end
        else:
            closest_point = segment_start + projection * segment_unit
        
        # Calculate distance to closest point
        distance = np.linalg.norm(point - closest_point)
        
        # Update minimum distance
        if distance < min_distance:
            min_distance = distance
    
    return min_distance 