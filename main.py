import numpy as np
import cv2
import open3d as o3d
import raillabel
import click

from modules.visualization import (create_lidar_geometries, visualize_rails_3d, 
                                  visualize_centerlines_3d, plot_centerlines_top_down, 
                                  plot_centerlines_with_error_markers)
from modules.projection import apply_inverse_projection, transform_camera_to_lidar_coords
from modules.data_loading import load_frame_annotations, load_3d_frame_annotations, create_rail_masks
from modules.utils import create_polyline_from_points, calculate_centerline
from modules.evaluation import (calculate_centerline_distance_error, 
                               calculate_errors_at_distances, print_error_report)

# Camera parameters
HEIGHT = 3.50629
CAMERA_INTRINSICS = [4609.471892628096, 0, 1257.158605934,
                     0, 4609.471892628096, 820.0498076210201,
                     0, 0, 1]
CAMERA_INTRINSICS = np.array(CAMERA_INTRINSICS).reshape(3, 3)
DIST_COEFFS = np.array([-0.0914603, 0.605326, 0, 0, 0.417134])
TILT_ANGLE = 0

# Evaluation parameters
DISTANCES_TO_CHECK = [0, 5, 10, 20, 30, 50]

@click.command()
@click.option('--scene-pcd', '-s', required=True, type=click.Path(exists=True), 
              help='Path to the LiDAR point cloud (.pcd) file')
@click.option('--labels', '-l', required=True, type=click.Path(exists=True), 
              help='Path to the labels (.json) file')
@click.option('--image', '-i', required=True, type=click.Path(exists=True), 
              help='Path to the RGB image file')
def main(scene_pcd, labels, image):
    # Use provided file paths instead of hard-coded ones
    scene_pcd_path = scene_pcd
    labels_path = labels
    image_path = image
    
    # Load point cloud
    print("Loading point cloud...")
    scene_pcd = o3d.t.io.read_point_cloud(scene_pcd_path)
    scene_pcd = scene_pcd.to_legacy()
    
    # Load labels
    print("Loading labels...")
    labels = raillabel.load(labels_path)
    
    # Load 2D and 3D annotations
    print("\nLoading annotations...")
    polylines_2d = load_frame_annotations(labels)
    polylines_3d = load_3d_frame_annotations(labels)
    
    print(f"\nFound {len(polylines_2d)} 2D polylines")
    print(f"Found {len(polylines_3d)} 3D polylines")
    
    # Load image and process it
    original_image = cv2.imread(image_path)
    
    if original_image is None:
        print(f"Error: Could not load image from {image_path}")
        import os
        print(f"Current working directory: {os.getcwd()}")
        print(f"File exists check: {os.path.exists(image_path)}")
        return
    
    # Create rail masks
    print("\nCreating rail masks...")
    left_rail_mask, right_rail_mask = create_rail_masks(polylines_2d, original_image.shape[:2], scale_factor=1.0)
    
    # Apply inverse projection to get 3D points in camera coordinates
    print("Projecting 2D masks to 3D...")
    left_rail_points = apply_inverse_projection(left_rail_mask, HEIGHT, TILT_ANGLE, CAMERA_INTRINSICS, DIST_COEFFS)
    right_rail_points = apply_inverse_projection(right_rail_mask, HEIGHT, TILT_ANGLE, CAMERA_INTRINSICS, DIST_COEFFS)
    
    # Transform camera coordinates to LiDAR coordinates
    print("Transforming to LiDAR coordinates...")
    left_rail_points_lidar = transform_camera_to_lidar_coords(left_rail_points)
    right_rail_points_lidar = transform_camera_to_lidar_coords(right_rail_points)
    
    # Create LiDAR geometries for visualization
    lidar_geometries = create_lidar_geometries(polylines_3d)
    
    # Visualize rails in 3D
    print("\nVisualizing rails in 3D...")
    visualize_rails_3d(scene_pcd, left_rail_points_lidar, right_rail_points_lidar, lidar_geometries)
    
    # Create polylines from the points
    print("\nCreating polylines and centerlines...")
    left_rail_polyline = create_polyline_from_points(left_rail_points_lidar, min_distance=0.1)
    right_rail_polyline = create_polyline_from_points(right_rail_points_lidar, min_distance=0.1)
    
    # Calculate the centerline from camera-derived rails
    camera_centerline = calculate_centerline(left_rail_polyline, right_rail_polyline)
    
    # Extract polylines from LiDAR annotations
    lidar_left_points = None
    lidar_right_points = None
    
    for annotation_id, data in polylines_3d.items():
        if data['attributes'].get('railSide') == 'leftRail':
            lidar_left_points = data['points']
        elif data['attributes'].get('railSide') == 'rightRail':
            lidar_right_points = data['points']
    
    # If we have both left and right rails from LiDAR
    if lidar_left_points is not None and lidar_right_points is not None:
        # Calculate LiDAR centerline
        lidar_centerline = calculate_centerline(lidar_left_points, lidar_right_points)
        
        # Calculate error metrics
        print("\nCalculating error metrics...")
        error_metrics, point_distances = calculate_centerline_distance_error(
            camera_centerline, lidar_centerline)
        
        # Calculate errors at specific distances
        error_results = calculate_errors_at_distances(
            camera_centerline, lidar_centerline, DISTANCES_TO_CHECK)
        
        # Print error report
        print_error_report(error_metrics, error_results, DISTANCES_TO_CHECK)
        
        # Create the top-down view plot
        print("\nGenerating 2D plots...")
        plot_centerlines_top_down(camera_centerline, lidar_centerline, 
                                left_rail_points_lidar, right_rail_points_lidar)
        
        # Create the enhanced top-down plot with error markers
        plot_centerlines_with_error_markers(camera_centerline, lidar_centerline, error_results,
                                         left_rail_points_lidar, right_rail_points_lidar)
        
        # Visualize centerlines in 3D
        print("\nVisualizing centerlines in 3D...")
        visualize_centerlines_3d(scene_pcd, camera_centerline, lidar_centerline,
                              left_rail_points_lidar, right_rail_points_lidar, lidar_geometries)
    else:
        print("Cannot calculate centerline - missing LiDAR data for one or both rails")

if __name__ == "__main__":
    main() 

