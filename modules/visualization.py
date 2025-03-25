import numpy as np
import matplotlib.pyplot as plt
import open3d as o3d

def create_lidar_geometries(polylines_3d):
    """
    Create Open3D visualization geometries for LiDAR polylines.
    
    Args:
        polylines_3d: Dictionary of 3D polylines with attributes
        
    Returns:
        List of Open3D geometry objects
    """
    lidar_geometries = []
    for annotation_id, data in polylines_3d.items():
        # Create point cloud for the polyline
        poly_pcd = o3d.geometry.PointCloud()
        poly_pcd.points = o3d.utility.Vector3dVector(data['points'])
        
        # Assign color based on rail side (if available)
        color = [0, 1, 0]  # Default: green
        if 'railSide' in data['attributes']:
            if data['attributes']['railSide'] == 'leftRail':
                color = [1, 0.5, 0]  # Orange for left rail
            elif data['attributes']['railSide'] == 'rightRail':
                color = [0, 0.5, 1]  # Light blue for right rail
        
        poly_pcd.colors = o3d.utility.Vector3dVector(np.ones_like(data['points']) * color)
        lidar_geometries.append(poly_pcd)
        
        # Create line set to show the polyline connections
        if len(data['points']) > 1:
            lines = [[i, i+1] for i in range(len(data['points'])-1)]
            line_set = o3d.geometry.LineSet()
            line_set.points = o3d.utility.Vector3dVector(data['points'])
            line_set.lines = o3d.utility.Vector2iVector(lines)
            line_set.colors = o3d.utility.Vector3dVector([color for _ in range(len(lines))])
            lidar_geometries.append(line_set)
    
    return lidar_geometries

def visualize_rails_3d(scene_pcd, left_rail_points_lidar, right_rail_points_lidar, lidar_geometries=None):
    """
    Visualize rails in 3D using Open3D.
    
    Args:
        scene_pcd: Open3D point cloud of the scene
        left_rail_points_lidar: Points for left rail
        right_rail_points_lidar: Points for right rail
        lidar_geometries: List of Open3D geometry objects for LiDAR annotations
    """
    # Create colored point clouds for visualization
    left_pcd = o3d.geometry.PointCloud()
    left_pcd.points = o3d.utility.Vector3dVector(left_rail_points_lidar)
    left_pcd.colors = o3d.utility.Vector3dVector(np.ones_like(left_rail_points_lidar) * [1, 0, 0])  # Red
    
    right_pcd = o3d.geometry.PointCloud()
    right_pcd.points = o3d.utility.Vector3dVector(right_rail_points_lidar)
    right_pcd.colors = o3d.utility.Vector3dVector(np.ones_like(right_rail_points_lidar) * [0, 0, 1])  # Blue
    
    # Add reference coordinate frame
    coordinate_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.5, origin=[0, 0, 0])
    
    # Visualize the 3D points
    vis = o3d.visualization.Visualizer()
    vis.create_window()
    vis.add_geometry(scene_pcd)
    vis.add_geometry(left_pcd)
    vis.add_geometry(right_pcd)
    
    # Add LiDAR polyline geometries if provided
    if lidar_geometries:
        for geometry in lidar_geometries:
            vis.add_geometry(geometry)
            
    # Add coordinate frame
    vis.add_geometry(coordinate_frame)
    
    # Configure rendering options
    opt = vis.get_render_option()
    opt.point_size = 1.5
    opt.line_width = 2.0  # Set line width for better visibility
    
    vis.run()   
    vis.destroy_window()

def visualize_centerlines_3d(scene_pcd, camera_centerline, lidar_centerline, 
                           left_rail_points_lidar, right_rail_points_lidar, lidar_geometries=None):
    """
    Visualize centerlines in 3D using Open3D.
    
    Args:
        scene_pcd: Open3D point cloud of the scene
        camera_centerline: Nx3 array of camera-derived centerline points
        lidar_centerline: Mx3 array of LiDAR centerline points
        left_rail_points_lidar: Points for left rail
        right_rail_points_lidar: Points for right rail
        lidar_geometries: List of Open3D geometry objects for LiDAR annotations
    """
    # Create rail point clouds
    left_pcd = o3d.geometry.PointCloud()
    left_pcd.points = o3d.utility.Vector3dVector(left_rail_points_lidar)
    left_pcd.colors = o3d.utility.Vector3dVector(np.ones_like(left_rail_points_lidar) * [1, 0, 0])  # Red
    
    right_pcd = o3d.geometry.PointCloud()
    right_pcd.points = o3d.utility.Vector3dVector(right_rail_points_lidar)
    right_pcd.colors = o3d.utility.Vector3dVector(np.ones_like(right_rail_points_lidar) * [0, 0, 1])  # Blue
    
    # Create centerline point clouds
    camera_centerline_pcd = o3d.geometry.PointCloud()
    camera_centerline_pcd.points = o3d.utility.Vector3dVector(camera_centerline)
    camera_centerline_pcd.colors = o3d.utility.Vector3dVector(np.ones_like(camera_centerline) * [1, 1, 0])  # Yellow
    
    lidar_centerline_pcd = o3d.geometry.PointCloud()
    lidar_centerline_pcd.points = o3d.utility.Vector3dVector(lidar_centerline)
    lidar_centerline_pcd.colors = o3d.utility.Vector3dVector(np.ones_like(lidar_centerline) * [0, 1, 0])  # Green
    
    # Create line sets for centerlines
    camera_line = o3d.geometry.LineSet()
    camera_line.points = o3d.utility.Vector3dVector(camera_centerline)
    camera_line.lines = o3d.utility.Vector2iVector([[i, i+1] for i in range(len(camera_centerline)-1)])
    camera_line.colors = o3d.utility.Vector3dVector(np.ones((len(camera_centerline)-1, 3)) * [1, 1, 0])  # Yellow
    
    lidar_line = o3d.geometry.LineSet()
    lidar_line.points = o3d.utility.Vector3dVector(lidar_centerline)
    lidar_line.lines = o3d.utility.Vector2iVector([[i, i+1] for i in range(len(lidar_centerline)-1)])
    lidar_line.colors = o3d.utility.Vector3dVector(np.ones((len(lidar_centerline)-1, 3)) * [0, 1, 0])  # Green
    
    # Add reference coordinate frame
    coordinate_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.5, origin=[0, 0, 0])
    
    # Visualize everything
    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name="Rail Centerline Comparison in 3D")
    
    # Add scene point cloud
    vis.add_geometry(scene_pcd)
    
    # Add rails
    vis.add_geometry(left_pcd)
    vis.add_geometry(right_pcd)
    
    # Add centerlines
    vis.add_geometry(camera_centerline_pcd)
    vis.add_geometry(lidar_centerline_pcd)
    vis.add_geometry(camera_line)
    vis.add_geometry(lidar_line)
    
    # Add spheres for better centerline visualization
    sphere_radius = 0.05
    interval = max(1, len(camera_centerline) // 20)
    
    for i in range(0, len(camera_centerline), interval):
        sphere = o3d.geometry.TriangleMesh.create_sphere(radius=sphere_radius)
        sphere.translate(camera_centerline[i])
        sphere.paint_uniform_color([1, 1, 0])  # Yellow
        vis.add_geometry(sphere)
    
    for i in range(0, len(lidar_centerline), interval):
        sphere = o3d.geometry.TriangleMesh.create_sphere(radius=sphere_radius)
        sphere.translate(lidar_centerline[i])
        sphere.paint_uniform_color([0, 1, 0])  # Green
        vis.add_geometry(sphere)
    
    # Add LiDAR polyline geometries if provided
    if lidar_geometries:
        for geometry in lidar_geometries:
            vis.add_geometry(geometry)
            
    # Add coordinate frame
    vis.add_geometry(coordinate_frame)
    
    # Configure rendering options
    opt = vis.get_render_option()
    opt.point_size = 1.5
    opt.line_width = 4.0  # Make lines thicker for visibility
    opt.background_color = np.array([0.1, 0.1, 0.1])  # Dark background for contrast
    
    vis.run()
    vis.destroy_window()

def plot_centerlines_top_down(camera_centerline, lidar_centerline, left_rail_points_lidar, right_rail_points_lidar):
    """
    Create a top-down 2D plot of centerlines and rails.
    
    Args:
        camera_centerline: Nx3 array of camera-derived centerline points
        lidar_centerline: Mx3 array of LiDAR centerline points
        left_rail_points_lidar: Points for left rail
        right_rail_points_lidar: Points for right rail
    """
    plt.figure(figsize=(12, 10))
    
    # Plot rails from top-down view
    if left_rail_points_lidar is not None:
        plt.scatter(left_rail_points_lidar[:, 0], left_rail_points_lidar[:, 1], 
                    c='red', s=3, alpha=0.5, label='Left Rail (Camera)')
    
    if right_rail_points_lidar is not None:
        plt.scatter(right_rail_points_lidar[:, 0], right_rail_points_lidar[:, 1], 
                    c='blue', s=3, alpha=0.5, label='Right Rail (Camera)')
    
    # Plot centerlines from top-down view (X-Y plane)
    plt.scatter(camera_centerline[:, 0], camera_centerline[:, 1], 
                c='gold', s=20, label='Camera Centerline')
    plt.scatter(lidar_centerline[:, 0], lidar_centerline[:, 1], 
                c='green', s=20, label='LiDAR Centerline')
    
    # Plot lines to connect points
    plt.plot(camera_centerline[:, 0], camera_centerline[:, 1], 'y-', linewidth=2.5)
    plt.plot(lidar_centerline[:, 0], lidar_centerline[:, 1], 'g-', linewidth=2.5)
    
    plt.xlabel('X (driving direction) [m]', fontsize=14)
    plt.ylabel('Y (left direction) [m]', fontsize=14)
    plt.title('Top-down View of Rail Centerlines', fontsize=16)
    plt.grid(True)
    plt.legend(fontsize=12)
    plt.axis('equal')  # Equal aspect ratio
    
    plt.tight_layout()
    plt.savefig('centerline_top_down.png', dpi=300)
    plt.show()

def plot_centerlines_with_error_markers(camera_centerline, lidar_centerline, error_results, 
                                       left_rail_points_lidar, right_rail_points_lidar):
    """
    Create a top-down 2D plot of centerlines with error markers at specific distances.
    """
    plt.figure(figsize=(14, 10))
    
    # Plot rails from top-down view with very small dots
    if left_rail_points_lidar is not None:
        plt.scatter(left_rail_points_lidar[:, 0], left_rail_points_lidar[:, 1], 
                    c='red', s=1, alpha=0.3, label='Left Rail (Camera)')
    
    if right_rail_points_lidar is not None:
        plt.scatter(right_rail_points_lidar[:, 0], right_rail_points_lidar[:, 1], 
                    c='blue', s=1, alpha=0.3, label='Right Rail (Camera)')
    
    # Plot centerlines as lines only, without scatter points
    plt.plot(camera_centerline[:, 0], camera_centerline[:, 1], 'y-', linewidth=2.5, label='Camera Centerline')
    plt.plot(lidar_centerline[:, 0], lidar_centerline[:, 1], 'g-', linewidth=2.5, label='LiDAR Centerline')
    
    # Plot error markers at specific distances
    for distance, result in error_results.items():
        if result['in_bounds']:
            camera_point = result['camera_point']
            lidar_point = result['lidar_point']
            error = result['error']
            
            # Draw a line connecting corresponding points
            plt.plot([camera_point[0], lidar_point[0]], 
                    [camera_point[1], lidar_point[1]], 
                    'r-', linewidth=2.0, zorder=9)
            
            # Use small hollow markers to mark the exact points without covering too much
            plt.plot(camera_point[0], camera_point[1], 'yo', markersize=6, 
                   markerfacecolor='none', markeredgewidth=2, zorder=10)
            plt.plot(lidar_point[0], lidar_point[1], 'go', markersize=6, 
                   markerfacecolor='none', markeredgewidth=2, zorder=10)
            
            # Position text labels to avoid overlap
            # Calculate midpoint for text placement
            mid_x = (camera_point[0] + lidar_point[0]) / 2
            mid_y = (camera_point[1] + lidar_point[1]) / 2
            
            # Add text showing distance and error with offset to avoid covering the line
            label_offset = 0.08  # Adjust based on your data scale
            
            # Calculate angle of error line for optimal text placement
            angle = np.arctan2(lidar_point[1] - camera_point[1], lidar_point[0] - camera_point[0])
            dx = label_offset * np.sin(angle)
            dy = label_offset * np.cos(angle)
            
            # Add error label with slight offset perpendicular to error line
            plt.text(mid_x + dx, mid_y + dy, 
                    f"{distance}m: {error:.3f}m", 
                    fontsize=10, ha='center', va='center',
                    bbox=dict(facecolor='white', alpha=0.8, pad=2, boxstyle='round,pad=0.3'),
                    zorder=11)
    
    plt.xlabel('X (driving direction) [m]', fontsize=14)
    plt.ylabel('Y (left direction) [m]', fontsize=14)
    plt.title('Top-down View of Rail Centerlines with Error Measurements', fontsize=16)
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=12)
    plt.axis('equal')  # Equal aspect ratio
    
    plt.tight_layout()
    plt.savefig('centerline_errors.png', dpi=300)
    plt.show() 