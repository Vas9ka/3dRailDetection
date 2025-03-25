import numpy as np
import cv2
import raillabel

def load_frame_annotations(labels, frame_id=None):
    """Load annotations from a specific frame or the first frame if not specified."""
    polylines_2d = {}
    
    # Get the frame to process
    if frame_id is None:
        frame_id = list(labels.frames.keys())[0]
    
    if frame_id not in labels.frames:
        raise ValueError(f"Frame ID {frame_id} not found in dataset")
        
    frame = labels.frames[frame_id]
    print(f"Processing frame ID: {frame_id}")
    
    # Process frame annotations
    for annotation_id, annotation in frame.annotations.items():
        if annotation.sensor_id == "rgb_center":
            # Check if the annotation is a Poly2d object with rail attributes
            if (hasattr(annotation, 'points') and 
                hasattr(annotation, 'attributes') and 
                annotation.attributes.get('trackID') == '0'):  # Check for trackID '0' as string
                
                # Extract points from Poly2d points attribute
                points = []
                for point in annotation.points:
                    points.append([point.x, point.y])
                points = np.array(points)
                
                # Store the polyline with its attributes
                polylines_2d[annotation_id] = {
                    'points': points,
                    'closed': annotation.closed if hasattr(annotation, 'closed') else False,
                    'attributes': annotation.attributes
                }
                
                # Print information about the polyline
                rail_info = ""
                if 'railSide' in annotation.attributes:
                    rail_info = f", Rail: {annotation.attributes['railSide']}"
                    
                print(f"Found polyline: ID={annotation_id}, Points={len(points)}, Track=0{rail_info}")
    
    if not polylines_2d:
        print("No polylines found for track 0!")
    
    return polylines_2d


def load_3d_frame_annotations(labels, frame_id=None):
    """Load 3D annotations from a specific frame or the first frame if not specified."""
    polylines_3d = {}
    
    # Get the frame to process
    if frame_id is None:
        frame_id = list(labels.frames.keys())[0]
    
    if frame_id not in labels.frames:
        raise ValueError(f"Frame ID {frame_id} not found in dataset")
        
    frame = labels.frames[frame_id]
    print(f"Processing 3D frame ID: {frame_id}")
    
    # Process frame annotations
    for annotation_id, annotation in frame.annotations.items():
        # Skip non-LiDAR annotations
        if not hasattr(annotation, 'sensor_id') or annotation.sensor_id != 'lidar':
            continue
            
        # Check if this is a Poly3d annotation
        if type(annotation).__name__ == 'Poly3d' and hasattr(annotation, 'points'):
            # Check for trackID=0
            if not hasattr(annotation, 'attributes') or annotation.attributes.get('trackID') != '0':
                continue
                
            # Extract points from the points attribute
            points = []
            for point in annotation.points:
                points.append([point.x, point.y, point.z])
            points = np.array(points)
            
            # Store the 3D polyline with its attributes
            polylines_3d[annotation_id] = {
                'points': points,
                'closed': annotation.closed if hasattr(annotation, 'closed') else False,
                'attributes': annotation.attributes
            }
            
            # Print information about the polyline
            rail_info = ""
            if 'railSide' in annotation.attributes:
                rail_info = f", Rail: {annotation.attributes['railSide']}"
                
            print(f"Found 3D polyline: ID={annotation_id}, Points={len(points)}, Track=0{rail_info}")
    
    if not polylines_3d:
        print("No 3D polylines found for track 0!")
    else:
        print(f"Successfully found {len(polylines_3d)} 3D polylines")
    
    return polylines_3d


def create_rail_masks(polylines_2d, image_shape, scale_factor=1.0):
    """
    Create separate masks for left and right rails from polylines.
    
    Args:
        polylines_2d: Dictionary of polylines with attributes
        image_shape: Shape of the image to create masks for (height, width)
        scale_factor: Scale factor to apply to polyline coordinates
        
    Returns:
        tuple: (left_rail_mask, right_rail_mask)
    """
    # Create blank masks for left and right rails
    left_rail_mask = np.zeros((image_shape[0], image_shape[1]), dtype=np.uint8)
    right_rail_mask = np.zeros((image_shape[0], image_shape[1]), dtype=np.uint8)
    
    # Process each polyline
    for annotation_id, data in polylines_2d.items():
        if 'railSide' not in data['attributes']:
            continue
            
        points = data['points'].copy()
        
        # Scale points if needed
        if scale_factor != 1.0:
            points = points * scale_factor
            
        # Convert to integer coordinates
        points_int = points.astype(np.int32)
        
        # Draw the polyline on the appropriate mask
        if data['attributes']['railSide'] == 'leftRail':
            cv2.polylines(left_rail_mask, [points_int], False, 1, 2)
        elif data['attributes']['railSide'] == 'rightRail':
            cv2.polylines(right_rail_mask, [points_int], False, 1, 2)
    
    return left_rail_mask, right_rail_mask 