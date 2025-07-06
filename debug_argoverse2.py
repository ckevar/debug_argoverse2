import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw
import os
import numpy as np
from scipy.spatial import ConvexHull

import overlapping

def quaternion_to_transformation_matrix(x, y, z, w, tx, ty, tz):
    """
    Converts a quaternion (x, y, z, w) and translation (tx, ty, tz)
    into a 4x4 transformation matrix.

    Args:
        x, y, z, w (float): Quaternion components.
        tx, ty, tz (float): Translation components along x, y, and z axes.

    Returns:
        np.array: A 4x4 homogeneous transformation matrix.
    """
    # Rotation matrix elements derived from quaternion components
    r00 = 1 - 2 * (y**2 + z**2)
    r01 = 2 * (x * y - w * z)
    r02 = 2 * (x * z + w * y)

    r10 = 2 * (x * y + w * z)
    r11 = 1 - 2 * (x**2 + z**2)
    r12 = 2 * (y * z - w * x)

    r20 = 2 * (x * z - w * y)
    r21 = 2 * (y * z + w * x)
    r22 = 1 - 2 * (x**2 + y**2)

    # Construct the 4x4 transformation matrix
    return np.array([
        [r00, r01, r02, tx],
        [r10, r11, r12, ty],
        [r20, r21, r22, tz],
        [0,   0,   0,   1]
    ])

def get_images_timestamp(camera_directory: str, camera_name: str) -> np.array:
    """
    Retrieves and sorts timestamps of images in a specified camera directory.

    Args:
        camera_directory (str): The base directory containing camera folders.
        camera_name (str): The name of the specific camera (e.g., 'ring_front_center').

    Returns:
        np.array: A sorted NumPy array of image timestamps (integers).
    """
    # List all files in the specific camera's directory
    image_files = os.listdir(os.path.join(camera_directory, camera_name))
    
    # Pre-allocate array for efficiency
    image_timestamps = np.empty(len(image_files), dtype=np.int64)

    # Extract timestamp from each image filename (assuming 'timestamp.jpg' format)
    for i, img_file in enumerate(image_files):
        # Remove the '.jpg' extension and convert to integer
        image_timestamps[i] = int(img_file[:-4])

    return np.sort(image_timestamps)

def get_annotations_timestamp(annotations_df: pd.DataFrame) -> np.array:
    """
    Extracts unique and sorted timestamps from the annotations DataFrame.

    Args:
        annotations_df (pd.DataFrame): DataFrame containing annotation data.

    Returns:
        np.array: A sorted NumPy array of unique annotation timestamps.
    """
    # Extract 'timestamp_ns' column and convert to NumPy array
    timestamps = annotations_df['timestamp_ns'].to_numpy()
    
    # Get unique timestamps and sort them
    return np.unique(np.sort(timestamps))

def get_transformation_vehicle(calibration_df: pd.DataFrame, sensor_name: str) -> np.array:
    """
    Extracts the transformation matrix from the vehicle to a specific sensor.

    Args:
        calibration_df (pd.DataFrame): DataFrame containing vehicle-to-sensor calibration data.
        sensor_name (str): The name of the sensor for which to get the transformation.

    Returns:
        np.array: A 4x4 homogeneous transformation matrix from vehicle to sensor.
    """
    # Filter the DataFrame for the specific sensor
    sensor_entry = calibration_df[calibration_df['sensor_name'] == sensor_name]
    
    # Extract quaternion and translation components for the sensor
    qx = sensor_entry['qx'].item()
    qy = sensor_entry['qy'].item()
    qz = sensor_entry['qz'].item()
    qw = sensor_entry['qw'].item()
    tx = sensor_entry['tx_m'].item()
    ty = sensor_entry['ty_m'].item()
    tz = sensor_entry['tz_m'].item()

    # Convert quaternion and translation to a transformation matrix
    return quaternion_to_transformation_matrix(qx, qy, qz, qw, tx, ty, tz)

def get_box_transformation(box_data: tuple) -> np.array:
    """
    Extracts the transformation matrix for a single 3D bounding box.

    Args:
        box_data (tuple): A tuple where the second element is a pandas Series
                          containing 'qx', 'qy', 'qz', 'qw', 'tx_m', 'ty_m', 'tz_m'.

    Returns:
        np.array: A 4x4 homogeneous transformation matrix for the bounding box.
    """
    # The box data is expected as (index, Series) from iterrows()
    s3e = box_data[1]

    # Extract quaternion and translation components for the bounding box
    qx = s3e['qx']
    qy = s3e['qy']
    qz = s3e['qz']
    qw = s3e['qw']
    tx = s3e['tx_m']
    ty = s3e['ty_m']
    tz = s3e['tz_m']

    # Convert quaternion and translation to a transformation matrix
    return quaternion_to_transformation_matrix(qx, qy, qz, qw, tx, ty, tz)

def get_intrinsic_and_distortion_parameters(
        intrinsics_df: pd.DataFrame, 
        sensor_name: str
        ) -> tuple[np.array, np.array]:
    """
    Extracts the camera intrinsic matrix and radial distortion coefficients for a sensor.

    Args:
        intrinsics_df (pd.DataFrame): DataFrame containing camera intrinsic parameters.
        sensor_name (str): The name of the sensor for which to get intrinsics.

    Returns:
        tuple[np.array, np.array]: A tuple containing:
            - np.array: The 3x4 camera intrinsic matrix (K).
            - np.array: A 1D array of radial distortion coefficients [k1, k2, k3].
    """
    # Filter the DataFrame for the specific sensor's intrinsics
    camera_intrinsics = intrinsics_df[intrinsics_df['sensor_name'] == sensor_name]

    # Extract intrinsic parameters
    fx = camera_intrinsics['fx_px'].item()
    fy = camera_intrinsics['fy_px'].item()
    cx = camera_intrinsics['cx_px'].item()
    cy = camera_intrinsics['cy_px'].item()

    # Extract radial distortion coefficients
    k1 = camera_intrinsics['k1'].item()
    k2 = camera_intrinsics['k2'].item()
    k3 = camera_intrinsics['k3'].item()

    # Construct the camera intrinsic matrix (K)
    intrinsic_matrix = np.array([
        [fx, 0, cx, 0],
        [0, fy, cy, 0],
        [0,  0,  1, 0]
    ])
    
    # Radial distortion coefficients
    radial_distortion_coefficients = np.array([k1, k2, k3])

    return intrinsic_matrix, radial_distortion_coefficients

def correct_radial_distortion(
        points_2d: np.array, 
        intrinsic_matrix: np.array, 
        radial_distortion_coeffs: np.array
        ) -> np.array:
    """
    Corrects radial lens distortion for 2D image points.

    Args:
        points_2d (np.array): A 2xN NumPy array of 2D image points (x, y).
        intrinsic_matrix (np.array): The 3x4 camera intrinsic matrix.
        radial_distortion_coeffs (np.array): A 1D array of radial distortion coefficients [k1, k2, k3].

    Returns:
        np.array: A 2xN NumPy array of distortion-corrected 2D image points.
    """
    # Extract intrinsic parameters from the matrix
    cx = intrinsic_matrix[0, 2]
    cy = intrinsic_matrix[1, 2]
    fx = intrinsic_matrix[0, 0]
    fy = intrinsic_matrix[1, 1]

    # Extract distortion coefficients
    k1 = radial_distortion_coeffs[0]
    k2 = radial_distortion_coeffs[1]
    k3 = radial_distortion_coeffs[2]

    # Normalize points relative to the principal point and focal length
    normalized_x = (points_2d[0, :] - cx) / fx
    normalized_y = (points_2d[1, :] - cy) / fy

    # Calculate squared radial distance
    r_squared = normalized_x**2 + normalized_y**2
    
    # Calculate the distortion factor (k)
    distortion_factor = 1 + r_squared * (k1 + k2 * r_squared + k3 * (r_squared**2))

    # Apply distortion correction
    points_2d[0, :] = cx + normalized_x * distortion_factor * fx
    points_2d[1, :] = cy + normalized_y * distortion_factor * fy

    return points_2d

def get_upright_3d_box_corners(bboxes_3d_df: pd.DataFrame) -> np.array:
    """
    Calculates the 3D world coordinates of the 8 corners for multiple upright bounding boxes.

    Args:
        bboxes_3d_df (pd.DataFrame): DataFrame containing 3D bounding box properties
                                     (length_m, width_m, height_m, and transformation info).

    Returns:
        np.array: A Nx8x4 NumPy array, where N is the number of boxes, and each 8x4
                  matrix represents the 8 corners (x, y, z, 1) in world coordinates.
    """
    # Half dimensions of the bounding boxes
    half_length = bboxes_3d_df['length_m'].to_numpy() * 0.5
    half_width = bboxes_3d_df['width_m'].to_numpy() * 0.5
    half_height = bboxes_3d_df['height_m'].to_numpy() * 0.5

    # Define the 8 corners of a generic unit box, scaled by half dimensions
    # Order: [l, w, h] permutations for bottom (negative h) and top (positive h)
    corners_relative = np.stack([
        half_length, half_width, -half_height,  # Front-top-right (bottom)
        -half_length, half_width, -half_height, # Front-top-left (bottom)
        -half_length, -half_width, -half_height, # Back-top-left (bottom)
        half_length, -half_width, -half_height, # Back-top-right (bottom)
        half_length, half_width, half_height,   # Front-top-right (top)
        -half_length, half_width, half_height,  # Front-top-left (top)
        -half_length, -half_width, half_height, # Back-top-left (top)
        half_length, -half_width, half_height   # Back-top-right (top)
    ], axis=-1).reshape(-1, 8, 3) # Reshape to (num_boxes, 8, 3)

    transformed_corners = []
    # Iterate through each box and apply its specific transformation
    for corner_set, box_series in zip(corners_relative, bboxes_3d_df.iterrows()):
        # Get the transformation matrix for the current box
        box_transform_matrix = get_box_transformation(box_series)
        
        # Convert corners to homogeneous coordinates (add a row of ones)
        # Transpose to 4x8 for matrix multiplication (corners as columns)
        corners_homogeneous = np.concatenate((corner_set, np.ones((8, 1))), axis=1).T
        
        # Apply the transformation: T @ corners
        transformed_corner_set = box_transform_matrix @ corners_homogeneous
        
        # Transpose back to 8x4 and append
        transformed_corners.append(transformed_corner_set.T)

    return np.array(transformed_corners)

def get_upright_3d_box_corners_with_center(bboxes_3d_df: pd.DataFrame) -> np.array:
    """
    Calculates the 3D world coordinates of the 8 corners and the center for multiple upright bounding boxes.

    Args:
        bboxes_3d_df (pd.DataFrame): DataFrame containing 3D bounding box properties
                                     (length_m, width_m, height_m, and transformation info).

    Returns:
        np.array: A Nx9x4 NumPy array, where N is the number of boxes, and each 9x4
                  matrix represents the 8 corners and the center (x, y, z, 1) in world coordinates.
    """
    # Half dimensions of the bounding boxes
    half_length = bboxes_3d_df['length_m'].to_numpy() * 0.5
    half_width = bboxes_3d_df['width_m'].to_numpy() * 0.5
    half_height = bboxes_3d_df['height_m'].to_numpy() * 0.5
    
    num_boxes = len(half_length) # Number of bounding boxes

    # Define the 8 corners of a generic unit box and a center point (0,0,0), scaled by half dimensions
    # Order: [l, w, h] permutations for bottom (negative h) and top (positive h)
    corners_and_center_relative = np.stack([
        half_length, half_width, -half_height,  # Front-top-right (bottom)
        -half_length, half_width, -half_height, # Front-top-left (bottom)
        -half_length, -half_width, -half_height, # Back-top-left (bottom)
        half_length, -half_width, -half_height, # Back-top-right (bottom)
        half_length, half_width, half_height,   # Front-top-right (top)
        -half_length, half_width, half_height,  # Front-top-left (top)
        -half_length, -half_width, half_height, # Back-top-left (top)
        half_length, -half_width, half_height,  # Back-top-right (top)
        np.zeros(num_boxes), np.zeros(num_boxes), np.zeros(num_boxes) # Center point (0,0,0)
    ], axis=-1).reshape(-1, 9, 3) # Reshape to (num_boxes, 9, 3)

    transformed_points = []
    # Iterate through each box and apply its specific transformation
    for points_set, box_series in zip(corners_and_center_relative, bboxes_3d_df.iterrows()):
        # Get the transformation matrix for the current box
        box_transform_matrix = get_box_transformation(box_series)
        
        # Convert points to homogeneous coordinates (add a row of ones)
        # Transpose to 4x9 for matrix multiplication (points as columns)
        points_homogeneous = np.concatenate((points_set, np.ones((9, 1))), axis=1).T
        
        # Apply the transformation: T @ points
        transformed_points_set = box_transform_matrix @ points_homogeneous
        
        # Transpose back to 9x4 and append
        transformed_points.append(transformed_points_set.T)

    return np.array(transformed_points)

def render_projected(camera_path: str, 
                     image_id: int, 
                     projected_2d_boxes: list[list[float]]
                     output_dir="/tmp": str):
    """
    Opens an image, draws 2D projected bounding boxes on it, and saves the result.

    Args:
        camera_path (str): The directory where the camera images are stored.
        image_id (int): The timestamp ID of the image to open.
        projected_2d_boxes (list[list[float]]): A list of bounding box coordinates,
                                                 each represented as [x1, y1, x2, y2].
    """
    # Construct the full image path
    image_file_path = f"{camera_path}/{image_id}.jpg"
    
    # Open the image using Pillow
    img_pil = Image.open(image_file_path)
    draw = ImageDraw.Draw(img_pil)

    # Draw each projected bounding box rectangle
    for bbox_coords in projected_2d_boxes:
        draw.rectangle(bbox_coords, width=2, outline="red") # Added outline color for visibility

    # Define output path and save the image
    output_path = f"{output_dir}/projected_bbox_{image_id}.jpg" # Changed output filename for clarity
    print(f"Saving projected image to: {output_path}")
    img_pil.save(output_path)
    img_pil.close()

def render_polygons(camera_path: str, 
                    image_id: int, 
                    projected_2d_boxes: list[list[float]],
                    output_dir="/tmp"):
    # Construct the full image path
    image_file_path = f"{camera_path}/{image_id}.jpg"
    
    # Open the image using Pillow
    img_pil = Image.open(image_file_path)
    draw = ImageDraw.Draw(img_pil)

    # Draw each projected bounding box rectangle
    for bbox_coords in projected_2d_boxes:
        draw.polygon(bbox_coords, width=2, outline="red") # Added outline color for visibility

    # Define output path and save the image
    output_path = f"{output_dir}/projected_pbox_{image_id}.jpg" # Changed output filename for clarity
    print(f"Saving projected image to: {output_path}")
    img_pil.save(output_path)
    img_pil.close()



def render_3d_points(image_file_path: str, points_2d: np.array):
    """
    Opens an image and draws individual 2D projected points on it.
    This function is currently not used in the main script logic.

    Args:
        image_file_path (str): The full path to the image file.
        points_2d (np.array): A 2xN NumPy array of 2D image points (x, y).
    """
    # Open the image using Pillow
    img_pil = Image.open(image_file_path)
    draw = ImageDraw.Draw(img_pil)
    
    # Transpose points to be Nx2 (for (x,y) pairs) for easier iteration
    points_to_draw = points_2d.T
    for x, y in points_to_draw:
        draw.point((x, y), fill="blue") # Added fill color for visibility

    img_pil.show() # Display the image
    img_pil.close()

# Function to get the convex hull using scipy
def get_polygon_from_projected_points(projected_points):
    """
    Computes the convex hull of a set of 2D points to find the polygon outline.

    Args:
        projected_points (np.array): A (N, 2) numpy array of 2D projected coordinates.

    Returns:
        list of tuples: List of (x, y) coordinates forming the convex hull polygon.
                        Returns an empty list if less than 3 points are provided.
    """
    ppT = projected_points.T
    if len(ppT) < 3:
        print("Warning: Not enough points to form a polygon (need at least 3).")
        return []

    # Compute the convex hull
    hull = ConvexHull(ppT)

    # The `vertices` attribute of the ConvexHull object gives the indices
    # of the points that form the convex hull, in counter-clockwise order.
    hull_points = ppT[hull.vertices]
    polygon_outline = [(0, 0)] * len(hull_points)

    for i, p in enumerate(hull_points):
        if p[0] < 0: p[0] = 0
        if p[1] < 1: p[1] = 0
        polygon_outline[i] = (p[0], p[1])

    return polygon_outline

def compute_overlapping(polygons):

    for i, clipping_poly in enumerate(polygons):
        for j in range(i + 1, len(polygons)):
            subject_poly = polygons[j]
            intersection = overlapping.sutherland_hodgman_original(subject_poly, clipping_poly)
            if intersection:
                intersection_area = overlapping.polygon_area(intersection)
                subject_area = overlapping.polygon_area(subject_poly) / intersection_area
                clipping_area = overlapping.polygon_area(clipping_poly) / intersection_area
                print(f"subject_area {subject_area}, clipping_area: {clipping_area}")
                overlapping.visualize_polygons_with_intersection(clipping_poly, subject_poly, intersection)

if __name__ == "__main__":
    # Convex hulls are the polygons that box the objects, instead of simple
    # bounding boxes, they are outter edges of the projected cuboids, so, we can
    # call them bounding polygons.
    convex_hulls_enabled = False 
    
    # Hulls overlapping, it computes the intersection of the overlapping of the
    # bounding polygons, it will pop a plot, showing two bounding polygons and
    # their intersection if avaialable, it also computes the area of the 
    # intersection polygon.
    hulls_overlapping_enabled = False

    # --- Configuration and Data Loading ---
    root_directory = "/home/chris/Documents/Datasets/mot/argoverse2"
    train_directory = os.path.join(root_directory, "train-000/sensor/train")
    
    # Choose one sequence directory for processing
    # sequence_directory = os.path.join(train_directory, "00a6ffc1-6ce9-3bc3-a060-6006e9893a1a")
    # sequence_directory = os.path.join(train_directory, "0a8a4cfa-4902-3a76-8301-08698d6290a2")
    sequence_directory = os.path.join(train_directory, "0b9321c1-9bc2-4727-beb8-8046aa4bb6c4")

    # Define paths to annotation and calibration files
    annotations_path = os.path.join(sequence_directory, "annotations.feather")
    vehicle_to_sensor_calib_path = os.path.join(sequence_directory, "calibration/egovehicle_SE3_sensor.feather")
    intrinsics_path = os.path.join(sequence_directory, "calibration/intrinsics.feather")

    # Define camera directory and specific camera to use
    cameras_directory = os.path.join(sequence_directory, "sensors/cameras")
    camera_name_to_use = "ring_front_center"
    
    # Camera Horizontal Field of View (used for filtering boxes)
    camera_fov_radians = 30 * np.pi / 90 # Original conversion was slightly off, corrected to np.deg2rad(60) if 60 deg is intended

    # Load dataframes
    annotations_df = pd.read_feather(annotations_path)
    calibration_df = pd.read_feather(vehicle_to_sensor_calib_path)
    intrinsics_df = pd.read_feather(intrinsics_path)

    # --- Pre-calculate Transformations and Intrinsics ---
    # Get the inverse transformation from world to vehicle (T_world_to_vehicle)
    # Assuming get_transformation_vehicle gives T_vehicle_to_sensor, then inv(T_vehicle_to_world) if it's Ego-vehicle.
    # Clarification needed: if calib_df gives T_ego_vehicle_to_sensor, then T_sensor_to_ego_vehicle = inv(T_ego_vehicle_to_sensor).
    # And if bounding boxes are in World frame, we need T_sensor_to_world = T_sensor_to_ego_vehicle @ T_ego_vehicle_to_world.
    # The current use of T_vehicle here suggests it's World_to_Vehicle. Let's assume it's T_world_to_vehicle.
    world_to_vehicle_transform = np.linalg.inv(get_transformation_vehicle(calibration_df, camera_name_to_use))
    
    # Get camera intrinsic matrix and radial distortion coefficients
    camera_intrinsic_matrix, radial_distortion_coefficients = get_intrinsic_and_distortion_parameters(
        intrinsics_df, camera_name_to_use
    )

    # --- Get Timestamps ---
    image_timestamps = get_images_timestamp(cameras_directory, camera_name_to_use)
    annotation_timestamps = get_annotations_timestamp(annotations_df)

    # --- Process Annotations and Project Bounding Boxes ---
    for annotation_ts in annotation_timestamps:
        # Find the closest image timestamp for the current annotation timestamp
        closest_image_idx = np.argmin(np.abs(annotation_ts - image_timestamps))
        current_image_timestamp = image_timestamps[closest_image_idx]
        
        # Filter annotations for the current timestamp and valid point count
        current_annotations = annotations_df[annotations_df['timestamp_ns'] == annotation_ts]
        current_annotations = current_annotations[current_annotations['num_interior_pts'] > 0]
        
        print(f"\nProcessing timestamp: {annotation_ts} (Closest image: {current_image_timestamp})")
        
        # Get 3D bounding box corners and center in world coordinates
        box_corners_and_center_world = get_upright_3d_box_corners_with_center(current_annotations)

        projected_2d_bounding_boxes = []
        convex_hulls = []
        for i, box_3d_points_world in enumerate(box_corners_and_center_world):
            # Transform box points from World coordinates to Vehicle coordinates
            # Note: box_3d_points_world is Nx4 (x,y,z,1) where N=9 (8 corners + 1 center)
            box_points_vehicle = world_to_vehicle_transform @ box_3d_points_world.T
            
            # Extract the center point for FOV filtering (it's the last point)
            box_center_point = box_points_vehicle[:, -1]
            
            # Filter boxes outside the camera's field of view
            # Calculate azimuth of the center point
            # The original code's cam_fov_rad assumes 30 * pi / 90. If the actual FOV is 60 degrees, this needs adjustment.
            # and `cam_fov_rad` is actually `half_fov_rad`
            
            # Calculating azimuth to filter boxes by fov
            azimuth_of_center = np.arctan2(box_center_point[0], box_center_point[2])
            
            # If the calculated `cam_fov_rad` (30 * pi / 90) is intended to be the *half* FOV in radians
            # then the check should be against `cam_fov_rad`.
            # If `cam_fov_rad` is the *full* FOV, then divide by 2 for the check.
            # Let's assume the variable name `cam_fov_rad` in the original code implies the full FOV,
            # and the division by 2 is correct for the check.
            if abs(azimuth_of_center) > (camera_fov_radians / 2): # Using the variable name from setup
                continue

            # Remove the center point from the points before projection
            box_points_vehicle_no_center = box_points_vehicle[:, :-1]
            
            # Project 3D points (in vehicle frame) to 2D image plane using camera intrinsics
            # K @ P_vehicle_homogeneous (where P_vehicle_homogeneous are the 8 corners)
            projected_points_homogeneous = np.matmul(camera_intrinsic_matrix, box_points_vehicle_no_center)
            
            # Normalize by depth (divide x and y by z) to get pixel coordinates
            projected_points_2d = projected_points_homogeneous[:2] / projected_points_homogeneous[2]
            
            # Radial Distortion Correction (if enabled)
            # You can uncomment the line below to apply distortion correction
            # projected_points_2d = correct_radial_distortion(projected_points_2d, camera_intrinsic_matrix, radial_distortion_coefficients)
            
            # Calculate the 2D bounding box from the projected 8 corners
            x1, y1 = projected_points_2d.min(axis=1)
            x2, y2 = projected_points_2d.max(axis=1)

            # Append the 2D bounding box coordinates
            projected_2d_bounding_boxes.append([x1, y1, x2, y2])

            if convex_hulls_enabled:

                # Calculate the convex hull from the projected 8 corners
                hull_points = get_polygon_from_projected_points(projected_points_2d)

                # Append the convex hull coordinates
                convex_hulls.append(hull_points)

                if hulls_overlapping_enabled:
                    compute_overlapping(convex_hulls)
        
        # --- Render Results ---
        current_camera_image_path = os.path.join(cameras_directory, camera_name_to_use)
        render_projected(current_camera_image_path, 
                         current_image_timestamp, 
                         projected_2d_bounding_boxes)

        if convex_hulls_enabled:
            render_polygons(current_camera_image_path,
                            current_image_timestamp,
                            convex_hulls)
