import numpy as np

def calculate_gaze_with_iris(landmarks, frame_width, frame_height):
    """
    Calculate gaze direction based on iris position relative to eye boundary.
    
    Args:
        landmarks: MediaPipe face landmarks.
        frame_width: Width of the frame (for scaling normalized coordinates).
        frame_height: Height of the frame (for scaling normalized coordinates).
    
    Returns:
        gaze_x, gaze_y: Normalized gaze coordinates (horizontal, vertical).
    """
    # Get iris center positions
    left_iris_center = landmarks[468]  # Left iris center
    right_iris_center = landmarks[473]  # Right iris center

    # Get eye boundary points
    left_eye_left_corner = landmarks[33]
    left_eye_right_corner = landmarks[133]
    right_eye_left_corner = landmarks[362]
    right_eye_right_corner = landmarks[263]

    # Calculate horizontal and vertical gaze for left eye
    left_gaze_x = (left_iris_center.x - left_eye_left_corner.x) / (left_eye_right_corner.x - left_eye_left_corner.x)
    left_gaze_y = (left_iris_center.y - left_eye_left_corner.y) / (left_eye_right_corner.y - left_eye_left_corner.y)

    # Calculate horizontal and vertical gaze for right eye
    right_gaze_x = (right_iris_center.x - right_eye_left_corner.x) / (right_eye_right_corner.x - right_eye_left_corner.x)
    right_gaze_y = (right_iris_center.y - right_eye_left_corner.y) / (right_eye_right_corner.y - right_eye_left_corner.y)

    # Average the gaze values from both eyes for overall gaze
    gaze_x = (left_gaze_x + right_gaze_x) / 2
    gaze_y = (left_gaze_y + right_gaze_y) / 2

    return gaze_x, gaze_y

# def calculate_gaze(landmarks, frame):
#     """
#     Calculate gaze direction based on eye landmarks.

#     Args:
#         landmarks: MediaPipe facial landmarks.
#         frame: The video frame (used to get dimensions).

#     Returns:
#         gaze_x: Normalized horizontal gaze direction.
#         gaze_y: Normalized vertical gaze direction.
#     """
#     # Get frame dimensions
#     frame_height, frame_width, _ = frame.shape

#     # Left and right eye landmarks indices in MediaPipe Face Mesh
#     left_eye_indices = [33, 133]    # Approximate outer corners of the left eye
#     right_eye_indices = [362, 263]  # Approximate outer corners of the right eye

#     # Convert normalized landmarks to pixel coordinates
#     left_eye = [(int(landmarks.landmark[i].x * frame_width),
#                  int(landmarks.landmark[i].y * frame_height)) for i in left_eye_indices]
#     right_eye = [(int(landmarks.landmark[i].x * frame_width),
#                   int(landmarks.landmark[i].y * frame_height)) for i in right_eye_indices]

#     # Calculate the center points of each eye
#     left_eye_center = np.mean(left_eye, axis=0)
#     right_eye_center = np.mean(right_eye, axis=0)

#     # Calculate gaze direction as the normalized deviation from the center
#     eyes_center = (left_eye_center + right_eye_center) / 2.0
#     frame_center = np.array([frame_width / 2, frame_height / 2])

#     gaze_vector = eyes_center - frame_center
#     gaze_x = gaze_vector[0] / (frame_width / 2)  # Normalize to range [-1, 1]
#     gaze_y = gaze_vector[1] / (frame_height / 2)  # Normalize to range [-1, 1]

#     return gaze_x, gaze_y


def calculate_eye_contact(head_pitch, head_yaw, gaze_x, gaze_y, pitch_threshold=10, yaw_threshold=10, gaze_threshold=0.3):
    """
    Determine if the user is making eye contact based on head pose and gaze direction.

    Args:
        head_pitch: Pitch angle in degrees.
        head_yaw: Yaw angle in degrees.
        gaze_x: Normalized horizontal gaze direction.
        gaze_y: Normalized vertical gaze direction.
        pitch_threshold: Maximum allowed pitch deviation.
        yaw_threshold: Maximum allowed yaw deviation.
        gaze_threshold: Maximum allowed gaze deviation.

    Returns:
        eye_contact_detected: Boolean indicating eye contact.
    """
    # print(f"Head Pitch: {head_pitch}, Head Yaw: {head_yaw}, Gaze X: {gaze_x}, Gaze Y: {gaze_y}")
    # print(f"Thresholds - Pitch: {10}, Yaw: {10}, Gaze: {0.3}")
    if abs(head_pitch) < pitch_threshold and abs(head_yaw) < yaw_threshold and abs(gaze_x) < gaze_threshold and abs(gaze_y) < gaze_threshold:
        return True
    return False


def calculate_gaze_variation(gaze_x_values, gaze_y_values):
    """
    Calculate gaze variation based on recent gaze points.

    Args:
        gaze_x_values: List of recent horizontal gaze positions.
        gaze_y_values: List of recent vertical gaze positions.

    Returns:
        gaze_var_x, gaze_var_y: Standard deviation of gaze in horizontal and vertical directions.
    """
    if len(gaze_x_values) > 1:
        gaze_var_x = np.std(gaze_x_values)
        gaze_var_y = np.std(gaze_y_values)
    else:
        # If insufficient data, return 0 variation
        gaze_var_x = 0.0
        gaze_var_y = 0.0

    return gaze_var_x, gaze_var_y

