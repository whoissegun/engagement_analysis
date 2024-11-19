import cv2
import numpy as np
import cv2
import numpy as np

def calculate_head_pose(face_landmarks, frame):
    size = frame.shape
    # Use a realistic focal length based on webcam specs (assuming ~60° horizontal FoV)
    focal_length = size[1] / (2 * np.tan(np.pi / 6))  # 60 degrees = π/3, so half is π/6
    center = (size[1] / 2, size[0] / 2)
    camera_matrix = np.array(
        [[focal_length, 0, center[0]],
         [0, focal_length, center[1]],
         [0, 0, 1]], dtype="double"
    )

    # Assuming no lens distortion for now
    dist_coeffs = np.zeros((4, 1))

    # 3D model points for average face proportions
    model_points = np.array([
        (0.0, 0.0, 0.0),             # Nose tip
        (0.0, -330.0, -65.0),        # Chin
        (-225.0, 170.0, -135.0),     # Left eye left corner
        (225.0, 170.0, -135.0),      # Right eye right corner
        (-150.0, -150.0, -125.0),    # Left mouth corner
        (150.0, -150.0, -125.0)      # Right mouth corner
    ])

    # Get 2D points from landmarks
    image_points = np.array([
        get_point(face_landmarks, 1, frame),     # Nose tip
        get_point(face_landmarks, 152, frame),   # Chin
        get_point(face_landmarks, 33, frame),    # Left eye left corner
        get_point(face_landmarks, 263, frame),   # Right eye right corner
        get_point(face_landmarks, 61, frame),    # Left mouth corner
        get_point(face_landmarks, 291, frame)    # Right mouth corner
    ], dtype="double")

    # Solve PnP for pose estimation
    success, rotation_vector, translation_vector = cv2.solvePnP(
        model_points, image_points, camera_matrix, dist_coeffs,
        flags=cv2.SOLVEPNP_ITERATIVE
    )

    # Convert rotation vector to rotation matrix
    rotation_matrix, _ = cv2.Rodrigues(rotation_vector)

    # Combine rotation matrix and translation vector to form projection matrix
    pose_mat = cv2.hconcat((rotation_matrix, translation_vector))

    # Decompose the projection matrix to get Euler angles
    _, _, _, _, _, _, euler_angles = cv2.decomposeProjectionMatrix(pose_mat)
    pitch, yaw, roll = [angle.item() for angle in euler_angles.flatten()]

    # Normalize angles
    pitch = normalize_pitch(pitch)
    yaw = normalize_angle(yaw)
    roll = normalize_angle(roll)

    return pitch, yaw, roll

def normalize_angle(angle):
    """Normalize angle to [-180, 180] range."""
    while angle > 180:
        angle -= 360
    while angle < -180:
        angle += 360
    return angle

def normalize_pitch(pitch):
    """
    Normalize the pitch angle to be within the range of [-90, 90].

    Args:
        pitch (float): The raw pitch angle in degrees.

    Returns:
        float: The normalized pitch angle.
    """
    # Map the pitch angle to the range [-180, 180]
    if pitch > 180:
        pitch -= 360

    # Invert the pitch angle for intuitive up/down movement
    pitch = -pitch

    # Ensure that the pitch is within the range of [-90, 90]
    if pitch < -90:
        pitch = -(180 + pitch)
    elif pitch > 90:
        pitch = 180 - pitch
        
    pitch = -pitch

    return pitch

def get_point(face_landmarks, index, frame):
    """Convert MediaPipe normalized landmark to pixel coordinates."""
    h, w, _ = frame.shape
    landmark = face_landmarks.landmark[index]
    x = int(landmark.x * w)
    y = int(landmark.y * h)
    return (max(0, min(x, w - 1)), max(0, min(y, h - 1)))
