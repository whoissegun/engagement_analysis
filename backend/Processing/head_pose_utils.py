import cv2
import numpy as np

def calculate_head_pose(face_landmarks, frame):
    size = frame.shape
    focal_length = size[1]
    center = (size[1] / 2, size[0] / 2)
    camera_matrix = np.array(
        [[focal_length, 0, center[0]],
         [0, focal_length, center[1]],
         [0, 0, 1]], dtype="double"
    )

    dist_coeffs = np.zeros((4, 1))  # Assuming no lens distortion

    # 3D model points
    model_points = np.array([
        (0.0, 0.0, 0.0),             # Nose tip
        (0.0, -330.0, -65.0),        # Chin
        (-225.0, 170.0, -135.0),     # Left eye left corner
        (225.0, 170.0, -135.0),      # Right eye right corner
        (-150.0, -150.0, -125.0),    # Left Mouth corner
        (150.0, -150.0, -125.0)      # Right mouth corner
    ])

    # 2D image points
    image_points = np.array([
        get_point(face_landmarks, 1, frame),     # Nose tip
        get_point(face_landmarks, 152, frame),   # Chin
        get_point(face_landmarks, 33, frame),    # Left eye left corner
        get_point(face_landmarks, 263, frame),   # Right eye right corner
        get_point(face_landmarks, 61, frame),    # Left mouth corner
        get_point(face_landmarks, 291, frame)    # Right mouth corner
    ], dtype="double")

    # print("Image Points:", image_points)

    # Solve PnP
    success, rotation_vector, translation_vector = cv2.solvePnP(
        model_points, image_points, camera_matrix, dist_coeffs
    )

    # Obtain rotation matrix
    rotation_matrix, jacobian = cv2.Rodrigues(rotation_vector)
    # print("Rotation Vector:", rotation_vector)
    # print("Rotation Matrix:", rotation_matrix)

    # Compute Euler angles
    pose_mat = cv2.hconcat((rotation_matrix, translation_vector))
    _, _, _, _, _, _, euler_angles = cv2.decomposeProjectionMatrix(pose_mat)

    pitch, yaw, roll = euler_angles.flatten()

    # Normalize pitch to [-90, 90]
    if pitch > 90:
        pitch = pitch - 360
    print(f"Pitch: {pitch}, Yaw: {yaw}, Roll: {roll}")
    return pitch, yaw, roll


def get_point(face_landmarks, index, frame):
    """
    Convert a normalized landmark to pixel coordinates.

    Args:
        face_landmarks: MediaPipe NormalizedLandmarkList.
        index: Index of the landmark.
        frame: Video frame (to get dimensions).

    Returns:
        (x, y): Tuple of pixel coordinates.
    """
    h, w, _ = frame.shape
    landmark = face_landmarks.landmark[index]
    x = int(landmark.x * w)
    y = int(landmark.y * h)
    # Clamp the coordinates to the frame bounds
    x = max(0, min(x, w - 1))
    y = max(0, min(y, h - 1))
    return (x, y)
