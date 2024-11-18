import numpy as np

def calculate_mouth_aspect_ratio(landmarks, frame_width, frame_height):
    """
    Calculate the Mouth Aspect Ratio (MAR) using facial landmarks.

    Args:
        landmarks: MediaPipe facial landmarks (face_landmarks.landmark).
        frame_width: Width of the video frame.
        frame_height: Height of the video frame.

    Returns:
        mar: Mouth aspect ratio.
    """
    # Define landmark indices for mouth landmarks
    top_lip_indices = [13, 14]       # Upper lip
    bottom_lip_indices = [17, 18]    # Lower lip
    left_corner_index = 78           # Left corner of the mouth
    right_corner_index = 308         # Right corner of the mouth

    # Convert normalized landmarks to pixel coordinates
    def landmark_to_point(landmark):
        return np.array([landmark.x * frame_width, landmark.y * frame_height])

    # Get coordinates for top lip points
    top_lip = np.array([landmark_to_point(landmarks[i]) for i in top_lip_indices])

    # Get coordinates for bottom lip points
    bottom_lip = np.array([landmark_to_point(landmarks[i]) for i in bottom_lip_indices])

    # Get coordinates for mouth corners
    left_corner = landmark_to_point(landmarks[left_corner_index])
    right_corner = landmark_to_point(landmarks[right_corner_index])

    # Calculate vertical distances between top and bottom lip points
    vertical_distances = np.linalg.norm(top_lip - bottom_lip, axis=1)

    # Calculate horizontal distance between mouth corners
    horizontal_distance = np.linalg.norm(left_corner - right_corner)

    # Calculate MAR
    mar = np.mean(vertical_distances) / horizontal_distance

    return mar



# Helper function for Euclidean distance (used in fallback mode)
def distance(point1, point2):
    """Calculate squared Euclidean distance."""
    return ((point1[0] - point2[0]) ** 2 + (point1[1] - point2[1]) ** 2) ** 0.5
