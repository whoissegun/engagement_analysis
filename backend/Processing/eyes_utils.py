import numpy as np
import math
import time
from config import GAZE_THRESHOLD_X, GAZE_THRESHOLD_Y, DISTRACTION_TIME_LIMIT


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


def calculate_gaze_variation(gaze_positions_x, gaze_positions_y):
    """
    Calculates the variation in gaze over a buffer of gaze points.
    """
    if len(gaze_positions_x) > 1:
        return np.std(gaze_positions_x), np.std(gaze_positions_y)
    return 0.0, 0.0


def calculate_eye_contact(gaze_x, gaze_y, gaze_threshold_x=GAZE_THRESHOLD_X, gaze_threshold_y=GAZE_THRESHOLD_Y):
    """
    Determines whether the user is looking at the screen based on gaze coordinates.
    Allows a wider range for vertical gaze.
    """
    # Center of the screen is approximately (0.5, 0.5) in normalized coordinates
    is_within_x = abs(gaze_x - 0.5) < gaze_threshold_x
    is_within_y = abs(gaze_y - 0.5) < gaze_threshold_y
    return is_within_x and is_within_y



def calculate_blinking_ratio(face_landmarks, eye_points):
    """
    Calculates a blinking ratio using the eye width and height.
    """
    left_corner = (face_landmarks[eye_points[0]].x, face_landmarks[eye_points[0]].y)
    right_corner = (face_landmarks[eye_points[3]].x, face_landmarks[eye_points[3]].y)
    top = _middle_point(face_landmarks[eye_points[1]], face_landmarks[eye_points[2]])
    bottom = _middle_point(face_landmarks[eye_points[5]], face_landmarks[eye_points[4]])

    eye_width = math.hypot(left_corner[0] - right_corner[0], left_corner[1] - right_corner[1])
    eye_height = math.hypot(top[0] - bottom[0], top[1] - bottom[1])

    if eye_height == 0:  # Prevent division by zero
        return None
    return eye_width / eye_height


def _middle_point(p1, p2):
    """
    Returns the midpoint between two landmarks.
    """
    return ((p1.x + p2.x) / 2, (p1.y + p2.y) / 2)
 
# class EyeContactBuffer:
#     def __init__(self, distraction_time_limit=DISTRACTION_TIME_LIMIT):
#         self.eye_contact_start = None
#         self.distraction_start = None
#         self.distraction_time_limit = distraction_time_limit

#     def update_eye_contact(self, eye_contact_detected):
#         current_time = time.time()
#         if eye_contact_detected:
#             self.eye_contact_start = current_time
#             self.distraction_start = None  # Reset distraction
#             return True, 0  # Focused
#         else:
#             if self.distraction_start is None:
#                 self.distraction_start = current_time

#             distraction_duration = current_time - self.distraction_start
#             if distraction_duration > self.distraction_time_limit:
#                 return False, distraction_duration  # Distracted
#             return True, distraction_duration  # Still within limit

class EyeContactBuffer:
    def __init__(self):
        self.eye_contact_start_time = None
        self.eye_contact_duration = 0.0

    def update_eye_contact(self, eye_contact_detected):
        current_time = time.time()
        if eye_contact_detected:
            if self.eye_contact_start_time is None:
                self.eye_contact_start_time = current_time
            self.eye_contact_duration = current_time - self.eye_contact_start_time
        else:
            self.eye_contact_start_time = None
            self.eye_contact_duration = 0.0
        return eye_contact_detected, self.eye_contact_duration
