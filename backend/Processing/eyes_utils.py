import numpy as np
import math
import time

# GAZE_THRESHOLD = 0.1  # Example threshold for detecting on-screen gaze
# BLINK_RATIO_THRESHOLD = 4.5  # Adjust this based on experimentation
# DISTRACTION_TIME_LIMIT = 4  # Threshold for displaying distraction message


# def calculate_gaze_with_iris(landmarks, frame_width, frame_height):
#     """
#     Calculate gaze direction based on iris position relative to eye boundary.
    
#     Args:
#         landmarks: MediaPipe face landmarks.
#         frame_width: Width of the frame (for scaling normalized coordinates).
#         frame_height: Height of the frame (for scaling normalized coordinates).
    
#     Returns:
#         gaze_x, gaze_y: Normalized gaze coordinates (horizontal, vertical).
#     """
#     # Get iris center positions
#     left_iris_center = landmarks[468]  # Left iris center
#     right_iris_center = landmarks[473]  # Right iris center

#     # Get eye boundary points
#     left_eye_left_corner = landmarks[33]
#     left_eye_right_corner = landmarks[133]
#     right_eye_left_corner = landmarks[362]
#     right_eye_right_corner = landmarks[263]

#     # Calculate horizontal and vertical gaze for left eye
#     left_gaze_x = (left_iris_center.x - left_eye_left_corner.x) / (left_eye_right_corner.x - left_eye_left_corner.x)
#     left_gaze_y = (left_iris_center.y - left_eye_left_corner.y) / (left_eye_right_corner.y - left_eye_left_corner.y)

#     # Calculate horizontal and vertical gaze for right eye
#     right_gaze_x = (right_iris_center.x - right_eye_left_corner.x) / (right_eye_right_corner.x - right_eye_left_corner.x)
#     right_gaze_y = (right_iris_center.y - right_eye_left_corner.y) / (right_eye_right_corner.y - right_eye_left_corner.y)

#     # Average the gaze values from both eyes for overall gaze
#     gaze_x = (left_gaze_x + right_gaze_x) / 2
#     gaze_y = (left_gaze_y + right_gaze_y) / 2

#     return gaze_x, gaze_y


# def calculate_gaze_variation(gaze_positions_x, gaze_positions_y):
#     """
#     Calculates the variation in gaze over a buffer of gaze points.
#     """
#     if len(gaze_positions_x) > 1:
#         return np.std(gaze_positions_x), np.std(gaze_positions_y)
#     return 0.0, 0.0


# def calculate_eye_contact(gaze_x, gaze_y, gaze_threshold=GAZE_THRESHOLD):
#     """
#     Determines whether the user is looking at the screen based on gaze coordinates.
#     """
#     if abs(gaze_x) < gaze_threshold and abs(gaze_y) < gaze_threshold:
#         return True
#     return False


# def calculate_blinking_ratio(face_landmarks, eye_points):
#     """
#     Calculates a blinking ratio using the eye width and height.
#     """
#     left_corner = (face_landmarks[eye_points[0]].x, face_landmarks[eye_points[0]].y)
#     right_corner = (face_landmarks[eye_points[3]].x, face_landmarks[eye_points[3]].y)
#     top = _middle_point(face_landmarks[eye_points[1]], face_landmarks[eye_points[2]])
#     bottom = _middle_point(face_landmarks[eye_points[5]], face_landmarks[eye_points[4]])

#     eye_width = math.hypot(left_corner[0] - right_corner[0], left_corner[1] - right_corner[1])
#     eye_height = math.hypot(top[0] - bottom[0], top[1] - bottom[1])

#     if eye_height == 0:  # Prevent division by zero
#         return None
#     return eye_width / eye_height


# def _middle_point(p1, p2):
#     """
#     Returns the midpoint between two landmarks.
#     """
#     return ((p1.x + p2.x) / 2, (p1.y + p2.y) / 2)


# class EyeContactBuffer:
#     """
#     Maintains a buffer to track eye contact duration and distractions.
#     """
#     def __init__(self, distraction_time_limit=DISTRACTION_TIME_LIMIT):
#         self.eye_contact_start = None
#         self.distraction_start = None
#         self.distraction_time_limit = distraction_time_limit

#     def update_eye_contact(self, eye_contact_detected):
#         """
#         Updates the buffer based on whether eye contact is detected.
#         """
#         current_time = time.time()
#         if eye_contact_detected:
#             self.eye_contact_start = current_time
#             self.distraction_start = None
#         else:
#             if self.distraction_start is None:
#                 self.distraction_start = current_time

#         if self.distraction_start and (current_time - self.distraction_start) > self.distraction_time_limit:
#             return False, current_time - self.distraction_start  # User is distracted
#         return True, 0  # User is focused
