import numpy as np
import math
import time
#from config import GAZE_THRESHOLD_X, GAZE_THRESHOLD_Y, DISTRACTION_TIME_LIMIT
import numpy as np
import math
import time

# Thresholds for various features
MAR_THRESHOLD = 0.6         # Threshold for yawning detection [study at a Glasgow Uni]

MOVEMENT_THRESHOLD = 7       # Threshold for significant head movement (degrees)

# Define eye points for blinking detection
LEFT_EYE_POINTS = [33, 160, 158, 133, 153, 144]
RIGHT_EYE_POINTS = [362, 385, 387, 263, 373, 380]
GAZE_THRESHOLD_X = 0.1 # Horizontal tolerance
GAZE_THRESHOLD_Y = 0.6  # Vertical tolerance (wider range for top/bottom)
HISTORY_WINDOW = 15
BLINK_RATIO_THRESHOLD = 4.5 

DISTRACTION_TIME_LIMIT = 4  # Threshold for displaying distraction message

def calculate_gaze_with_iris(landmarks, frame_width, frame_height, debug=False):
    """
    Calculate gaze direction based on iris position relative to eye boundary.
    
    Args:
        landmarks: MediaPipe face landmarks.
        frame_width: Width of the frame (for scaling normalized coordinates).
        frame_height: Height of the frame (for potential future scaling).
        debug: Enable detailed debugging output
    
    Returns:
        gaze_x, gaze_y: Normalized gaze coordinates (horizontal, vertical).
    """
    def print_landmark_details(index, name):
        landmark = landmarks[index]
        print(f"{name} Landmark - x: {landmark.x}, y: {landmark.y}, z: {landmark.z}")

    # Camera position compensation - adjusted for more natural range
    TOP_CAMERA_VERTICAL_BIAS = 0.5  # Increased to center the range

    # Landmark indices
    IRIS_LEFT = 468
    IRIS_RIGHT = 473
    EYE_LEFT_CORNER = 33
    EYE_RIGHT_CORNER = 133
    EYE_RIGHT_LEFT_CORNER = 362
    EYE_RIGHT_RIGHT_CORNER = 263
    
    # Additional vertical landmark indices for better vertical tracking
    LEFT_EYE_TOP = 159     # Top point of left eye
    LEFT_EYE_BOTTOM = 145  # Bottom point of left eye
    RIGHT_EYE_TOP = 386    # Top point of right eye
    RIGHT_EYE_BOTTOM = 374 # Bottom point of right eye

    if debug:
        print_landmark_details(IRIS_LEFT, "Left Iris")
        print_landmark_details(IRIS_RIGHT, "Right Iris")
        print_landmark_details(EYE_LEFT_CORNER, "Left Eye Left Corner")
        print_landmark_details(EYE_RIGHT_CORNER, "Left Eye Right Corner")

    # Get landmark coordinates
    left_iris_center = landmarks[IRIS_LEFT]
    right_iris_center = landmarks[IRIS_RIGHT]
    left_eye_left_corner = landmarks[EYE_LEFT_CORNER]
    left_eye_right_corner = landmarks[EYE_RIGHT_CORNER]
    right_eye_left_corner = landmarks[EYE_RIGHT_LEFT_CORNER]
    right_eye_right_corner = landmarks[EYE_RIGHT_RIGHT_CORNER]
    
    # Get vertical landmarks
    left_eye_top = landmarks[LEFT_EYE_TOP]
    left_eye_bottom = landmarks[LEFT_EYE_BOTTOM]
    right_eye_top = landmarks[RIGHT_EYE_TOP]
    right_eye_bottom = landmarks[RIGHT_EYE_BOTTOM]

    def safe_division(a, b, default=0.5):
        return a / b if b != 0 else default

    # Horizontal calculations remain the same
    left_eye_width = max(left_eye_right_corner.x - left_eye_left_corner.x, 0.01)
    right_eye_width = max(right_eye_right_corner.x - right_eye_left_corner.x, 0.01)
    
    # Improved vertical measurements using top and bottom points
    left_eye_height = max(left_eye_bottom.y - left_eye_top.y, 0.01)
    right_eye_height = max(right_eye_bottom.y - right_eye_top.y, 0.01)

    # Horizontal gaze calculation
    left_gaze_x = safe_division(left_iris_center.x - left_eye_left_corner.x, left_eye_width)
    right_gaze_x = safe_division(right_iris_center.x - right_eye_left_corner.x, right_eye_width)
    
    # Improved vertical gaze calculation
    left_gaze_y = safe_division(left_iris_center.y - left_eye_top.y, left_eye_height) - 0.5
    right_gaze_y = safe_division(right_iris_center.y - right_eye_top.y, right_eye_height) - 0.5

    if debug:
        print(f"Left Gaze - x: {left_gaze_x}, y: {left_gaze_y}")
        print(f"Right Gaze - x: {right_gaze_x}, y: {right_gaze_y}")

    # Average the gaze values
    gaze_x = (left_gaze_x + right_gaze_x) / 2
    gaze_y = (left_gaze_y + right_gaze_y) / 2

    # Apply camera bias and normalize to [-1, 1] range first
    gaze_y = gaze_y + TOP_CAMERA_VERTICAL_BIAS

    # Now normalize to [0, 1] range for final output
    gaze_x = max(0, min(gaze_x, 1))
    gaze_y = max(0, min((gaze_y + 1) / 2, 1))  # Convert from [-1, 1] to [0, 1]

    if debug:
        print(f"Final Gaze - x: {gaze_x}, y: {gaze_y}")

    return gaze_x, gaze_y


def calculate_gaze_variation(gaze_positions_x, gaze_positions_y):
    """
    Calculates the variation in gaze over a buffer of gaze points.
    """
    if len(gaze_positions_x) > 1:
        return np.std(gaze_positions_x), np.std(gaze_positions_y)
    return 0.0, 0.0


def calculate_eye_contact(
    gaze_x, 
    gaze_y, 
    gaze_threshold_x=GAZE_THRESHOLD_X, 
    gaze_threshold_y=GAZE_THRESHOLD_Y, 
    camera_placement='top_middle'
):
    """
    Enhanced eye contact detection with adaptive thresholds.
    
    Args:
        gaze_x (float): Normalized horizontal gaze coordinate
        gaze_y (float): Normalized vertical gaze coordinate
        gaze_threshold_x (float): Horizontal deviation threshold
        gaze_threshold_y (float): Vertical deviation threshold
        camera_placement (str): Camera position for contextual adjustment
    
    Returns:
        bool: Whether the gaze is considered within screen focus
    """
    # Adaptive center based on camera placement
    center_x = 0.5
    center_y = {
        'top_middle': 0.4,   # MacBook webcam typical position
        'center': 0.5,       # Centered camera
        'bottom': 0.6        # Lower camera placement
    }.get(camera_placement, 0.5)

    # Implement multi-stage detection with weighted zones
    x_deviation = abs(gaze_x - center_x)
    y_deviation = abs(gaze_y - center_y)

    # Potential for more nuanced thresholding
    x_in_range = x_deviation < gaze_threshold_x
    y_in_range = y_deviation < gaze_threshold_y

    # Optional: Add confidence scoring or logging
    if x_in_range and y_in_range:
        return True
    return False


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
 
class EyeContactBuffer:
    def __init__(self):
        self.eye_contact_start_time = None
        self.eye_contact_duration = 0.0
        self.distraction_start_time = None
        self.distraction_duration = 0.0

    def update_eye_contact(self, eye_contact_detected):
        """
        Updates the status of eye contact and tracks durations.

        Args:
            eye_contact_detected (bool): Whether eye contact is detected.

        Returns:
            Tuple[bool, float, float]: A tuple containing:
                - Whether eye contact is detected.
                - Eye contact duration (seconds).
                - Distraction duration (seconds).
        """
        current_time = time.time()

        if eye_contact_detected:
            # If eye contact is detected, start or continue tracking eye contact duration
            if self.eye_contact_start_time is None:
                self.eye_contact_start_time = current_time
            self.eye_contact_duration = current_time - self.eye_contact_start_time

            # Reset distraction tracking
            self.distraction_start_time = None
            self.distraction_duration = 0.0
        else:
            # If no eye contact is detected, start or continue tracking distraction duration
            if self.distraction_start_time is None:
                self.distraction_start_time = current_time
            self.distraction_duration = current_time - self.distraction_start_time

            # Reset eye contact tracking
            self.eye_contact_start_time = None
            self.eye_contact_duration = 0.0

        return eye_contact_detected, self.eye_contact_duration, self.distraction_duration

