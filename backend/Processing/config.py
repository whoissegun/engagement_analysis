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