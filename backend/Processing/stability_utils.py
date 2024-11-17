import time
import numpy as np

import numpy as np

def calculate_stability(current_landmarks, previous_landmarks, frame_width, frame_height):
    """
    Calculate the stability of facial landmarks between frames.

    Args:
        current_landmarks: MediaPipe landmarks for the current frame.
        previous_landmarks: MediaPipe landmarks for the previous frame.
        frame_width: Width of the video frame.
        frame_height: Height of the video frame.

    Returns:
        stability_score: A score between 0 and 1 indicating stability.
    """
    if previous_landmarks is None:
        # No previous landmarks to compare with
        return 1.0

    # Select key landmarks indices (e.g., nose tip, eyes, mouth corners)
    key_indices = [1, 33, 61, 199, 263, 291]  # Nose tip, left eye, left mouth corner, chin, right eye, right mouth corner

    # Calculate displacement for each key landmark
    displacements = []
    for idx in key_indices:
        curr_point = np.array([
            current_landmarks.landmark[idx].x * frame_width,
            current_landmarks.landmark[idx].y * frame_height
        ])
        prev_point = np.array([
            previous_landmarks.landmark[idx].x * frame_width,
            previous_landmarks.landmark[idx].y * frame_height
        ])
        displacement = np.linalg.norm(curr_point - prev_point)
        displacements.append(displacement)

    # Average displacement
    avg_displacement = np.mean(displacements)

    # Normalize the stability score (assuming max expected displacement is 20 pixels)
    MAX_DISPLACEMENT = 20.0
    stability_score = max(0.0, 1.0 - (avg_displacement / MAX_DISPLACEMENT))

    return stability_score


import time

def time_since_last_event(last_time):
    """
    Calculate the time since the last recorded event.

    Args:
        last_time: Timestamp of the last event (time.time()).

    Returns:
        elapsed_time: Time elapsed since the last event in seconds.
    """
    current_time = time.time()
    elapsed_time = current_time - last_time
    return elapsed_time
