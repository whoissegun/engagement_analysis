import cv2
import mediapipe as mp
import time
import numpy as np
from head_pose_utils import calculate_head_pose
from gaze_utils import calculate_gaze, calculate_eye_contact, calculate_gaze_variation
from stability_utils import calculate_stability
from mouth_utils import calculate_mouth_aspect_ratio  
from config import MAR_THRESHOLD, MOVEMENT_THRESHOLD, GAZE_SHIFT_THRESHOLD


# Initialize MediaPipe
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(max_num_faces=1,
                                  static_image_mode=False,
                                  refine_landmarks=True,
                                  min_detection_confidence=0.5,
                                  min_tracking_confidence=0.5)

# Feature placeholders
prev_landmarks = None
gaze_variation_x = []
gaze_variation_y = []
eye_contact_start = None
last_head_movement_time = time.time()
last_gaze_shift_time = time.time()
last_pitch, last_yaw, last_roll = 0, 0, 0
last_gaze_x, last_gaze_y = 0, 0
yawn_detected_time = None

# Start video capture
cap = cv2.VideoCapture(0)

print("\n*\n*\n*\nStarting real-time engagement detection. Press 'q' to quit.\n*\n*\n*\n")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Convert BGR to RGB for MediaPipe processing
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(frame_rgb)

    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            # Get frame dimensions
            frame_height, frame_width, _ = frame.shape

            # Draw facial landmarks
            for idx, landmark in enumerate(face_landmarks.landmark):
                x = int(landmark.x * frame_width)
                y = int(landmark.y * frame_height)
                # Draw a small circle for each landmark
                cv2.circle(frame, (x, y), 1, (0, 255, 255), -1)
            # Draw connections to form a mesh
            for connection in mp_face_mesh.FACEMESH_TESSELATION:
                start_idx = connection[0]
                end_idx = connection[1]
                start_landmark = face_landmarks.landmark[start_idx]
                end_landmark = face_landmarks.landmark[end_idx]

                # Convert normalized coordinates to pixel coordinates
                x_start = int(start_landmark.x * frame_width)
                y_start = int(start_landmark.y * frame_height)
                x_end = int(end_landmark.x * frame_width)
                y_end = int(end_landmark.y * frame_height)

                # Draw the connection
                cv2.line(frame, (x_start, y_start), (x_end, y_end), (255, 0, 0), 1)
            # Draw iris landmarks separately
            for connection in mp_face_mesh.FACEMESH_IRISES:
                start_idx = connection[0]
                end_idx = connection[1]
                start_landmark = face_landmarks.landmark[start_idx]
                end_landmark = face_landmarks.landmark[end_idx]

                x_start = int(start_landmark.x * frame_width)
                y_start = int(start_landmark.y * frame_height)
                x_end = int(end_landmark.x * frame_width)
                y_end = int(end_landmark.y * frame_height)

                cv2.line(frame, (x_start, y_start), (x_end, y_end), (0, 255, 0), 1)  # Green lines for iris

            # Extract features
            head_pitch, head_yaw, head_roll = calculate_head_pose(face_landmarks, frame)
            stability = calculate_stability(face_landmarks, prev_landmarks, frame_width, frame_height)
            gaze_x, gaze_y = calculate_gaze(face_landmarks, frame)




            # Append gaze data for variation calculation
            gaze_variation_x.append(gaze_x)
            gaze_variation_y.append(gaze_y)
            if len(gaze_variation_x) > 100:
                gaze_variation_x.pop(0)
                gaze_variation_y.pop(0)
        
            if len(gaze_variation_x) > 1:
                gaze_var_x, gaze_var_y = calculate_gaze_variation(gaze_variation_x, gaze_variation_y)
            else:
                gaze_var_x, gaze_var_y = 0.0, 0.0  # Default values when not enough data

            eye_contact_detected = calculate_eye_contact(head_pitch, head_yaw, gaze_x, gaze_y,
                                                        pitch_threshold=180, yaw_threshold=15, gaze_threshold=0.3)
            gaze_var_x, gaze_var_y = calculate_gaze_variation(gaze_variation_x, gaze_variation_y)
            mar = calculate_mouth_aspect_ratio(face_landmarks.landmark, frame_width, frame_height)

            # Eye contact duration
            print(eye_contact_detected)
            if eye_contact_detected:
                if eye_contact_start is None:
                    eye_contact_start = time.time()
                eye_contact_duration = time.time() - eye_contact_start
            else:
                eye_contact_start = None
                eye_contact_duration = 0

            if abs(head_pitch - last_pitch) > MOVEMENT_THRESHOLD or \
            abs(head_yaw - last_yaw) > MOVEMENT_THRESHOLD or \
            abs(head_roll - last_roll) > MOVEMENT_THRESHOLD:
                last_head_movement_time = time.time()  # Update time for major movement
            time_since_head_movement = time.time() - last_head_movement_time  # Calculate time elapsed
            last_pitch, last_yaw, last_roll = head_pitch, head_yaw, head_roll  # Update last head position

            # Time since last major gaze shift
            if abs(gaze_x - last_gaze_x) > GAZE_SHIFT_THRESHOLD or \
            abs(gaze_y - last_gaze_y) > GAZE_SHIFT_THRESHOLD:
                last_gaze_shift_time = time.time()  # Update time for major gaze shift
            time_since_gaze_shift = time.time() - last_gaze_shift_time  # Calculate time elapsed
            last_gaze_x, last_gaze_y = gaze_x, gaze_y

            # Yawning detection (Mouth Aspect Ratio)
            if mar > MAR_THRESHOLD:
                if yawn_detected_time is None:
                    yawn_detected_time = time.time()
            else:
                yawn_detected_time = None

            if yawn_detected_time is not None and time.time() - yawn_detected_time < 3:
                # Calculate text position
                text = "YAWNING DETECTED!"
                font = cv2.FONT_HERSHEY_SIMPLEX
                font_scale = 1
                thickness = 2
                color = (255, 0, 0)
                
                text_size = cv2.getTextSize(text, font, font_scale, thickness)[0]
                text_width = text_size[0]
                frame_width = frame.shape[1]
                text_x = (frame_width - text_width) // 2
                text_y = 50  # Position near the top

                # Draw text on frame
                cv2.putText(frame, text, (text_x, text_y), font, font_scale, color, thickness)

            # Display extracted features
            cv2.putText(frame, f"Pitch: {head_pitch:.2f}, Yaw: {head_yaw:.2f}, Roll: {head_roll:.2f}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            cv2.putText(frame, f"Gaze X: {gaze_x:.2f}, Gaze Y: {gaze_y:.2f}", (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            cv2.putText(frame, f"Eye Contact Duration: {eye_contact_duration:.2f}", (10, 90),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            cv2.putText(frame, f"MAR: {mar:.2f}", (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)

    # Display the frame
    cv2.imshow("Lock'dIn Processor", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()


