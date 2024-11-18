import cv2
import mediapipe as mp
import time
import numpy as np
from head_pose_utils import calculate_head_pose
from stability_utils import calculate_stability
from mouth_utils import calculate_mouth_aspect_ratio  
from config import MAR_THRESHOLD, BLINK_RATIO_THRESHOLD, LEFT_EYE_POINTS, RIGHT_EYE_POINTS
from eyes_utils import (
    calculate_gaze_with_iris,
    calculate_eye_contact,
    calculate_gaze_variation,
    calculate_blinking_ratio,
    EyeContactBuffer
)

# Initialize MediaPipe
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(max_num_faces=2,
                                  static_image_mode=False,
                                  refine_landmarks=True,
                                  min_detection_confidence=0.5,
                                  min_tracking_confidence=0.5)

# Feature placeholders
prev_landmarks = None
gaze_positions_x = []  
gaze_positions_y = []  # Renamed for clarity with eyes_utils
eye_contact_start = None
last_head_movement_time = time.time()
last_gaze_shift_time = time.time()
last_pitch, last_yaw, last_roll = 0, 0, 0
last_gaze_x, last_gaze_y = 0, 0
yawn_detected_time = None

# Initialize eye contact buffer
eye_contact_buffer = EyeContactBuffer()


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
                if idx in [468, 473, 33, 133, 362, 263, 1, 152, 61, 291, 13, 14, 17, 18, 78, 308]:
                    cv2.circle(frame, (x, y), 5, (0, 140, 255), -1)  # Orange (BGR: 0, 140, 255)
                else:
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

            gaze_x, gaze_y = calculate_gaze_with_iris(face_landmarks.landmark, frame_width, frame_height, True)
            if abs(head_pitch - last_pitch) > 2 or abs(head_yaw - last_yaw) > 2 or abs(head_roll - last_roll) > 2:
                last_head_movement_time = time.time()  # Update on significant movement
            time_since_head_movement = time.time() - last_head_movement_time

            if abs(gaze_x - last_gaze_x) > 0.05 or abs(gaze_y - last_gaze_y) > 0.05:
                last_gaze_shift_time = time.time()  # Update on significant gaze shift
            time_since_gaze_shift = time.time() - last_gaze_shift_time

            stability = calculate_stability(face_landmarks, prev_landmarks, frame_width, frame_height)

            mar = calculate_mouth_aspect_ratio(face_landmarks.landmark, frame_width, frame_height)


            


            # Append gaze data for variation calculation
            gaze_positions_x.append(gaze_x)
            gaze_positions_y.append(gaze_y)
            if len(gaze_positions_x) > 100:
                gaze_positions_x.pop(0)
                gaze_positions_y.pop(0)
        
            gaze_var_x, gaze_var_y = calculate_gaze_variation(gaze_positions_x, gaze_positions_y)

            # Update previous metrics
            last_pitch, last_yaw, last_roll = head_pitch, head_yaw, head_roll
            last_gaze_x, last_gaze_y = gaze_x, gaze_y

            # Estimate face confidence
            # Use the visibility attribute of a key landmark (e.g., nose tip)
            face_confidence = face_landmarks.landmark[1].visibility

            # Calculate eye contact using gaze coordinates
            eye_contact_detected = calculate_eye_contact(gaze_x, gaze_y)
            is_focused, eye_contact_duration, distraction_duration = eye_contact_buffer.update_eye_contact(eye_contact_detected)
            # eye_contact_detected, eye_contact_duration = eye_contact_buffer.update_eye_contact(eye_contact_detected)
            # is_focused, distraction_duration = eye_contact_buffer.update_eye_contact(eye_contact_detected)
            # eye_contact_duration = eye_contact_buffer.eye_contact_duration

            # Calculate blink ratio
            left_blink_ratio = calculate_blinking_ratio(face_landmarks.landmark, LEFT_EYE_POINTS)
            right_blink_ratio = calculate_blinking_ratio(face_landmarks.landmark, RIGHT_EYE_POINTS)
            if left_blink_ratio and right_blink_ratio:
                blink_ratio = (left_blink_ratio + right_blink_ratio) / 2
                is_blinking = blink_ratio > BLINK_RATIO_THRESHOLD
            else:
                is_blinking = False

            # Yawning detection
            if mar > MAR_THRESHOLD:
                if yawn_detected_time is None:
                    yawn_detected_time = time.time()
            else:
                yawn_detected_time = None

            # Display yawning alert
            if yawn_detected_time is not None and time.time() - yawn_detected_time <=2:
                text = "YAWNING DETECTED!"
                font = cv2.FONT_HERSHEY_SIMPLEX
                font_scale = 1
                thickness = 2
                color = (255, 0, 0)
                
                text_size = cv2.getTextSize(text, font, font_scale, thickness)[0]
                text_width = text_size[0]
                text_x = (frame_width - text_width) // 2
                cv2.putText(frame, text, (text_x, 50), font, font_scale, color, thickness)

            # Enhanced visualization with iris-based gaze information
            cv2.putText(frame, f"Pitch (Up and Down Movement): {head_pitch:.2f}", 
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            cv2.putText(frame, f"Yaw (Left and Right Movement): {head_yaw:.2f}", 
                    (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            cv2.putText(frame, f"Roll (Tilt or Sideways Movement): {head_roll:.2f}", 
                    (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            cv2.putText(frame, f"Iris Gaze X: {gaze_x:.2f}, Y: {gaze_y:.2f}", 
                    (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            cv2.putText(frame, f"Gaze Variation X: {gaze_var_x:.3f}, Y: {gaze_var_y:.3f}", 
                    (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            cv2.putText(frame, f"Eye Contact: {'Yes' if eye_contact_detected else 'No'}", 
                    (10, 180), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            cv2.putText(frame, f"Blink: {'Yes' if is_blinking else 'No'}", 
                    (10, 210), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            cv2.putText(frame, f"MAR (Mouth Aspect Ratio): {mar:.2f}", 
                    (10, 240), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)

            # print("")
            # print(f"Head Pose: Pitch: {head_pitch:.2f}, Yaw: {head_yaw:.2f}, Roll: {head_roll:.2f}")
            # print(f"Iris Gaze: X: {gaze_x:.2f}, Y: {gaze_y:.2f}")
            # print(f"Gaze Variation: X: {gaze_var_x:.3f}, Y: {gaze_var_y:.3f}")
            # print(f"Eye Contact Duration: {eye_contact_duration:.2f}s")
            # print(f"Face Confidence: {face_confidence:.3f}")
            # print(f"Landmarks Stability: {stability:.3f}")
            # print(f"Time Since Head Movement: {time_since_head_movement:.1f}s")
            # print(f"Time Since Gaze Shift: {time_since_gaze_shift:.1f}s")

            # Display distraction warning
            if not is_focused:
                text = f"Distracted for {distraction_duration:.1f}s"
                font = cv2.FONT_HERSHEY_SIMPLEX
                font_scale = 0.8
                thickness = 2
                color = (0, 0, 255)  # Red

                # Calculate text size
                text_size = cv2.getTextSize(text, font, font_scale, thickness)[0]
                text_width = text_size[0]
                text_height = text_size[1]

                # Calculate bottom-center position
                text_x = (frame.shape[1] - text_width) // 2
                text_y = frame.shape[0] - 20  # Slightly above the bottom edge

                # Draw the text
                cv2.putText(frame, text, (text_x, text_y), font, font_scale, color, thickness)


            prev_landmarks = face_landmarks
    # Display the frame
    cv2.imshow("Lock'dIn Processor", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()


