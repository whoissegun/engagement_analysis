import cv2
import mediapipe as mp
import numpy as np
from FaceFeatureExtractor import FaceFeatureExtractor

# Initialize MediaPipe
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(max_num_faces=2,
                                  static_image_mode=False,
                                  refine_landmarks=True,
                                  min_detection_confidence=0.5,
                                  min_tracking_confidence=0.5)

feature_extractor = FaceFeatureExtractor()
# gaze_heatmap = GazeHeatmap(width=300, height=159)
cap = cv2.VideoCapture(0)
print("\n*\n*\n*\nStarting real-time engagement detection. Press 'q' to quit.\n*\n*\n*\n")
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Convert BGR to RGB for MediaPipe processing
    frame = cv2.flip(frame, 1)
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(frame_rgb)

    # heatmap_image = process_frame_with_heatmap(frame, gaze_heatmap)

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

            features = feature_extractor.extract_features(frame, face_landmarks)

             # Visualization with feature information
            cv2.putText(frame, f"Pitch (Up and Down Movement): {features.head_pitch:.2f}",
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            cv2.putText(frame, f"Yaw (Left and Right Movement): {features.head_yaw:.2f}",
                        (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            cv2.putText(frame, f"Roll (Tilt or Sideways Movement): {features.head_roll:.2f}",
                        (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            cv2.putText(frame, f"Iris Gaze X: {features.gaze_x:.2f}, Y: {features.gaze_y:.2f}",
                        (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            cv2.putText(frame, f"Gaze Variation X: {features.gaze_variation_x:.3f}, Y: {features.gaze_variation_y:.3f}",
                        (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            cv2.putText(frame, f"Eye Contact: {'Yes' if features.eye_contact_detected else 'No'}",
                        (10, 180), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            cv2.putText(frame, f"Blink: {'Yes' if features.is_blinking else 'No'}",
                        (10, 210), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            cv2.putText(frame, f"MAR (Mouth Aspect Ratio): {features.mar:.2f}",
                        (10, 240), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)
            
            # Check if a yawn was detected and display the alert
            if features.yawn_detected:
                text = "YAWNING DETECTED!"
                font = cv2.FONT_HERSHEY_SIMPLEX
                font_scale = 1
                thickness = 2
                color = (255, 0, 0)

                text_size = cv2.getTextSize(text, font, font_scale, thickness)[0]
                text_width = text_size[0]
                text_x = (frame.shape[1] - text_width) // 2  # Center text horizontally
                cv2.putText(frame, text, (text_x, 50), font, font_scale, color, thickness)

            # Display distraction warning
            if not features.is_focused:
                text = f"Distracted for {features.distraction_duration:.1f}s"
                font = cv2.FONT_HERSHEY_SIMPLEX
                font_scale = 0.8
                thickness = 2
                color = (0, 0, 255)  # Red

                # Calculate text size
                text_size = cv2.getTextSize(text, font, font_scale, thickness)[0]
                text_width = text_size[0]

                # Calculate bottom-center position
                text_x = (frame.shape[1] - text_width) // 2
                text_y = frame.shape[0] - 20  # Slightly above the bottom edge

                # Draw the text
                cv2.putText(frame, text, (text_x, text_y), font, font_scale, color, thickness)

    # Display the frame
    cv2.imshow("Lock'dIn Processor", frame)
    # cv2.imshow("Screen Gaze Heatmap", heatmap_image)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()


