import cv2
import mediapipe as mp
import numpy as np
import time
from dataclasses import dataclass
from typing import Dict, List, Tuple

@dataclass
class FaceFeatures:
    head_pitch: float
    head_yaw: float
    head_roll: float
    gaze_x: float
    gaze_y: float
    eye_contact_duration: float
    gaze_variation_x: float
    gaze_variation_y: float
    face_confidence: float
    landmarks_stability: float
    time_since_head_movement: float
    time_since_gaze_shift: float

class FaceFeatureExtractor:
    def __init__(self):
         # Initialize gaze smoothing variables
        self.last_lx = 0.0
        self.last_ly = 0.0
        self.last_rx = 0.0
        self.last_ry = 0.0
        # Initialize MediaPipe
        self.mp_face_mesh = mp.solutions.face_mesh
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            max_num_faces=1,
            static_image_mode=False,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )

        # Initialize tracking variables
        self.prev_landmarks = None
        self.gaze_history_x: List[float] = []
        self.gaze_history_y: List[float] = []
        self.last_head_movement_time = time.time()
        self.last_gaze_shift_time = time.time()
        self.eye_contact_start = None
        self.is_eye_contact = False
        self.last_positions = {
            'head': {'pitch': 0, 'yaw': 0, 'roll': 0},
            'gaze': {'x': 0, 'y': 0}
        }

        # Constants
        # self.MOVEMENT_THRESHOLD = 5.0  # degrees
        # self.GAZE_SHIFT_THRESHOLD = 0.2  # normalized units
        # self.HISTORY_WINDOW = 30  # frames
        self.MOVEMENT_THRESHOLD = 1.0  # Reduced from 5.0
        self.GAZE_SHIFT_THRESHOLD = 0.05  # Reduced from 0.2
        self.HISTORY_WINDOW = 15  # Reduced from 30

        self.EYE_CONTACT_THRESHOLD = 0.1  # threshold for considering eye contact
        self.EAR_THRESHOLD = 0.2  # threshold for eye aspect ratio (closed eyes)

        # MediaPipe indices for left and right eyes
        self.LEFT_EYE_INDICES = [362, 385, 387, 263, 373, 380]  # Upper and lower points
        self.RIGHT_EYE_INDICES = [33, 160, 158, 133, 153, 144]  # Upper and lower points

    def calculate_ear(self, landmarks, frame) -> float:
        """Calculate the average eye aspect ratio for both eyes."""
        def eye_aspect_ratio(eye_points):
            # Compute vertical distances
            v1 = np.linalg.norm(eye_points[1] - eye_points[5])
            v2 = np.linalg.norm(eye_points[2] - eye_points[4])
            # Compute horizontal distance
            h = np.linalg.norm(eye_points[0] - eye_points[3])
            # Calculate EAR
            return (v1 + v2) / (2.0 * h) if h > 0 else 0.0

        # Get coordinates for both eyes
        left_eye_points = np.array([self._get_landmark_coords(landmarks, idx, frame) 
                                  for idx in self.LEFT_EYE_INDICES])
        right_eye_points = np.array([self._get_landmark_coords(landmarks, idx, frame) 
                                   for idx in self.RIGHT_EYE_INDICES])

        # Calculate EAR for both eyes
        left_ear = eye_aspect_ratio(left_eye_points)
        right_ear = eye_aspect_ratio(right_eye_points)

        # Return average EAR
        return (left_ear + right_ear) / 2.0

    def is_making_eye_contact(self, gaze_x: float, gaze_y: float, ear: float) -> bool:
        """Determine if the person is making eye contact based on gaze direction and eye openness."""
        eyes_open = ear > self.EAR_THRESHOLD
        gaze_centered = (abs(gaze_x) < self.EYE_CONTACT_THRESHOLD and 
                        abs(gaze_y) < self.EYE_CONTACT_THRESHOLD)
        return eyes_open and gaze_centered
    
    # Optional: Add dynamic weight adjustment method
    def adjust_feature_sensitivity(self):
        """Dynamically adjust feature sensitivity to capture more nuanced changes."""
        # Reduce movement thresholds gradually
        self.MOVEMENT_THRESHOLD *= 0.9  # Gradually become more sensitive
        self.GAZE_SHIFT_THRESHOLD *= 0.9

    def extract_features(self, frame) -> FaceFeatures:
        """Extract all face features from a single frame."""
        # Convert to RGB for MediaPipe
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(frame_rgb)

        # Initialize features with default values
        features = FaceFeatures(
            head_pitch=0.0, head_yaw=0.0, head_roll=0.0,
            gaze_x=0.0, gaze_y=0.0, eye_contact_duration=0.0,
            gaze_variation_x=0.0, gaze_variation_y=0.0,
            face_confidence=0.0, landmarks_stability=0.0,
            time_since_head_movement=0.0, time_since_gaze_shift=0.0
        )
        

        if not results.multi_face_landmarks:
            self.is_eye_contact = False
            self.eye_contact_start = None
            return features

        # Draw face mesh
        self.mp_drawing.draw_landmarks(
            image=frame,
            landmark_list=results.multi_face_landmarks[0],
            connections=self.mp_face_mesh.FACEMESH_TESSELATION,
            landmark_drawing_spec=None,
            connection_drawing_spec=self.mp_drawing_styles.get_default_face_mesh_tesselation_style()
        )

        landmarks = results.multi_face_landmarks[0]
        
        # Calculate head pose
        pitch, yaw, roll = self.calculate_head_pose(landmarks, frame)
        
        # Calculate gaze
        gaze_x, gaze_y = self.calculate_gaze(landmarks, frame)

        # Calculate eye aspect ratio
        ear = self.calculate_ear(landmarks, frame)
        
        # Update eye contact tracking
        current_eye_contact = self.is_making_eye_contact(gaze_x, gaze_y, ear)
        
        if current_eye_contact and not self.is_eye_contact:
            # Started making eye contact
            self.eye_contact_start = time.time()
            self.is_eye_contact = True
        elif not current_eye_contact and self.is_eye_contact:
            # Stopped making eye contact
            self.eye_contact_start = None
            self.is_eye_contact = False

        eye_contact_duration = time.time() - self.eye_contact_start if self.eye_contact_start else 0.0
        
        # Update gaze history
        self.gaze_history_x.append(gaze_x)
        self.gaze_history_y.append(gaze_y)
        if len(self.gaze_history_x) > self.HISTORY_WINDOW:
            self.gaze_history_x.pop(0)
            self.gaze_history_y.pop(0)

        # Calculate gaze variation
        gaze_variation_x = np.std(self.gaze_history_x) if self.gaze_history_x else 0.0
        gaze_variation_y = np.std(self.gaze_history_y) if self.gaze_history_y else 0.0

        # Update movement timestamps
        if (abs(pitch - self.last_positions['head']['pitch']) > self.MOVEMENT_THRESHOLD or
            abs(yaw - self.last_positions['head']['yaw']) > self.MOVEMENT_THRESHOLD or
            abs(roll - self.last_positions['head']['roll']) > self.MOVEMENT_THRESHOLD):
            self.last_head_movement_time = time.time()

        if (abs(gaze_x - self.last_positions['gaze']['x']) > self.GAZE_SHIFT_THRESHOLD or
            abs(gaze_y - self.last_positions['gaze']['y']) > self.GAZE_SHIFT_THRESHOLD):
            self.last_gaze_shift_time = time.time()

        # Update last positions
        self.last_positions['head'].update({'pitch': pitch, 'yaw': yaw, 'roll': roll})
        self.last_positions['gaze'].update({'x': gaze_x, 'y': gaze_y})

        # Calculate stability
        stability = self.calculate_stability(landmarks, self.prev_landmarks, frame)
        self.prev_landmarks = landmarks

        # Draw gaze direction
        h, w, _ = frame.shape
        center_x, center_y = int(w/2), int(h/2)
        gaze_x_pixel = int(center_x + gaze_x * w/2)
        gaze_y_pixel = int(center_y + gaze_y * h/2)
        cv2.line(frame, (center_x, center_y), (gaze_x_pixel, gaze_y_pixel), (0, 255, 0), 2)
        
        # Package all features
        features = FaceFeatures(
            head_pitch=pitch,
            head_yaw=yaw,
            head_roll=roll,
            gaze_x=gaze_x,
            gaze_y=gaze_y,
            eye_contact_duration=eye_contact_duration,
            gaze_variation_x=gaze_variation_x,
            gaze_variation_y=gaze_variation_y,
            face_confidence=results.multi_face_landmarks[0].landmark[0].visibility 
                          if hasattr(results.multi_face_landmarks[0].landmark[0], 'visibility') else 1.0,
            landmarks_stability=stability,
            time_since_head_movement=time.time() - self.last_head_movement_time,
            time_since_gaze_shift=time.time() - self.last_gaze_shift_time
        )

        # Draw eye openness status
        cv2.putText(frame, f"EAR: {ear:.2f}", (10, 150), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(frame, f"Eyes {'Open' if ear > self.EAR_THRESHOLD else 'Closed'}", 
                    (10, 180), cv2.FONT_HERSHEY_SIMPLEX, 0.7, 
                    (0, 255, 0) if ear > self.EAR_THRESHOLD else (0, 0, 255), 2)
        
        print(f"Head Position: Pitch={pitch}, Yaw={yaw}, Roll={roll}")
        print(f"Gaze: x={gaze_x}, y={gaze_y}")
        print(f"Eye Contact: {current_eye_contact}")
        print(f"EAR: {ear}")

        return features

    # def calculate_head_pose(self, landmarks, frame) -> Tuple[float, float, float]:
    #     """Calculate head pose angles using facial landmarks."""
    #     size = frame.shape
    #     focal_length = size[1]
    #     center = (size[1]/2, size[0]/2)
    #     camera_matrix = np.array([
    #         [focal_length, 0, center[0]],
    #         [0, focal_length, center[1]],
    #         [0, 0, 1]
    #     ], dtype=np.float32)
        
    #     # Assuming no lens distortion
    #     dist_coeffs = np.zeros((4, 1))

    #     # 3D model points
    #     model_points = np.array([
    #         (0.0, 0.0, 0.0),          # Nose tip
    #         (0.0, -330.0, -65.0),     # Chin
    #         (-225.0, 170.0, -135.0),  # Left eye corner
    #         (225.0, 170.0, -135.0),   # Right eye corner
    #         (-150.0, -150.0, -125.0), # Left mouth corner
    #         (150.0, -150.0, -125.0)   # Right mouth corner
    #     ], dtype=np.float32)

    #     # 2D points
    #     image_points = np.array([
    #         self._get_landmark_coords(landmarks, 1, frame),    # Nose tip
    #         self._get_landmark_coords(landmarks, 152, frame),  # Chin
    #         self._get_landmark_coords(landmarks, 33, frame),   # Left eye corner
    #         self._get_landmark_coords(landmarks, 263, frame),  # Right eye corner
    #         self._get_landmark_coords(landmarks, 61, frame),   # Left mouth corner
    #         self._get_landmark_coords(landmarks, 291, frame)   # Right mouth corner
    #     ], dtype=np.float32)

    #     # Solve PnP
    #     success, rotation_vec, translation_vec = cv2.solvePnP(
    #         model_points, image_points, camera_matrix, dist_coeffs)

    #     # Convert rotation vector to Euler angles
    #     rotation_mat, _ = cv2.Rodrigues(rotation_vec)
    #     pose_mat = cv2.hconcat([rotation_mat, translation_vec])
    #     _, _, _, _, _, _, euler_angles = cv2.decomposeProjectionMatrix(pose_mat)
        
    #     pitch, yaw, roll = [float(angle) for angle in euler_angles.flatten()]
        
    #     # Normalize pitch
    #     if pitch > 90:
    #         pitch = pitch - 360
        
    #     pitch = max(-60, min(60, pitch))
    #     yaw = max(-45, min(45, yaw))
    #     roll = max(-30, min(30, roll))

    #     return pitch, yaw, roll

    def calculate_head_pose(self, landmarks, frame) -> Tuple[float, float, float]:
        """Calculate head pose angles using facial landmarks."""
        h, w, _ = frame.shape
        
        # Expanded set of 3D model points
        face_3d = np.array([
            [0.0, 0.0, 0.0],        # Nose tip
            [0.0, -330.0, -65.0],   # Chin
            [-225.0, 170.0, -135.0],# Left eye corner
            [225.0, 170.0, -135.0], # Right eye corner
            [-150.0, -150.0, -125.0],  # Left mouth corner
            [150.0, -150.0, -125.0]    # Right mouth corner
        ], dtype=np.float64)

        # Corresponding 2D points
        face_2d = []
        # Indices for nose tip, chin, left eye, right eye, left mouth, right mouth
        landmark_indices = [1, 152, 33, 263, 61, 291]
        for idx in landmark_indices:
            lm = landmarks.landmark[idx]
            x, y = int(lm.x * w), int(lm.y * h)
            face_2d.append([x, y])
        face_2d = np.array(face_2d, dtype=np.float64)

        # Camera matrix
        focal_length = 1 * w
        center = (w / 2, h / 2)
        camera_matrix = np.array([
            [focal_length, 0, center[0]],
            [0, focal_length, center[1]],
            [0, 0, 1]
        ], dtype=np.float64)

        # Solve PnP
        dist_coeffs = np.zeros((4, 1), dtype=np.float64)
        success, rotation_vec, translation_vec = cv2.solvePnP(
            face_3d, face_2d, camera_matrix, dist_coeffs)

        # Convert to rotation matrix and then to Euler angles
        rotation_mat, _ = cv2.Rodrigues(rotation_vec)
        pose_mat = cv2.hconcat([rotation_mat, translation_vec])
        _, _, _, _, _, _, euler_angles = cv2.decomposeProjectionMatrix(pose_mat)
        
        # Extract and convert angles
        pitch, yaw, roll = [float(angle) for angle in euler_angles.flatten()]

        # Normalize angles
        if pitch > 180:
            pitch -= 360
        
        # Constrain angles
        pitch = max(-60, min(60, pitch))
        yaw = max(-45, min(45, yaw))
        roll = max(-30, min(30, roll))

        return pitch, yaw, roll

    def calculate_gaze(self, landmarks, frame) -> Tuple[float, float]:
        """Calculate normalized gaze direction."""
        h, w, _ = frame.shape

        # Get eye landmark coordinates
        left_eye = np.mean([self._get_landmark_coords(landmarks, i, frame) 
                           for i in [33, 133]], axis=0)
        right_eye = np.mean([self._get_landmark_coords(landmarks, i, frame) 
                            for i in [362, 263]], axis=0)

        # Calculate gaze direction
        eyes_center = (left_eye + right_eye) / 2
        frame_center = np.array([w/2, h/2])
        gaze_vector = eyes_center - frame_center

        # Normalize gaze vector
        gaze_x = gaze_vector[0] / (w/2)  # Range: [-1, 1]
        gaze_y = gaze_vector[1] / (h/2)  # Range: [-1, 1]

        return gaze_x, gaze_y

    def calculate_stability(self, current_landmarks, previous_landmarks, frame) -> float:
        """Calculate stability score based on landmark movement."""
        if previous_landmarks is None:
            return 1.0

        h, w, _ = frame.shape
        key_points = [1, 33, 61, 199, 263, 291]  # Important facial landmarks
        
        displacements = []
        for idx in key_points:
            curr_point = self._get_landmark_coords(current_landmarks, idx, frame)
            prev_point = self._get_landmark_coords(previous_landmarks, idx, frame)
            displacement = np.linalg.norm(curr_point - prev_point)
            displacements.append(displacement)

        avg_displacement = np.mean(displacements)
        stability_score = max(0.0, 1.0 - (avg_displacement / 20.0))  # Normalize to [0,1]
        return stability_score

    def _get_landmark_coords(self, landmarks, idx: int, frame) -> np.ndarray:
        """Convert MediaPipe landmark to pixel coordinates."""
        h, w, _ = frame.shape
        landmark = landmarks.landmark[idx]
        return np.array([landmark.x * w, landmark.y * h])
    

def main():
    # Initialize camera and feature extractor
    cap = cv2.VideoCapture(0)
    extractor = FaceFeatureExtractor()

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Extract features
        features = extractor.extract_features(frame)

        # Display features on frame (for debugging)
        cv2.putText(frame, f"Pitch: {features.head_pitch:.1f}", (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(frame, f"Yaw: {features.head_yaw:.1f}", (10, 60), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(frame, f"Eye Contact: {features.eye_contact_duration:.1f}s", (10, 90), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(frame, f"Stability: {features.landmarks_stability:.2f}", (10, 120), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        # Show frame
        cv2.imshow('Face Features', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

# if __name__ == "__main__":
#     main()