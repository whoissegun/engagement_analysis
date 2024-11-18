import torch
import numpy as np
from collections import deque
import time
from backend.models.EngagementClassifierV1 import EngagementClassifierV1
import mediapipe as mp
from backend.Processing.FaceFeatureExtractor import FaceFeatureExtractor, FaceFeatures
import cv2

# Initialize MediaPipe
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    max_num_faces=2,
    static_image_mode=False,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

class EngagementPredictor:
    def __init__(self, model_path: str, window_size: int = 30, threshold: float = 0.5):
        """
        Initialize the engagement predictor with a pre-trained model.
        
        Args:
            model_path: Path to the saved model weights
            window_size: Number of frames to consider for temporal smoothing
            threshold: Confidence threshold for predictions
        """
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = EngagementClassifierV1().to(self.device)
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.eval()
        
        self.window_size = window_size
        self.threshold = threshold
        self.prediction_history = deque(maxlen=window_size)
        self.feature_scaler = None  # Add your fitted StandardScaler here if used during training
        
        # Labels for engagement levels
        self.engagement_labels = ['Disengaged', 'Partially Engaged', 'Fully Engaged']
        
    def _preprocess_features(self, features: FaceFeatures) -> torch.Tensor:
        """Convert FaceFeatures to model input tensor."""
        feature_vector = np.array([
            features.head_pitch,
            features.head_yaw,
            features.head_roll,
            features.gaze_x,
            features.gaze_y,
            features.eye_contact_duration,
            features.gaze_variation_x,
            features.gaze_variation_y,
            features.face_confidence,
            features.landmarks_stability,
            features.time_since_head_movement,
            features.time_since_gaze_shift
        ]).reshape(1, -1)
        
        if self.feature_scaler is not None:
            feature_vector = self.feature_scaler.transform(feature_vector)
            
        return torch.FloatTensor(feature_vector).to(self.device)
    
    def predict(self, features: FaceFeatures) -> dict:
        """
        Predict engagement level from face features.
        Returns prediction with confidence and smoothed prediction.
        """
        with torch.no_grad():
            input_tensor = self._preprocess_features(features)
            logits = self.model(input_tensor)
            probabilities = torch.softmax(logits, dim=1)
            prediction = torch.argmax(probabilities, dim=1).item()
            confidence = probabilities[0][prediction].item()
            
            self.prediction_history.append(prediction)
            
            if len(self.prediction_history) >= self.window_size:
                counts = np.bincount(list(self.prediction_history))
                smoothed_prediction = np.argmax(counts)
            else:
                smoothed_prediction = prediction
                
            return {
                'raw_prediction': self.engagement_labels[prediction],
                'smoothed_prediction': self.engagement_labels[smoothed_prediction],
                'confidence': confidence,
                'probabilities': probabilities[0].cpu().numpy().tolist()
            }

def main():
    # Initialize components
    cap = cv2.VideoCapture(0)
    feature_extractor = FaceFeatureExtractor()
    predictor = EngagementPredictor(
        model_path='backend/models/best_model_v3.pth',
        window_size=30,
        threshold=0.5
    )

    print("\n*\n*\n*\nStarting real-time engagement detection. Press 'q' to quit.\n*\n*\n*\n")
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(frame_rgb)

        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                frame_height, frame_width, _ = frame.shape

                # Draw facial landmarks
                for idx, landmark in enumerate(face_landmarks.landmark):
                    x = int(landmark.x * frame_width)
                    y = int(landmark.y * frame_height)
                    if idx in [468, 473, 33, 133, 362, 263, 1, 152, 61, 291, 13, 14, 17, 18, 78, 308]:
                        cv2.circle(frame, (x, y), 5, (0, 140, 255), -1)
                    else:
                        cv2.circle(frame, (x, y), 1, (0, 255, 255), -1)

                # Draw mesh connections
                for connection in mp_face_mesh.FACEMESH_TESSELATION:
                    start_idx = connection[0]
                    end_idx = connection[1]
                    start_landmark = face_landmarks.landmark[start_idx]
                    end_landmark = face_landmarks.landmark[end_idx]

                    x_start = int(start_landmark.x * frame_width)
                    y_start = int(start_landmark.y * frame_height)
                    x_end = int(end_landmark.x * frame_width)
                    y_end = int(end_landmark.y * frame_height)

                    cv2.line(frame, (x_start, y_start), (x_end, y_end), (255, 0, 0), 1)

                # Draw iris landmarks
                for connection in mp_face_mesh.FACEMESH_IRISES:
                    start_idx = connection[0]
                    end_idx = connection[1]
                    start_landmark = face_landmarks.landmark[start_idx]
                    end_landmark = face_landmarks.landmark[end_idx]

                    x_start = int(start_landmark.x * frame_width)
                    y_start = int(start_landmark.y * frame_height)
                    x_end = int(end_landmark.x * frame_width)
                    y_end = int(end_landmark.y * frame_height)

                    cv2.line(frame, (x_start, y_start), (x_end, y_end), (0, 255, 0), 1)

                # Extract features and get prediction
                features = feature_extractor.extract_features(frame, face_landmarks)
                pred_result = predictor.predict(features=features)

                # Display feature information
                info_text = [
                    f"Pitch: {features.head_pitch:.2f}",
                    f"Yaw: {features.head_yaw:.2f}",
                    f"Roll: {features.head_roll:.2f}",
                    f"Gaze X: {features.gaze_x:.2f}, Y: {features.gaze_y:.2f}",
                    f"Gaze Var X: {features.gaze_variation_x:.3f}, Y: {features.gaze_variation_y:.3f}",
                    f"Eye Contact: {'Yes' if features.eye_contact_detected else 'No'}",
                    f"Blink: {'Yes' if features.is_blinking else 'No'}",
                    f"MAR: {features.mar:.2f}",
                    f"Prediction: {pred_result['smoothed_prediction']}"  # Fixed f-string syntax
                ]

                for i, text in enumerate(info_text):
                    y_pos = 30 + (i * 30)
                    cv2.putText(frame, text, (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

                # Display warnings
                if features.yawn_detected:
                    text = "YAWNING DETECTED!"
                    text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 1, 2)[0]
                    text_x = (frame.shape[1] - text_size[0]) // 2
                    cv2.putText(frame, text, (text_x, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

                if not features.is_focused:
                    text = f"Distracted for {features.distraction_duration:.1f}s"
                    text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)[0]
                    text_x = (frame.shape[1] - text_size[0]) // 2
                    text_y = frame.shape[0] - 20
                    cv2.putText(frame, text, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

        cv2.imshow("Lock'dIn Processor", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()