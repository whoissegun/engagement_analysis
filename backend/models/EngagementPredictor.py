import torch
import numpy as np
from collections import deque
import time
from backend.models.EngagementClassifierV1 import EngagementClassifierV1

# from ..Processing.FaceFeatureExtractor import FaceFeatures, FaceFeatureExtractor
from backend.Processing.FaceFeatureExtractor import FaceFeatureExtractor, FaceFeatures
import cv2
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
        
        # Scale features if scaler was used during training
        if self.feature_scaler is not None:
            feature_vector = self.feature_scaler.transform(feature_vector)
            
        return torch.FloatTensor(feature_vector).to(self.device)
    
    def predict(self, features: FaceFeatures) -> dict:
        """
        Predict engagement level from face features.
        Returns prediction with confidence and smoothed prediction.
        """
        with torch.no_grad():
            # Preprocess features
            input_tensor = self._preprocess_features(features)
            
            # Get model prediction
            logits = self.model(input_tensor)
            probabilities = torch.softmax(logits, dim=1)
            prediction = torch.argmax(probabilities, dim=1).item()
            confidence = probabilities[0][prediction].item()
            
            # Add to history for temporal smoothing
            self.prediction_history.append(prediction)
            
            # Calculate smoothed prediction
            if len(self.prediction_history) >= self.window_size:
                counts = np.bincount(list(self.prediction_history))
                smoothed_prediction = np.argmax(counts)
            else:
                smoothed_prediction = prediction
                
            return {
                'raw_prediction': self.engagement_labels[prediction],
                'smoothed_prediction': self.engagement_labels[smoothed_prediction],
                'confidence': confidence,
                'probabilities': probabilities[0].cpu().numpy()
            }

def main():
    # Initialize components
    cap = cv2.VideoCapture(0)
    extractor = FaceFeatureExtractor()
    predictor = EngagementPredictor(
        model_path='backend/models/model (1).pth',
        window_size=30,
        threshold=0.5
    )
    
    # Performance monitoring
    frame_times = deque(maxlen=30)
    frame_count = 0
    while cap.isOpened():
        frame_start = time.time()
        
        
        # Capture and process frame
        ret, frame = cap.read()
        if not ret:
            break
            
        # Extract features
        features = extractor.extract_features(frame)
        frame_count += 1
        if frame_count % 100 == 0:
            extractor.adjust_feature_sensitivity()
        #print("features:: ", features)
        
        # Get prediction
        prediction = predictor.predict(features)
        #print("prediction:: ", prediction)
        
        # Calculate FPS
        frame_times.append(time.time() - frame_start)
        fps = 1.0 / (sum(frame_times) / len(frame_times))
        
        # Display results
        cv2.putText(frame, f"Engagement: {prediction['smoothed_prediction']}", 
                    (10, 210), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(frame, f"Confidence: {prediction['confidence']:.2f}", 
                    (10, 240), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(frame, f"FPS: {fps:.1f}", 
                    (10, 270), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        # Display probabilities as bar chart
        bar_width = 100
        bar_height = 20
        for i, prob in enumerate(prediction['probabilities']):
            width = int(prob * bar_width)
            cv2.rectangle(frame, 
                         (300, 210 + i*30), 
                         (300 + width, 230 + i*30), 
                         (0, 255, 0), 
                         -1)
            cv2.putText(frame, 
                       f"{predictor.engagement_labels[i]}: {prob:.2f}", 
                       (410, 225 + i*30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 
                       0.6, 
                       (0, 255, 0), 
                       2)
        
        # Show frame
        cv2.imshow('Engagement Classification', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()