import cv2
import threading
import queue
import time
from collections import deque
from backend.Processing.FaceFeatureExtractor import FaceFeatureExtractor
from backend.models.EngagementClassifierV1 import EngagementClassifierV1

# Global variables for frame sharing between threads
frame_queue = queue.Queue(maxsize=2)

class StreamingProcessor:
    def __init__(self, model_path='backend/models/model (1).pth'):
        self.extractor = FaceFeatureExtractor()
        self.predictor = EngagementPredictor(
            model_path=model_path,
            window_size=30,
            threshold=0.5
        )
        self.frame_times = deque(maxlen=30)
        self.frame_count = 0
        self.processing = True

    def process_frames(self):
        cap = cv2.VideoCapture(0)
        while self.processing:
            frame_start = time.time()
            ret, frame = cap.read()
            if not ret:
                break

            # Extract features and get predictions
            features = self.extractor.extract_features(frame)
            self.frame_count += 1
            if self.frame_count % 100 == 0:
                self.extractor.adjust_feature_sensitivity()

            prediction = self.predictor.predict(features)

            # Calculate FPS
            self.frame_times.append(time.time() - frame_start)
            fps = 1.0 / (sum(self.frame_times) / len(self.frame_times))

            # Draw on frame
            self._draw_overlay(frame, features, prediction, fps)

            # Update frame queue
            if frame_queue.full():
                try:
                    frame_queue.get_nowait()
                except queue.Empty:
                    pass
            frame_queue.put(frame)

        cap.release()

    def _draw_overlay(self, frame, features, prediction, fps):
        # Draw feature information
        cv2.putText(frame, f"Pitch: {features.head_pitch:.1f}", (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(frame, f"Yaw: {features.head_yaw:.1f}", (10, 60), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(frame, f"Eye Contact: {features.eye_contact_duration:.1f}s", (10, 90), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(frame, f"Stability: {features.landmarks_stability:.2f}", (10, 120), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        # Draw engagement prediction
        cv2.putText(frame, f"Engagement: {prediction['smoothed_prediction']}", 
                    (10, 210), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(frame, f"Confidence: {prediction['confidence']:.2f}", 
                    (10, 240), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(frame, f"FPS: {fps:.1f}", 
                    (10, 270), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        # Draw probability bars
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
                       f"{self.predictor.engagement_labels[i]}: {prob:.2f}", 
                       (410, 225 + i*30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 
                       0.6, 
                       (0, 255, 0), 
                       2)

def generate_frames():
    while True:
        if not frame_queue.empty():
            frame = frame_queue.get()
            ret, buffer = cv2.imencode('.jpg', frame)
            frame_bytes = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
        else:
            time.sleep(0.01)
