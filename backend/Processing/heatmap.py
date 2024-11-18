import cv2
import numpy as np
import mediapipe as mp
import time

from backend.Processing.eyes_utils import calculate_gaze_with_iris
from backend.Processing.head_pose_utils import calculate_head_pose

# Initialize MediaPipe
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    max_num_faces=1,
    static_image_mode=False,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

class GazeHeatmap:
    def __init__(self, width=1920, height=1080, decay_factor=0.95):
        """
        Initialize heatmap with screen dimensions and decay
        
        Args:
            width (int): Width of the screen
            height (int): Height of the screen
            decay_factor (float): Rate at which previous points fade (0-1)
        """
        self.width = width
        self.height = height
        self.heatmap = np.zeros((height, width), dtype=np.float32)
        self.decay_factor = decay_factor

    def update(self, gaze_x, gaze_y):
        """
        Update heatmap with new gaze point
        
        Args:
            gaze_x (float): Normalized horizontal gaze coordinate (0-1)
            gaze_y (float): Normalized vertical gaze coordinate (0-1)
        """
        # Decay existing heatmap
        self.heatmap *= self.decay_factor

        # Convert normalized coordinates to pixel coordinates
        x = int(gaze_x * self.width)
        y = int(gaze_y * self.height)

        # Add heat at the gaze point with Gaussian distribution
        x_range = np.arange(max(0, x-30), min(self.width, x+31))
        y_range = np.arange(max(0, y-30), min(self.height, y+31))
        xx, yy = np.meshgrid(x_range, y_range)
        
        # 2D Gaussian kernel
        gaussian = np.exp(-((xx - x)**2 + (yy - y)**2) / (2 * 15**2))
        
        # Add gaussian to heatmap
        self.heatmap[yy, xx] += gaussian

    def get_heatmap_image(self):
        """
        Convert heatmap to a color-mapped image in BGR with black background
        
        Returns:
            numpy.ndarray: Color heatmap image in BGR
        """
        # Normalize heatmap
        normalized = cv2.normalize(
            self.heatmap, 
            None, 
            alpha=0, 
            beta=255, 
            norm_type=cv2.NORM_MINMAX, 
            dtype=cv2.CV_8U
        )
        
        # Apply color map
        heatmap_color = cv2.applyColorMap(normalized, cv2.COLORMAP_JET)
        
        # Set areas with zero heat to black
        heatmap_color[normalized == 0] = [0, 0, 0]
        
        return heatmap_color

def main():
    cap = cv2.VideoCapture(0)
    
    # # Set fixed capture resolution
    # cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    # cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    
    # # Use full screen dimensions for heatmap
    screen_width = 1920
    screen_height = 1080
    gaze_heatmap = GazeHeatmap(width=screen_width, height=screen_height)

    # # Resize the window to screen dimensions
    # cv2.namedWindow("Screen Gaze Heatmap", cv2.WINDOW_NORMAL)
    # cv2.resizeWindow("Screen Gaze Heatmap", screen_width, screen_height)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Convert BGR to RGB for MediaPipe
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(frame_rgb)

        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                # Calculate gaze (returns normalized coordinates)
                gaze_x, gaze_y = calculate_gaze_with_iris(
                    face_landmarks.landmark, 
                    frame.shape[1], 
                    frame.shape[0]
                )
                print(f"Gaze X: {gaze_x}, Gaze Y: {gaze_y}")

                # Update heatmap with scaled gaze coordinates
                gaze_heatmap.update(gaze_x, gaze_y)

        # Get colored heatmap
        heatmap_image = gaze_heatmap.get_heatmap_image()
        
        # Add text to indicate it's screen heatmap
        cv2.putText(heatmap_image, "Screen Gaze Heatmap", 
                    (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 
                    1, (255, 255, 255), 2)

        cv2.imshow("Screen Gaze Heatmap", heatmap_image)
        cv2.imshow("Webcam", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()

