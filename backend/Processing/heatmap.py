import cv2
import numpy as np
import mediapipe as mp
import time
from eyes_utils import calculate_gaze_with_iris

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
    def __init__(self, width=300, height=169, decay_factor=0.95, scale_factor=1):
        self.width = width
        self.height = height
        self.scale_factor = scale_factor
        self.scaled_width = int(width * scale_factor)
        self.scaled_height = int(height * scale_factor)
        self.heatmap = np.zeros((self.scaled_height, self.scaled_width), dtype=np.float32)
        self.decay_factor = decay_factor
        self.screen_bounds = self.calculate_screen_bounds()
        
        # Debug counters
        self.min_x = float('inf')
        self.max_x = float('-inf')
        self.min_y = float('inf')
        self.max_y = float('-inf')

    def calculate_screen_bounds(self):
        # Calculate screen dimensions (make it proportional to window size)
        screen_width = int(self.scaled_width * 0.2)  # Increased from 0.15 to 0.6 for better visibility
        screen_height = int(screen_width * (9/16))  # Using 16:9 aspect ratio
        
        # Center the screen horizontally and vertically
        x1 = (self.scaled_width - screen_width) // 2
        y1 = (self.scaled_height - screen_height) // 2 +40
        x2 = x1 + screen_width
        y2 = y1 + screen_height
        
        return (x1, y1, x2, y2)

    def map_coordinates(self, gaze_x, gaze_y):
        """
        Map gaze coordinates to full window coordinates
        """
        # Update observed ranges (for debugging)
        self.min_x = min(self.min_x, gaze_x)
        self.max_x = max(self.max_x, gaze_x)
        self.min_y = min(self.min_y, gaze_y)
        self.max_y = max(self.max_y, gaze_y)
        
        # Map to full window coordinates
        window_x = int(gaze_x * self.scaled_width)
        window_y = int(gaze_y * self.scaled_height)

        return window_x, window_y

    def update(self, gaze_x, gaze_y):
        # Decay existing heatmap
        self.heatmap *= self.decay_factor

        # Map coordinates to window space
        x, y = self.map_coordinates(gaze_x, gaze_y)

        # Print debug info
        print(f"Raw gaze: ({gaze_x:.3f}, {gaze_y:.3f})")
        print(f"Mapped window pos: ({x}, {y})")
        print(f"Observed ranges - X: [{self.min_x:.3f}, {self.max_x:.3f}], Y: [{self.min_y:.3f}, {self.max_y:.3f}]")

        # Calculate heat kernel for entire window
        kernel_size = int(self.scaled_width * 0.05)  # Slightly larger kernel for smaller resolution
        x_range = np.arange(max(0, x-kernel_size), min(self.scaled_width, x+kernel_size+1))
        y_range = np.arange(max(0, y-kernel_size), min(self.scaled_height, y+kernel_size+1))
        xx, yy = np.meshgrid(x_range, y_range)
        
        # Gaussian spread
        gaussian = np.exp(-((xx - x)**2 + (yy - y)**2) / (2 * (kernel_size/3)**2))
        
        # Add heat regardless of screen bounds
        self.heatmap[yy, xx] += gaussian

    def get_heatmap_image(self):
        # Create black background
        heatmap_color = np.zeros((self.scaled_height, self.scaled_width, 3), dtype=np.uint8)
        
        # Normalize heatmap
        normalized = cv2.normalize(
            self.heatmap, 
            None, 
            alpha=0, 
            beta=255, 
            norm_type=cv2.NORM_MINMAX, 
            dtype=cv2.CV_8U
        )
        
        # Apply color map to entire heatmap
        heat_mask = normalized > 0
        colored_heat = cv2.applyColorMap(normalized, cv2.COLORMAP_JET)
        heatmap_color[heat_mask] = colored_heat[heat_mask]
        
        # Draw MacBook screen boundary
        x1, y1, x2, y2 = self.screen_bounds
        cv2.rectangle(heatmap_color, (x1, y1), (x2, y2), (255, 255, 255), 2)
        
        return heatmap_color

def main():
    cap = cv2.VideoCapture(0)
    
    base_width = 300
    base_height = 169
    scale_factor = 1
    
    gaze_heatmap = GazeHeatmap(
        width=base_width, 
        height=base_height, 
        scale_factor=scale_factor,
        decay_factor=0.90
    )
    
    cv2.namedWindow("Screen Gaze Heatmap", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Screen Gaze Heatmap", base_width, base_height)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.flip(frame, 1)
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(frame_rgb)

        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                gaze_x, gaze_y = calculate_gaze_with_iris(
                    face_landmarks.landmark, 
                    frame.shape[1], 
                    frame.shape[0]
                )
                gaze_heatmap.update(gaze_x, gaze_y)

        heatmap_image = gaze_heatmap.get_heatmap_image()
        cv2.imshow("Screen Gaze Heatmap", heatmap_image)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
