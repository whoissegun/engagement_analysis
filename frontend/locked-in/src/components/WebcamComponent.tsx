import React, { useState, useRef, useCallback, useEffect } from 'react';
import Webcam from 'react-webcam';
import { motion } from 'framer-motion';
import { Camera, CameraOff } from 'lucide-react';


const WebcamComponent = () => {
  const webcamRef = useRef<Webcam>(null);
  const [isEnabled, setIsEnabled] = useState(false);

  const handleToggleWebcam = useCallback(() => {
    setIsEnabled(!isEnabled);
  }, [isEnabled]);

  // Function to send frame to backend
  const sendFrameToBackend = useCallback(async () => {
    if (webcamRef.current) {
      const imageSrc = webcamRef.current.getScreenshot(); // Capture a single frame
      if (imageSrc) {
        try {
          const response = await fetch('http://127.0.0.1:5011/api/process-frame', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ frame: imageSrc }),
          });

          if (!response.ok) {
            console.error('Failed to process frame:', response.statusText);
            return;
          }

          const data = await response.json();
          setPrediction(data); // Update the prediction state with the response
          console.log('Prediction:', data);
        } catch (error) {
          console.error('Error sending frame to backend:', error);
        }
      } else {
        console.warn('No frame captured. Webcam might not be active.');
      }
    }
  }, []);

  useEffect(() => {
    let interval = null;

    if (isEnabled) {
      interval = setInterval(sendFrameToBackend, 1000); // Send frame every 1 second
    }

    return () => {
      if (interval) clearInterval(interval); // Cleanup on disable
    };
  }, [isEnabled, sendFrameToBackend]);

  return (
    <motion.div
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      className="max-w-4xl mx-auto px-4 py-8"
    >
      <div className={`relative rounded-2xl overflow-hidden bg-card-bg backdrop-blur-sm border border-card-border transition-all duration-300 ${isEnabled ? 'camera-glow' : ''}`}>
        {isEnabled ? (
          <Webcam
            ref={webcamRef}
            audio={false}
            screenshotFormat="image/jpeg"
            className="w-full aspect-video object-cover rounded-2xl"
          />
        ) : (
          <div className="w-full aspect-video bg-gradient-to-br from-primary/20 to-secondary/20 flex items-center justify-center">
            <p className="text-secondary">Camera disabled</p>
          </div>
        )}

        <motion.button
          whileHover={{ scale: 1.1 }}
          whileTap={{ scale: 0.9 }}
          onClick={handleToggleWebcam}
          className="absolute bottom-4 right-4 p-3 rounded-full bg-card-bg hover:bg-primary/20 backdrop-blur-sm transition-colors"
          aria-label={isEnabled ? 'Disable camera' : 'Enable camera'}
        >
          {isEnabled ? (
            <CameraOff className="w-6 h-6" />
          ) : (
            <Camera className="w-6 h-6" />
          )}
        </motion.button>
      </div>

      {/* Engagement Scale */}
      <div className="mt-6 p-4 rounded-xl bg-card-bg backdrop-blur-sm border border-card-border">
        <h3 className="text-lg font-semibold mb-3">Engagement Level</h3>
        <div className="h-2 bg-primary/10 rounded-full overflow-hidden">
          <motion.div
            className="h-full bg-gradient-to-r from-primary to-secondary"
            initial={{ width: '0%' }}
            animate={{ width: '70%' }}
            transition={{ duration: 1 }}
          />
        </div>
      </div>
    </motion.div>
  );
};

export default WebcamComponent;

function setPrediction(data: any) {
  throw new Error('Function not implemented.');
}
