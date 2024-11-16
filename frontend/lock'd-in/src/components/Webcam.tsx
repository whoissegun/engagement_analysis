import React, { useRef } from "react";
import Webcam from "react-webcam";

interface Props {
  audio: boolean;
  ref: React.RefObject<Webcam>;
  videoConstraints: {};
}
const WebcamComponent = () => {
  const webcamRef = useRef<Webcam>(null);

  const videoConstraints = {
    width: 1280,
    height: 720,
  };

  return (
    <div>
      <Webcam
        audio={false}
        ref={webcamRef}
        videoConstraints={videoConstraints}
      />
    </div>
  );
};

export default WebcamComponent;
