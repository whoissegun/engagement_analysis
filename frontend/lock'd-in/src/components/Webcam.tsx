// WebcamComponent.tsx
import React, { forwardRef } from "react";
import Webcam, { WebcamProps } from "react-webcam";

interface Props extends WebcamProps {
  audio: boolean;
  videoConstraints: {
    width: number;
    height: number;
  };
}

const WebcamComponent = forwardRef<Webcam, Props>((props, ref) => {
  return (
    <div>
      <Webcam {...props} ref={ref} />
    </div>
  );
});

export default WebcamComponent;
