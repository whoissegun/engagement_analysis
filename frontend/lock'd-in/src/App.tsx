// App.tsx
import React from "react";
import WebcamComponent from "./components/Webcam";
import "./index.css";

function App() {
  const webcamProps = {
    audio: false,
    videoConstraints: {
      width: 1280,
      height: 720,
    },
  };

  return (
    <div>
      <h1>Lock'd In</h1>
      <WebcamComponent {...webcamProps} />
    </div>
  );
}

export default App;
