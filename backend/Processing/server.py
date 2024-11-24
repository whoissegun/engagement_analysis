from flask import Flask, Response
import cv2
import socket
import pickle
import numpy as np

app = Flask(__name__)

# Socket setup
s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
ip = "127.0.0.1"
port = 6666
s.bind((ip, port))

# Generator to stream video frames
def generate_frames():
    while True:
        try:
            # Receive data from client
            x = s.recvfrom(1000000)  # Use a larger buffer size for large frames
            data = x[0]  # Extract the data part

            # Deserialize and decode the frame
            compressed_frame = pickle.loads(data)
            frame = cv2.imdecode(compressed_frame, cv2.IMREAD_COLOR)

            # Encode the frame as JPEG
            _, buffer = cv2.imencode('.jpg', frame)
            frame_bytes = buffer.tobytes()

            # Yield frame as part of an HTTP response
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
        except Exception as e:
            print(f"Error receiving or processing frame: {e}")
            break

# Flask route to stream video
@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

# Flask route for the homepage (testing)
@app.route('/')
def index():
    return '''
    <html>
        <head>
            <title>Video Stream</title>
        </head>
        <body>
            <h1>Live Video Stream</h1>
            <img src="/video_feed" style="width: 100%; max-width: 640px; height: auto;" />
        </body>
    </html>
    '''

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5801, debug=False)
