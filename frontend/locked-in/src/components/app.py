from flask import Flask, request, jsonify
from flask_cors import CORS
import base64
import cv2
import numpy as np
from flask_cors import CORS

app = Flask(__name__)  # Initialize Flask app
CORS(app)    

@app.route('/routes', methods=['GET'])
def list_routes():
    routes = []
    for rule in app.url_map.iter_rules():
        routes.append({
            "endpoint": rule.endpoint,
            "methods": list(rule.methods),
            "route": str(rule)
        })
    return jsonify(routes), 200


# Variables to maintain the last prediction
last_predicted_class = None
last_confidence = None
processing_active = True

@app.route('/api/process-frame', methods=['POST'])
def process_frame():
    global last_predicted_class, last_confidence

    if not processing_active:
        return jsonify({"message": "Processing is currently stopped"}), 400

    try:
        # Validate the frame
        data = request.json
        if 'frame' not in data:
            return jsonify({"message": "Frame data not provided"}), 400

        # Decode the Base64-encoded frame
        image_data = data['frame'].split(",")[1]  # Extract the Base64 part
        image_bytes = base64.b64decode(image_data)
        nparr = np.frombuffer(image_bytes, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        # !!!replace this with actual model processing
        predicted_class = "Class_A"  # Example class
        confidence = 0.95  # Example confidence score

        # Save the prediction
        last_predicted_class = predicted_class
        last_confidence = confidence

        return jsonify({
            "predicted_class": predicted_class,
            "confidence": confidence
        }), 200

    except Exception as e:
        return jsonify({"message": str(e)}), 500


@app.route('/api/get-prediction', methods=['GET'])
def get_prediction():
    if last_predicted_class is None:
        return jsonify({"message": "No predictions made yet"}), 400

    return jsonify({
        "predicted_class": last_predicted_class,
        "confidence": last_confidence
    }), 200


@app.route('/api/status', methods=['GET'])
def get_status():
    return jsonify({
        "status": "ready" if processing_active else "stopped"
    }), 200


@app.route('/api/stop-processing', methods=['POST'])
def stop_processing():
    global processing_active
    processing_active = False
    return jsonify({"status": "stopped"}), 200


@app.route('/api/start-processing', methods=['POST'])
def start_processing():
    global processing_active
    processing_active = True
    return jsonify({"status": "ready"}), 200


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
