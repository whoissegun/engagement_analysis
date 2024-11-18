from flask import Blueprint, Response
from threading import Thread
from .camera import StreamingProcessor, generate_frames

main = Blueprint('main', __name__)

# Global variable for the StreamingProcessor instance
streaming_processor = None

@main.route('/video_feed')
def video_feed():
    global streaming_processor

    # Start StreamingProcessor if not already running
    if streaming_processor is None or not streaming_processor.processing:
        streaming_processor = StreamingProcessor()
        thread = Thread(target=streaming_processor.process_frames, daemon=True)
        thread.start()

    # Return the video stream
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')
