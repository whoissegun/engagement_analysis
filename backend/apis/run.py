from apis import create_app
from .camera import StreamingProcessor
import threading

def start_streaming():
    processor = StreamingProcessor()
    processing_thread = threading.Thread(target=processor.process_frames)
    processing_thread.daemon = True
    processing_thread.start()

if __name__ == '__main__':
    app = create_app()
    start_streaming()
    app.run(host='0.0.0.0', port=5000, debug=False)