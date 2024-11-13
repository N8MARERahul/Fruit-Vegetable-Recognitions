from flask import Flask, request, jsonify, Response, render_template
import cv2
from ultralytics import YOLO
import json

app = Flask(__name__)


try:
    model = YOLO("best.onnx", task="detect")

except Exception as e:
    print(e)

threshold = 0.5

cap = cv2.VideoCapture(0)

if not cap.isOpened():  # Error
    print("Could not open camera")
    exit()

# latest_detection = {
#     "label": "No detection",
#     "score": 0
# }

def generate_frames():
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Can't receive frame (stream end?). Exiting ...")
            break
        results = model(frame)[0]
        
        for result in results.boxes.data.tolist():
            x1, y1, x2, y2, score, class_id = result
            if score > threshold:
                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (255, 0, 0), 2)
                label = results.names[int(class_id)].upper()
                # latest_detection["label"] = label
                # latest_detection["score"] = round(score * 100)
                cv2.putText(frame, results.names[int(class_id)].upper(), (int(x1), int(y1 - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                    

        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()

        yield (b'--frame\r\n'
                b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

# @app.route('/current_detection')
# def current_detection():
#     # data = latest_detection
#     # latest_detection["label"] = "No detection"
#     # latest_detection["score"] = 0
#     return jsonify(latest_detection)

@app.route('/current_detection')
def current_detection():
    ret, frame = cap.read()
    if not ret:
        return jsonify({"label": "No detection", "score": 0})

    results = model(frame)[0]
    for result in results.boxes.data.tolist():
        x1, y1, x2, y2, score, class_id = result
        if score > threshold:
            label = results.names[int(class_id)].upper()
            return jsonify({"label": label, "score": round(score * 100)})
    
    return jsonify({"label": "No detection", "score": 0})

@app.route('/')
def index():
    return render_template('video2.html')

if __name__ == '__main__':
    app.run(debug=True, host = "0.0.0.0", port=5000)