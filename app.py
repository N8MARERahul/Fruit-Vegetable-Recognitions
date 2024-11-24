from flask import Flask, request, jsonify, Response, render_template
import cv2
from ultralytics import YOLO
import pandas as pd
import json

app = Flask(__name__)


try:
    model = YOLO("best.onnx", task="detect")
    nutrition_data = pd.read_csv('vegetable_fruit_nutrition.csv')

except Exception as e:
    print(e)
    exit()

threshold = 0.5

cap = cv2.VideoCapture(2)

if not cap.isOpened():  # Error
    print("Could not open camera")
    exit()

# latest_detection = {
#     "label": "No detection",
#     "score": 0
# }
current_detection = {"label": "No detection", "score": 0, "nutrition": {}}

def generate_frames():
    global current_detection
    detection_found = False
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Can't receive frame (stream end?). Exiting ...")
            break

        results = model(frame)[0]
        
        for result in results.boxes.data.tolist():
            x1, y1, x2, y2, score, class_id = result
            if score > threshold:
                label = results.names[int(class_id)].upper()
                nutrition = nutrition_data.loc[nutrition_data['Item'].str.upper() == label]
                nutrition_info = nutrition.iloc[0].to_dict() if not nutrition.empty else {"nutrition": "No data available"}
                
                # Update shared detection result
                current_detection = {
                    "label": str(label),
                    "score": round(score * 100),
                    "nutrition": nutrition_info
                }
                print(nutrition_info)
                detection_found = True
                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (255, 0, 0), 2)
                cv2.putText(frame, results.names[int(class_id)].upper(), (int(x1), int(y1 - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            else:
                detection_found = False

        if not detection_found:
            current_detection = {"label": "No detection", "score": 0, "nutrition": {}}
                    
        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()

        yield (b'--frame\r\n'
                b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/current_detection')
def current_detection_route():

    global current_detection

    current_detection_json = json.dumps(current_detection)
    return jsonify(current_detection_json)

@app.route("/")
def welcome():
    return render_template("welcome.html")

@app.route('/detection')
def detection():
    return render_template('video.html')

if __name__ == '__main__':
    app.run(debug=True, host = "0.0.0.0", port=5000)