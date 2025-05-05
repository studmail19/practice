import os
import cv2
import tempfile
from ultralytics import YOLO
from flask import Flask, render_template, request, redirect, url_for, flash
import json

UPLOAD_FOLDER = 'uploads'
RESULT_FOLDER = 'static/results'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'mp4', 'avi'}

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESULT_FOLDER, exist_ok=True)

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.secret_key = 'INSERT_SECRET_KEY_HERE' # <--------------------------------Key

history = []


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def load_model():
    model = YOLO("yolo11n.pt")
    return model


model = load_model()


def draw_boxes(img, boxes):
    annotated_img = img.copy()
    box_color = (0, 255, 0)
    thickness = 2

    for box in boxes:
        xmin, ymin, xmax, ymax = map(int, box)
        cv2.rectangle(annotated_img, (xmin, ymin), (xmax, ymax), box_color, thickness)

    return annotated_img


def process_image(img_path):
    img = cv2.imread(img_path)
    if img is None:
        raise ValueError("Unable to load image for processing.")

    results = model(img)
    result_obj = results[0]
    if not hasattr(result_obj, 'boxes'):
        raise ValueError("The model result does not contain boxes information.")

    boxes_tensor = result_obj.boxes.xyxy
    boxes = boxes_tensor.cpu().numpy() if hasattr(boxes_tensor, 'cpu') else boxes_tensor

    count = len(boxes)
    annotated_img = draw_boxes(img, boxes)

    result_filename = os.path.basename(img_path)
    save_path = os.path.join(RESULT_FOLDER, result_filename)
    cv2.imwrite(save_path, annotated_img)

    return os.path.join('results', result_filename), count


def process_video(video_path):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError("Unable to read video file.")

    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps    = cap.get(cv2.CAP_PROP_FPS)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')

    base_name = os.path.splitext(os.path.basename(video_path))[0]
    result_filename = base_name + "_processed.mp4"
    save_path = os.path.join(RESULT_FOLDER, result_filename)
    out = cv2.VideoWriter(save_path, fourcc, fps, (width, height))

    frame_count = 0
    detection_total = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        results = model(frame)
        result_obj = results[0]
        if not hasattr(result_obj, 'boxes'):
            annotated_frame = frame.copy()
            curr_count = 0
        else:
            boxes_tensor = result_obj.boxes.xyxy
            boxes = boxes_tensor.cpu().numpy() if hasattr(boxes_tensor, 'cpu') else boxes_tensor
            curr_count = len(boxes)
            annotated_frame = draw_boxes(frame, boxes)

        detection_total += curr_count
        frame_count += 1

        out.write(annotated_frame)

    cap.release()
    out.release()

    average_detections = int(detection_total / frame_count) if frame_count > 0 else 0

    return os.path.join('results', result_filename), average_detections


@app.route("/", methods=["GET", "POST"])
def index():
    global history
    if request.method == "POST":
        if 'file' not in request.files:
            flash("No file part")
            return redirect(request.url)
        file = request.files['file']
        if file.filename == '':
            flash("No file selected")
            return redirect(request.url)
        if file and allowed_file(file.filename):
            filename = file.filename
            upload_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(upload_path)

            file_ext = filename.rsplit('.', 1)[1].lower()
            try:
                if file_ext in ['png', 'jpg', 'jpeg']:
                    result_path, count = process_image(upload_path)
                elif file_ext in ['mp4', 'avi']:
                    result_path, count = process_video(upload_path)
                else:
                    flash("Unsupported file type")
                    return redirect(request.url)
            except Exception as e:
                flash(f"Error during processing: {e}")
                return redirect(request.url)

            history.append({"filename": filename, "detections": count})
            with open("history.json", 'w') as json_file:
                json.dump(history, json_file, indent=4)
            return render_template("index.html", result_video=result_path if file_ext in ['mp4', 'avi'] else None,
                                   result_image=result_path if file_ext in ['png', 'jpg', 'jpeg'] else None,
                                   history=history)
        else:
            flash("File extension not allowed")
            return redirect(request.url)

    return render_template("index.html", history=history)


if __name__ == "__main__":
    app.run(debug=True)
