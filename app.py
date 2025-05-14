from flask import Flask, request, render_template
from ultralytics import YOLO
from PIL import Image, ImageDraw, ImageFont
import io
import base64
from collections import Counter

app = Flask(__name__)

# Load the YOLO model
model_path = "C:\\Users\\akhil\\OneDrive\\Documents\\flask_projects\\yolo_black_grains_flask_app\\wheat_best.pt"
model = YOLO(model_path)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files or request.files['file'].filename == '':
        return render_template('result.html', error="No file uploaded or selected", class_counts=None, result_image=None, class_mapping=None)

    file = request.files['file']

    # Read image from memory
    image = Image.open(io.BytesIO(file.read()))

    # Run prediction
    results = model(image)

    # Draw results on the image
    draw = ImageDraw.Draw(image)

    try:
        font = ImageFont.truetype("arial.ttf", size=40)
    except IOError:
        font = ImageFont.load_default()

    class_ids = []  # Collect class IDs
    for result in results:
        for box in result.boxes:
            x_min, y_min, x_max, y_max = box.xyxy[0]
            x_min, y_min, x_max, y_max = float(x_min), float(y_min), float(x_max), float(y_max)

            # Draw bounding boxes and record class IDs
            draw.rectangle([(x_min, y_min), (x_max, y_max)], outline="red", width=3)
            class_id = int(box.cls.item())
            class_ids.append(class_id)

            # Add text label
            text = f"ID: {class_id}"
            text_bbox = draw.textbbox((x_min, y_min - 20), text, font=font)
            text_width = text_bbox[2] - text_bbox[0]
            text_height = text_bbox[3] - text_bbox[1]
            draw.rectangle([(x_min, y_min - text_height), (x_min + text_width, y_min)], fill="red")
            draw.text((x_min, y_min - text_height), text, font=font, fill="white")

    # Count occurrences of each class ID
    class_counts = dict(Counter(class_ids))

    # Map class IDs to names
    class_mapping = {
        0: "broken",
        1: "damage",
        2: "fm_inorganic",
        3: "fm_organic",
        4: "healthy",
        5: "immature",
        6: "shrivelled",
        7: "weevilled"
    }

    # Convert image to base64
    buffered = io.BytesIO()
    image.save(buffered, format="JPEG")
    img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")

    return render_template('result.html', error=None, class_counts=class_counts, result_image=img_str, class_mapping=class_mapping)

if __name__ == "__main__":
    app.run(debug=True)
