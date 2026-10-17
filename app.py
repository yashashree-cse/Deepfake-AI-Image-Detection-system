from flask import Flask, render_template, request
from PIL import Image
from transformers import pipeline

app = Flask(__name__)

# AI model for identifying things in an image
classifier = pipeline("image-classification")
ai_detector = pipeline(
    "image-classification",
    model="dima806/deepfake_vs_real_image_detection"
)

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    file = request.files["image"]

    image = Image.open(file)

    image_name = file.filename
    image_type = image.format
    image_width, image_height = image.size

    results = classifier(image)

    detected_thing = results[0]["label"]
    confidence = round(results[0]["score"] * 100, 2)
    ai_results = ai_detector(image)

    ai_label = ai_results[0]["label"]
    ai_confidence = round(ai_results[0]["score"] * 100, 2)

    return f"""
    <h1>Image Information</h1>

    <p><b>Image Name:</b> {image_name}</p>
    <p><b>Image Type:</b> {image_type}</p>
    <p><b>Image Dimensions:</b> {image_width} × {image_height}</p>

    <h2>What is in the image?</h2>
    <p><b>Detected:</b> {detected_thing}</p>
    <p><b>Confidence:</b> {confidence}%</p>
    <h2>AI Image Detection</h2>
<p><b>Result:</b> {ai_label}</p>
<p><b>Confidence:</b> {ai_confidence}%</p>
    """

if __name__ == "__main__":
    app.run(debug=True)