import os
import uuid
import logging
from datetime import datetime
from io import BytesIO

import cv2
import joblib
import numpy as np

from PIL import Image, ExifTags
from flask import Flask, render_template, request, send_file, url_for
from werkzeug.utils import secure_filename
from werkzeug.exceptions import RequestEntityTooLarge
from skimage.feature import hog

from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.lib.utils import ImageReader
from reportlab.pdfgen import canvas


# ==========================================================
# CONFIGURATION
# ==========================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

UPLOAD_FOLDER = os.path.join(BASE_DIR, "static", "uploads")
REPORT_FOLDER = os.path.join(BASE_DIR, "reports")
MODEL_FOLDER = os.path.join(BASE_DIR, "models")

MODEL_PATH = os.path.join(MODEL_FOLDER, "deepfake_svm.pkl")
SCALER_PATH = os.path.join(MODEL_FOLDER, "scaler.pkl")

ALLOWED_EXTENSIONS = {"jpg", "jpeg", "png", "webp"}
MAX_FILE_SIZE = 10 * 1024 * 1024

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(REPORT_FOLDER, exist_ok=True)

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = MAX_FILE_SIZE
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
app.config["REPORT_FOLDER"] = REPORT_FOLDER

logging.basicConfig(level=logging.INFO)

model = None
scaler = None
MODEL_ERROR = None

try:
    if not os.path.isfile(MODEL_PATH):
        raise FileNotFoundError(
            f"Model file not found: {MODEL_PATH}"
        )

    if not os.path.isfile(SCALER_PATH):
        raise FileNotFoundError(
            f"Scaler file not found: {SCALER_PATH}"
        )

    model = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)

    if not hasattr(model, "predict"):
        raise TypeError("The loaded model does not support predict().")

    if not hasattr(scaler, "transform"):
        raise TypeError("The loaded scaler does not support transform().")

except Exception as exc:
    MODEL_ERROR = str(exc)
    app.logger.exception("Model/scaler loading failed.")


# ==========================================================
# IMAGE FEATURES AND PREDICTION
# HOG settings must match train_model.py
# Class 0 = REAL, Class 1 = FAKE
# ==========================================================

def allowed_file(filename):
    return (
        "." in filename
        and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS
    )


def extract_features(image_path):
    image = cv2.imread(image_path)

    if image is None:
        raise ValueError(
            "The uploaded file could not be read as an image."
        )

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.resize(gray, (128, 128))

    features = hog(
        gray,
        orientations=9,
        pixels_per_cell=(8, 8),
        cells_per_block=(2, 2),
        block_norm="L2-Hys",
        transform_sqrt=True,
        feature_vector=True
    )

    return features.reshape(1, -1)


def get_label_name(label):
    value = str(label).strip().lower()

    if value in {"0", "0.0", "real", "genuine", "authentic"}:
        return "REAL"

    if value in {
        "1", "1.0", "fake", "ai",
        "ai generated", "ai-generated", "deepfake"
    }:
        return "FAKE"

    raise ValueError(
        f"Unknown model label {label!r}. "
        "Check class labels in train_model.py."
    )


def predict_image(image_path):
    if model is None or scaler is None:
        raise RuntimeError(
            f"Model is not ready: {MODEL_ERROR}"
        )

    features = extract_features(image_path)
    scaled_features = scaler.transform(features)

    predicted_label = model.predict(scaled_features)[0]
    result = get_label_name(predicted_label)

    if not hasattr(model, "predict_proba"):
        raise RuntimeError(
            "The loaded model has no predict_proba(). "
            "Check how the model was trained."
        )

    probabilities = model.predict_proba(scaled_features)[0]
    classes = list(model.classes_)

    real_probability = None
    fake_probability = None

    for index, label in enumerate(classes):
        label_name = get_label_name(label)
        probability = float(probabilities[index]) * 100

        if label_name == "REAL":
            real_probability = probability
        elif label_name == "FAKE":
            fake_probability = probability

    if real_probability is None or fake_probability is None:
        raise RuntimeError(
            "The model must have both REAL and FAKE classes."
        )

    confidence = (
        real_probability if result == "REAL"
        else fake_probability
    )

    if fake_probability >= 75:
        risk_level = "HIGH"
    elif fake_probability >= 45:
        risk_level = "MEDIUM"
    else:
        risk_level = "LOW"

    return {
        "result": result,
        "confidence": round(confidence, 2),
        "real_probability": round(real_probability, 2),
        "fake_probability": round(fake_probability, 2),
        "risk_level": risk_level,
        "reliability": "Indicative only",
        "description": (
            "The model classified this image as REAL."
            if result == "REAL"
            else "The model classified this image as FAKE / AI GENERATED."
        )
    }


# ==========================================================
# IMAGE INFORMATION AND EXIF METADATA
# ==========================================================

def analyze_image(image_path):
    with Image.open(image_path) as image:
        image.load()

        width, height = image.size
        file_size = os.path.getsize(image_path)

        gray = np.array(
            image.convert("RGB").convert("L"),
            dtype=np.uint8
        )

        brightness = float(np.mean(gray))
        contrast = float(np.std(gray))
        sharpness = float(
            cv2.Laplacian(gray, cv2.CV_64F).var()
        )

        metadata = {
            "exif": "Not Available",
            "camera_make": "Not Available",
            "camera_model": "Not Available",
            "software": "Not Available",
            "date_created": "Not Available"
        }

        try:
            exif = image.getexif()

            if exif:
                decoded = {
                    ExifTags.TAGS.get(tag, str(tag)): str(value)
                    for tag, value in exif.items()
                }

                metadata["exif"] = "Yes"
                metadata["camera_make"] = decoded.get(
                    "Make", "Not Available"
                )
                metadata["camera_model"] = decoded.get(
                    "Model", "Not Available"
                )
                metadata["software"] = decoded.get(
                    "Software", "Not Available"
                )
                metadata["date_created"] = decoded.get(
                    "DateTimeOriginal",
                    decoded.get("DateTime", "Not Available")
                )

        except Exception:
            pass

        if file_size < 1024 * 1024:
            size_text = f"{file_size / 1024:.2f} KB"
        else:
            size_text = f"{file_size / (1024 * 1024):.2f} MB"

        return {
            "format": image.format or "Unknown",
            "width": width,
            "height": height,
            "dimensions": f"{width} × {height}",
            "aspect_ratio": (
                f"{width / height:.2f}" if height else "Unknown"
            ),
            "color_mode": image.mode,
            "size": size_text,
            "brightness": round(brightness, 2),
            "contrast": round(contrast, 2),
            "sharpness": round(sharpness, 2),
            **metadata
        }


# ==========================================================
# PDF REPORT
# Page 1: Uploaded image and prediction
# Page 2: Probability, image details and metadata
# Text and image have separate areas
# ==========================================================

def generate_pdf_report(
    path, image_path, filename, details,
    prediction, detection_time
):
    W, H = A4
    pdf = canvas.Canvas(path, pagesize=A4)

    navy = colors.HexColor("#14213D")
    cyan = colors.HexColor("#18B6D9")
    white = colors.white
    dark = colors.HexColor("#202B3C")
    muted = colors.HexColor("#68758A")
    light = colors.HexColor("#F2F6FA")
    border = colors.HexColor("#DCE4EE")
    green = colors.HexColor("#159957")
    red = colors.HexColor("#D93652")

    result = str(
        prediction.get("result", "UNKNOWN")
    ).upper()

    result_text = (
        "REAL IMAGE" if result == "REAL"
        else "AI / FAKE IMAGE"
    )

    result_color = green if result == "REAL" else red

    real_p = float(prediction.get("real_probability", 0))
    fake_p = float(prediction.get("fake_probability", 0))
    confidence = float(prediction.get("confidence", 0))

    def header(subtitle, page_no):
        pdf.setFillColor(navy)
        pdf.rect(0, H - 72, W, 72, fill=1, stroke=0)

        pdf.setFillColor(white)
        pdf.setFont("Helvetica-Bold", 16)
        pdf.drawString(
            38, H - 32, "DeepFake Image Detection"
        )

        pdf.setFillColor(cyan)
        pdf.setFont("Helvetica", 9)
        pdf.drawString(38, H - 50, subtitle)

        pdf.setStrokeColor(border)
        pdf.line(38, 39, W - 38, 39)

        pdf.setFillColor(muted)
        pdf.setFont("Helvetica", 7.5)
        pdf.drawString(
            38, 25,
            "Automated image analysis | Model estimate only"
        )
        pdf.drawRightString(W - 38, 25, f"Page {page_no}")

    def card(x, y, w, h, fill=light):
        pdf.setFillColor(fill)
        pdf.roundRect(
            x, y, w, h, 9, fill=1, stroke=0
        )

    def section_title(title, y):
        pdf.setFillColor(dark)
        pdf.setFont("Helvetica-Bold", 12)
        pdf.drawString(38, y, title)
        return y - 22

    def detail_row(label, value, x, y, value_x=None, max_chars=52):
        pdf.setFillColor(muted)
        pdf.setFont("Helvetica", 8.5)
        pdf.drawString(x, y, str(label)[:32])

        pdf.setFillColor(dark)
        pdf.setFont("Helvetica-Bold", 8.5)
        pdf.drawString(
            value_x or x + 150,
            y,
            str(value)[:max_chars]
        )

    def draw_bar(label, value, y, bar_color):
        value = max(0.0, min(100.0, float(value)))

        pdf.setFillColor(dark)
        pdf.setFont("Helvetica", 9)
        pdf.drawString(44, y + 2, label)

        pdf.setFillColor(muted)
        pdf.drawRightString(
            W - 44, y + 2, f"{value:.2f}%"
        )

        bx, bw, bh = 44, W - 88, 9

        pdf.setFillColor(border)
        pdf.roundRect(
            bx, y - 13, bw, bh, 4, fill=1, stroke=0
        )

        if value > 0:
            pdf.setFillColor(bar_color)
            pdf.roundRect(
                bx, y - 13, bw * value / 100,
                bh, 4, fill=1, stroke=0
            )

    # ------------------------------------------------------
    # PAGE 1: IMAGE AND RESULT
    # ------------------------------------------------------

    header("IMAGE ANALYSIS REPORT", 1)

    pdf.setFillColor(muted)
    pdf.setFont("Helvetica", 8.5)
    pdf.drawString(40, H - 94, f"File: {filename[:70]}")
    pdf.drawString(
        40, H - 109,
        f"Analysis date: {detection_time}"
    )

    # Uploaded image card
    image_x = 40
    image_y = H - 420
    image_w = 245
    image_h = 275

    card(image_x, image_y, image_w, image_h)

    pdf.setFillColor(dark)
    pdf.setFont("Helvetica-Bold", 10)
    pdf.drawString(
        image_x + 15,
        image_y + image_h - 23,
        "UPLOADED IMAGE"
    )

    preview_x = image_x + 15
    preview_y = image_y + 18
    preview_w = image_w - 30
    preview_h = image_h - 55

    try:
        with Image.open(image_path) as img:
            img.load()
            preview = img.convert("RGB")

            preview.thumbnail((
                int(preview_w),
                int(preview_h)
            ))

            image_buffer = BytesIO()
            preview.save(
                image_buffer,
                format="JPEG",
                quality=90
            )
            image_buffer.seek(0)

            pdf.drawImage(
                ImageReader(image_buffer),
                preview_x,
                preview_y,
                width=preview_w,
                height=preview_h,
                preserveAspectRatio=True,
                anchor="c",
                mask="auto"
            )

    except Exception:
        pdf.setFillColor(muted)
        pdf.setFont("Helvetica", 9)
        pdf.drawCentredString(
            image_x + image_w / 2,
            image_y + image_h / 2,
            "Image preview unavailable"
        )

    # Prediction card placed separately from the image
    result_x = 302
    result_y = image_y
    result_w = W - result_x - 40
    result_h = image_h

    card(result_x, result_y, result_w, result_h)

    pdf.setFillColor(dark)
    pdf.setFont("Helvetica-Bold", 10)
    pdf.drawString(
        result_x + 14,
        result_y + result_h - 23,
        "PREDICTION"
    )

    pdf.setFillColor(result_color)
    pdf.setFont("Helvetica-Bold", 17)
    pdf.drawString(
        result_x + 14,
        result_y + result_h - 53,
        result_text
    )

    details_y = result_y + result_h - 88

    for label, value in [
        ("Confidence", f"{confidence:.2f}%"),
        ("Real probability", f"{real_p:.2f}%"),
        ("Fake / AI probability", f"{fake_p:.2f}%"),
        (
            "Risk indicator",
            prediction.get("risk_level", "Not available")
        ),
    ]:
        pdf.setFillColor(muted)
        pdf.setFont("Helvetica", 8)
        pdf.drawString(
            result_x + 14, details_y, label
        )

        pdf.setFillColor(dark)
        pdf.setFont("Helvetica-Bold", 8.5)
        pdf.drawString(
            result_x + 14,
            details_y - 13,
            str(value)[:28]
        )

        details_y -= 42

    # Summary below the image and result cards
    sy = image_y - 32
    sy = section_title("IMAGE SUMMARY", sy)

    summary = [
        ("Format", details.get("format", "Not Available")),
        ("Dimensions", details.get("dimensions", "Not Available")),
        ("File size", details.get("size", "Not Available")),
        ("Color mode", details.get("color_mode", "Not Available")),
    ]

    col_positions = [42, 215, 400, 42]

    for i, (label, value) in enumerate(summary):
        col = i if i < 3 else 0
        x = col_positions[col]
        y = sy if i < 3 else sy - 48

        card(x, y - 25, 155, 38, white)

        pdf.setStrokeColor(border)
        pdf.roundRect(
            x, y - 25, 155, 38, 7,
            fill=0, stroke=1
        )

        pdf.setFillColor(muted)
        pdf.setFont("Helvetica", 7.5)
        pdf.drawString(x + 9, y + 1, label)

        pdf.setFillColor(dark)
        pdf.setFont("Helvetica-Bold", 8.5)
        pdf.drawString(
            x + 9, y - 13, str(value)[:25]
        )

    pdf.showPage()

    # ------------------------------------------------------
    # PAGE 2: TECHNICAL DETAILS AND METADATA
    # ------------------------------------------------------

    header("TECHNICAL IMAGE ANALYSIS", 2)

    y = H - 100
    y = section_title("PROBABILITY ANALYSIS", y)

    draw_bar("Real probability", real_p, y, green)
    draw_bar("Fake / AI probability", fake_p, y - 43, red)

    y -= 100
    y = section_title(
        "IMAGE QUALITY AND TECHNICAL DETAILS", y
    )

    rows = [
        ("Filename", filename),
        ("Format", details.get("format", "Not Available")),
        ("Dimensions", details.get("dimensions", "Not Available")),
        ("Aspect ratio", details.get("aspect_ratio", "Not Available")),
        ("File size", details.get("size", "Not Available")),
        ("Color mode", details.get("color_mode", "Not Available")),
        ("Brightness", details.get("brightness", "Not Available")),
        ("Contrast", details.get("contrast", "Not Available")),
        ("Sharpness", details.get("sharpness", "Not Available")),
    ]

    for i, (label, value) in enumerate(rows):
        if i % 2 == 0:
            card(40, y - 9, W - 80, 25, white)

            pdf.setStrokeColor(border)
            pdf.roundRect(
                40, y - 9, W - 80, 25, 5,
                fill=0, stroke=1
            )

        detail_row(
            label, value, 50, y,
            value_x=205, max_chars=58
        )

        y -= 29

    y -= 8
    y = section_title("IMAGE METADATA (EXIF)", y)

    metadata_rows = [
        (
            "Metadata availability",
            details.get("exif", "Not Available")
        ),
        (
            "Camera make",
            details.get("camera_make", "Not Available")
        ),
        (
            "Camera model",
            details.get("camera_model", "Not Available")
        ),
        (
            "Software",
            details.get("software", "Not Available")
        ),
        (
            "Date created",
            details.get("date_created", "Not Available")
        ),
    ]

    for label, value in metadata_rows:
        detail_row(
            label, value, 45, y,
            value_x=190, max_chars=62
        )
        y -= 20

    # Disclaimer in a separate footer area
    pdf.setFillColor(colors.HexColor("#FFF4D8"))
    pdf.roundRect(
        40, 52, W - 80, 44, 7,
        fill=1, stroke=0
    )

    pdf.setFillColor(colors.HexColor("#805A12"))
    pdf.setFont("Helvetica-Bold", 8)
    pdf.drawString(50, 81, "IMPORTANT NOTE")

    pdf.setFont("Helvetica", 7.5)
    pdf.drawString(
        50, 67,
        "Confidence is a model estimate, not verified accuracy."
    )
    pdf.drawString(
        50, 56,
        "Missing EXIF data or image-quality values do not prove an image is fake."
    )

    pdf.save()


# ==========================================================
# FLASK ROUTES
# ==========================================================

def render_home(**kwargs):
    context = {
        "model_ready": model is not None and scaler is not None,
        "model_error": MODEL_ERROR
    }

    context.update(kwargs)

    return render_template("index.html", **context)


@app.route("/", methods=["GET"])
def index():
    return render_home()


@app.route("/predict", methods=["POST"])
def predict():
    if model is None or scaler is None:
        return render_home(
            error=f"Model is not ready: {MODEL_ERROR}"
        )

    uploaded = request.files.get("image")

    if uploaded is None or not uploaded.filename:
        return render_home(
            error="Please select an image first."
        )

    if not allowed_file(uploaded.filename):
        return render_home(
            error="Allowed formats: JPG, JPEG, PNG and WEBP."
        )

    original_filename = (
        secure_filename(uploaded.filename) or "image.jpg"
    )

    extension = original_filename.rsplit(".", 1)[1].lower()
    unique_name = f"{uuid.uuid4().hex}.{extension}"

    image_path = os.path.join(
        UPLOAD_FOLDER, unique_name
    )

    try:
        uploaded.save(image_path)

        # Verify that the uploaded content is an image
        with Image.open(image_path) as img:
            img.verify()

        details = analyze_image(image_path)
        prediction = predict_image(image_path)

        detection_time = datetime.now().strftime(
            "%d-%m-%Y %I:%M:%S %p"
        )

        report_filename = (
            f"report_{uuid.uuid4().hex}.pdf"
        )

        report_path = os.path.join(
            REPORT_FOLDER, report_filename
        )

        generate_pdf_report(
            report_path,
            image_path,
            original_filename,
            details,
            prediction,
            detection_time
        )

        return render_home(
            **prediction,
            image_path=url_for(
                "static",
                filename=f"uploads/{unique_name}"
            ),
            filename=original_filename,
            details=details,
            report_available=True,
            report_filename=report_filename,
            detection_time=detection_time
        )

    except Exception as exc:
        app.logger.exception("Image analysis failed")

        try:
            if os.path.isfile(image_path):
                os.remove(image_path)
        except OSError:
            pass

        return render_home(
            error=f"Analysis failed: {exc}"
        )


@app.route(
    "/download_report/<report_filename>",
    methods=["GET"]
)
def download_report(report_filename):
    safe_name = secure_filename(report_filename)

    if safe_name != report_filename or not safe_name.endswith(".pdf"):
        return "Invalid report filename.", 400

    report_path = os.path.join(
        REPORT_FOLDER, safe_name
    )

    if not os.path.isfile(report_path):
        return "Report not found. Analyze the image again.", 404

    return send_file(
        report_path,
        as_attachment=True,
        download_name=safe_name,
        mimetype="application/pdf"
    )


@app.errorhandler(RequestEntityTooLarge)
def file_too_large(_error):
    return render_home(
        error="Image is too large. Please upload an image under 10 MB."
    ), 413


@app.route("/health", methods=["GET"])
def health():
    return {
        "status": "ok",
        "model_loaded": model is not None,
        "scaler_loaded": scaler is not None,
        "model_error": MODEL_ERROR
    }


if __name__ == "__main__":
    app.run(debug=True)