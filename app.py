from flask import Flask, request, send_file, jsonify
import cv2
import numpy as np
from pdf2image import convert_from_bytes
from PIL import Image
import io
import imutils

app = Flask(__name__, static_folder="static")


def order_points(pts):
    """Order 4 points: top-left, top-right, bottom-right, bottom-left."""
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    return rect


def four_point_transform(image, pts):
    """Apply perspective warp to isolate document."""
    rect = order_points(pts)
    (tl, tr, br, bl) = rect
    widthA = np.linalg.norm(br - bl)
    widthB = np.linalg.norm(tr - tl)
    maxWidth = max(int(widthA), int(widthB))
    heightA = np.linalg.norm(tr - br)
    heightB = np.linalg.norm(tl - bl)
    maxHeight = max(int(heightA), int(heightB))
    dst = np.array(
        [[0, 0], [maxWidth - 1, 0], [maxWidth - 1, maxHeight - 1], [0, maxHeight - 1]],
        dtype="float32",
    )
    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))
    return warped


def detect_and_warp(image):
    """Detect document edges and warp perspective. Returns warped or original."""
    orig = image.copy()
    ratio = image.shape[0] / 500.0
    image = imutils.resize(image, height=500)

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blur, 75, 200)

    cnts, _ = cv2.findContours(edges.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    cnts = sorted(cnts, key=cv2.contourArea, reverse=True)[:5]

    screenCnt = None
    for c in cnts:
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.02 * peri, True)
        if len(approx) == 4:
            screenCnt = approx
            break

    if screenCnt is None:
        return orig  # no document boundary found, return original

    warped = four_point_transform(orig, screenCnt.reshape(4, 2) * ratio)
    return warped


def apply_scanner_filter(image, mode="scanner", brightness=0, contrast=20, threshold=150):
    """Apply selected filter mode to image (BGR numpy array)."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Brightness + Contrast
    factor = (259 * (contrast + 255)) / (255 * (259 - contrast))
    gray = np.clip(factor * (gray.astype(np.float32) - 128) + 128 + brightness, 0, 255).astype(np.uint8)

    if mode == "scanner":
        result = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
        )
        result = cv2.cvtColor(result, cv2.COLOR_GRAY2BGR)

    elif mode == "highcontrast":
        _, result = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY)
        result = cv2.cvtColor(result, cv2.COLOR_GRAY2BGR)

    elif mode == "grayscale":
        result = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

    elif mode == "magic":
        # Enhance saturation for "magic color" effect
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV).astype(np.float32)
        hsv[:, :, 1] = np.clip(hsv[:, :, 1] * 1.6, 0, 255)
        hsv[:, :, 2] = np.clip(factor * (hsv[:, :, 2] - 128) + 128 + brightness, 0, 255)
        result = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)

    else:
        result = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

    return result


def encode_image(img_bgr, fmt="jpeg"):
    """Encode cv2 BGR image to bytes."""
    ext = ".jpg" if fmt == "jpeg" else ".png"
    _, buf = cv2.imencode(ext, img_bgr)
    return buf.tobytes()


@app.route("/")
def index():
    return app.send_static_file("index.html")


@app.route("/scan", methods=["POST"])
def scan():
    """
    POST /scan
    Form fields:
      file         — image (jpg/png) or PDF
      mode         — scanner | magic | grayscale | highcontrast  (default: scanner)
      brightness   — int -100..100 (default: 0)
      contrast     — int -100..100 (default: 20)
      threshold    — int 80..220   (default: 150)
      warp         — 0|1 perspective correction (default: 1)
      output_fmt   — jpeg|png (default: jpeg)
    Returns:
      Processed image bytes (single image or first page of PDF).
    """
    if "file" not in request.files:
        return jsonify(error="No file uploaded"), 400

    file = request.files["file"]
    mode = request.form.get("mode", "scanner")
    brightness = int(request.form.get("brightness", 0))
    contrast = int(request.form.get("contrast", 20))
    threshold = int(request.form.get("threshold", 150))
    do_warp = request.form.get("warp", "1") == "1"
    output_fmt = request.form.get("output_fmt", "jpeg")

    raw = file.read()
    mime = file.content_type or ""

    # --- PDF handling ---
    if mime == "application/pdf" or file.filename.lower().endswith(".pdf"):
        try:
            pages = convert_from_bytes(raw, dpi=200)
        except Exception as e:
            return jsonify(error=f"PDF conversion failed. Is poppler installed? Details: {str(e)}"), 500

        results = []
        for page_pil in pages:
            img = cv2.cvtColor(np.array(page_pil), cv2.COLOR_RGB2BGR)
            if do_warp:
                img = detect_and_warp(img)
            img = apply_scanner_filter(img, mode, brightness, contrast, threshold)
            results.append(img)

        if len(results) == 1:
            return send_file(
                io.BytesIO(encode_image(results[0], output_fmt)),
                mimetype=f"image/{output_fmt}",
            )

        # Multiple pages → return as multipage PDF
        pil_pages = [Image.fromarray(cv2.cvtColor(r, cv2.COLOR_BGR2RGB)) for r in results]
        out_buf = io.BytesIO()
        pil_pages[0].save(out_buf, "PDF", save_all=True, append_images=pil_pages[1:])
        out_buf.seek(0)
        resp = send_file(out_buf, mimetype="application/pdf", download_name="scanned.pdf")
        resp.headers['x-page-count'] = str(len(pil_pages))
        return resp

    # --- Image handling ---
    img_array = np.frombuffer(raw, np.uint8)
    img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
    if img is None:
        return jsonify(error="Could not decode image — make sure it is a valid JPG or PNG"), 400

    try:
        if do_warp:
            img = detect_and_warp(img)
        img = apply_scanner_filter(img, mode, brightness, contrast, threshold)
    except Exception as e:
        return jsonify(error=f"Processing failed: {str(e)}"), 500
    return send_file(
        io.BytesIO(encode_image(img, output_fmt)),
        mimetype=f"image/{output_fmt}",
    )


@app.route("/scan/batch", methods=["POST"])
def scan_batch():
    """
    POST /scan/batch
    Upload multiple images, returns a single combined PDF.
    Form fields: same as /scan, plus multiple 'files' entries.
    """
    files = request.files.getlist("files")
    if not files:
        return jsonify(error="No files uploaded"), 400

    mode = request.form.get("mode", "scanner")
    brightness = int(request.form.get("brightness", 0))
    contrast = int(request.form.get("contrast", 20))
    threshold = int(request.form.get("threshold", 150))
    do_warp = request.form.get("warp", "1") == "1"

    pil_pages = []
    for f in files:
        raw = f.read()
        img_array = np.frombuffer(raw, np.uint8)
        img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
        if img is None:
            continue
        if do_warp:
            img = detect_and_warp(img)
        img = apply_scanner_filter(img, mode, brightness, contrast, threshold)
        pil_pages.append(Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB)))

    if not pil_pages:
        return jsonify(error="No valid images"), 400

    out_buf = io.BytesIO()
    pil_pages[0].save(out_buf, "PDF", save_all=True, append_images=pil_pages[1:])
    out_buf.seek(0)
    return send_file(out_buf, mimetype="application/pdf", download_name="batch_scan.pdf")


if __name__ == "__main__":
    app.run(debug=True, port=5000)
