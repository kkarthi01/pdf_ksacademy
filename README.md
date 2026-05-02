# DocScanner — Web App

A DocScanner-style preprocessing pipeline built with **Flask + OpenCV**.
Cleans up photos of documents into crisp scanned-looking images.

---

## Features

- Scanner B/W (adaptive threshold)
- Magic Color (enhanced saturation)
- Soft Grayscale
- High Contrast
- Auto perspective correction (detects document edges, warps)
- PDF support (converts pages → processes → rebuilds PDF)
- Adjustable brightness, contrast, threshold
- Batch scan endpoint (`/scan/batch`)

---

## Setup

### 1. Install system dependency (pdf2image needs poppler)

**Ubuntu/Debian:**
```bash
sudo apt install poppler-utils
```

**macOS:**
```bash
brew install poppler
```

**Windows:**  
Download poppler from https://github.com/oschwartz10612/poppler-windows and add `bin/` to PATH.

---

### 2. Create virtual environment & install packages

```bash
python -m venv venv
source venv/bin/activate      # Windows: venv\Scripts\activate

pip install -r requirements.txt
```

---

### 3. Run the server

```bash
python app.py
```

Then open **http://localhost:5000** in your browser.

---

## Project Structure

```
docscanner/
├── app.py                  # Flask backend (all processing logic)
├── requirements.txt
├── README.md
└── static/
    └── index.html          # Frontend (drag-drop UI)
```

---

## API Reference

### `POST /scan`

| Field        | Type    | Default    | Description                              |
|--------------|---------|------------|------------------------------------------|
| `file`       | file    | required   | Image (jpg/png) or PDF                   |
| `mode`       | string  | scanner    | scanner / magic / grayscale / highcontrast |
| `brightness` | int     | 0          | -100 to 100                              |
| `contrast`   | int     | 20         | -100 to 100                              |
| `threshold`  | int     | 150        | 80 to 220 (used for highcontrast mode)   |
| `warp`       | 0 or 1  | 1          | Enable perspective correction            |
| `output_fmt` | string  | jpeg       | jpeg or png                              |

Returns: processed image bytes (or PDF if input was multi-page PDF).

---

### `POST /scan/batch`

Same params as `/scan` but accepts multiple files via `files` field.  
Returns: single combined PDF with all pages.

---

## Processing Pipeline

```
Input image/PDF
    ↓
PDF → convert pages to images (pdf2image)
    ↓
Resize for edge detection
    ↓
Grayscale → Gaussian blur → Canny edge detection
    ↓
Find largest 4-point contour (document boundary)
    ↓
Perspective warp (four_point_transform)
    ↓
Brightness + contrast adjustment
    ↓
Selected filter:
  scanner      → adaptiveThreshold (Gaussian)
  highcontrast → simple binary threshold
  grayscale    → keep as gray
  magic        → boost HSV saturation
    ↓
Output JPEG/PNG or rebuild PDF
```

---

## Extending

- Add **OCR** with `pytesseract`: `pip install pytesseract`
- Add **denoise** step: `cv2.fastNlMeansDenoising(gray, h=10)`
- Add **deskew** for tilted pages: use `imutils.rotate_bound`
- Deploy to **Render / Railway / Fly.io** (free tiers work fine for this)
