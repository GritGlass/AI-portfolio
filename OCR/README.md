# 🛠️ Document OCR 

A tool that processes document images through layout detection → table extraction → text OCR → visualization/storage in a single pipeline.

<p align="center">
  <img src="https://img.shields.io/badge/Python-3776AB?style=flat&logo=python&logoColor=white"/>
  <img src="https://img.shields.io/badge/Pandas-150458?style=flat&logo=pandas&logoColor=white"/>
  <img src="https://img.shields.io/badge/PyTorch-EE4C2C?style=flat&logo=pytorch&logoColor=white"/>
</p>

---

## 📷 Example
<div align="center">
    <img src="https://github.com/GritGlass/AI-portfolio/blob/3a4e16a130a4b0aba49cca62070fb28ec93e339f/OCR/assets/OCR.png"
         width="640" />
  </a>
  <br/>
  <sub>Image Segmentation Analysis App</sub>
</div>

---

## 🔧 How it Works (Pipeline)

1. Input Load: Load a document image and convert to RGB
2. Layout Detection: Run DocLayout-YOLO to get bounding boxes and class labels
3. Branch Processing
      - If class contains "table" → process via img2table + PaddleOCR (CSV) + PP-Chart2Table (JSON)
      - Else → run doctr OCR (JSON output)
4. Visualization (show_result)
      - Left panel → input/crop image
      - Right panel → OCR text (with line wrapping) or table (rendered via ax.table)
5. Optional: --save_vis overlays bounding boxes and class labels on the original document

## 📦 Installation

1. Clone this repository
    ```
    git clone https://github.com/yourname/document-ocr-analyzer.git

    cd document-ocr-analyzer
    ```

2. Install dependencies from requirements.txt
   ```
   pip install -r requirements.txt
   ```

3. Run the script
   ```
    python your_script.py --image ./samples/doc1.png --save_vis ./output
   ```
  - Example output folder (./output):
    - ocr_1.png, ocr_1.json
    - table_2.png, table_2.csv, table_2.json
    - vis.png 

## 📌 Notes
- `--conf` sets YOLO detection confidence (default: 0.3)
- Class names are compared in lowercase (e.g. 'table')
- Long text is wrapped with wrap_text(width=30) before display
- Tables are resized for readability (set_fontsize, scale)

## Features

- Document Layout Detection (DocLayout-YOLO v10)
  - Detects document regions and crops by class
  - table regions are routed to the table pipeline, other regions to the text OCR pipeline
        
- Table Extraction (dual approach)
  - img2table + PaddleOCR → structured tables saved as CSV

- PP-Chart2Table (ChartParsing) → tables/charts exported as JSON

- Text OCR (doctr)
  - Runs ocr_predictor on cropped images
  - Exports recognized text as JSON

- Result Visualization (matplotlib)
  - Left: cropped/processed image

  - Right:
    - Text results → automatically wrapped every N chars, displayed centrally
    - Table results → rendered as matplotlib.table for DataFrame-like visualization
    - Postprocessing Utilities
    - extract_values → recursively collects only "value" keys from JSON
    - wrap_text → breaks long strings every N characters (\n)
    - Basic noise cleanup (e.g. removing stray " 0")

- Outputs
  - ocr_*.json, ocr_*.png for text crops
  - table_*.csv, table_*.json, table_*.png for tables
  - vis.png with overlaid detections (when --save_vis is used)

---

## 🧪 Tech Stack

| Tool       | Purpose               |
|------------|------------------------|
| DocLayout-YOLO v10     | Document layout detection  |
| img2table + PaddleOCR  | Table structure extraction (CSV) |
| Doctr   | Text OCR and JSON export  |




