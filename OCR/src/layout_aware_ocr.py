import argparse
from pathlib import Path
from typing import List
from PIL import Image, ImageDraw
import torch
import numpy as np
from doclayout_yolo import YOLOv10
from paddleocr import TextRecognition,TableCellsDetection
from img2table.document import Image as ImgDoc
from img2table.ocr import PaddleOCR as PaddleOCR_Img
import os
import cv2
from paddleocr import ChartParsing
from doctr.io import DocumentFile
from doctr.models import ocr_predictor
import json

def main_layout():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", required=True)
    parser.add_argument("--save_vis", default=None)
    parser.add_argument("--text_like", nargs="*", default=[
        "text", "paragraph", "title", "list", "caption"
    ], help="텍스트로 간주할 YOLO 클래스 이름 목록 (소문자 기준)")
    parser.add_argument("--conf", type=float, default=0.3)
    args = parser.parse_args()

    yolo = YOLOv10("E:/glass_git/AI-portfolio/OCR/model/doclayout_yolo_docstructbench_imgsz1024.pt")
    ocr_model = TextRecognition(model_name="PP-OCRv5_server_rec")
    table_engine =TableCellsDetection(model_name="RT-DETR-L_wired_table_cell_det")
    chartmodel = ChartParsing(model_name="PP-Chart2Table")
    ocr_model2 = ocr_predictor(pretrained=True)

    image = Image.open(args.image).convert("RGB")
    w, h = image.size

    # YOLO 추론
    yres = yolo.predict(source=image, conf=args.conf, verbose=False)[0]
    class_names = [str(c).lower() for c in yres.names.values()]
    boxes = yres.boxes.xyxy.cpu().numpy() if yres.boxes is not None else []
    clses = yres.boxes.cls.cpu().numpy().astype(int) if yres.boxes is not None else []

    draw = ImageDraw.Draw(image)

    for i, (bbox, ci) in enumerate(zip(boxes, clses), 1):
        cls_name = class_names[ci] if 0 <= ci < len(class_names) else str(ci)

        x1, y1, x2, y2 = map(int, bbox.tolist())
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)
        crop = image.crop((x1, y1, x2, y2))

        out_dir = Path(args.save_vis) if args.save_vis else Path("./output")
        out_dir.mkdir(parents=True, exist_ok=True)

        if 'table' in cls_name:
            crop.save(str(out_dir / f"table_{i}.png"), format="PNG")

            doc = ImgDoc(str(out_dir / f"table_{i}.png"))
            ocr = PaddleOCR_Img(lang="en")  
            tables = doc.extract_tables(ocr=ocr) 
            for _, t in enumerate(tables, 1):
                t.df.to_csv(str(out_dir / f"table_{i}.csv"), index=False, encoding="utf-8-sig")
            
            results = chartmodel.predict(
                input={"image": str(out_dir / f"table_{i}.png")},
                batch_size=1)
            
            for res in results:
                res.print()
                res.save_to_json(str(out_dir / f"table_{i}.json"))

        else:
            crop.save(str(out_dir / f"ocr_{i}.png"), format="PNG")
            doc = DocumentFile.from_images(str(out_dir / f"ocr_{i}.png"))  
            result = ocr_model2(doc)
            output = result.export()
            with open(str(out_dir / f"ocr_{i}.json"), "w", encoding="utf-8") as f:
                json.dump(output, f, ensure_ascii=False, indent=4)
                    
        if args.save_vis:
            draw.rectangle([x1, y1, x2, y2], outline=(255, 0, 0), width=2)
            draw.text((x1+3, y1+3), cls_name, fill=(255, 0, 0))

    if args.save_vis:
        image.save(str(Path(args.save_vis) / "vis.png"))


   
if __name__ == "__main__":
    main_layout()
