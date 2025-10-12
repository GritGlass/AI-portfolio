import argparse
import json
from pathlib import Path
from PIL import Image
from pdf2image import convert_from_path
import torch
from transformers import TrOCRProcessor, VisionEncoderDecoderModel

@torch.no_grad()
def trocr_infer_image(img: Image.Image, processor, model, device: str, max_new_tokens: int = 256, num_beams: int = 3):
    pixel_values = processor(images=img, return_tensors="pt").pixel_values.to(device)
    outputs = model.generate(pixel_values, max_new_tokens=max_new_tokens, num_beams=num_beams)
    text = processor.batch_decode(outputs, skip_special_tokens=True)[0]
    return text.strip()


def main_pdf():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pdf", required=True)
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--model", default="microsoft/trocr-base-printed", help="파인튜닝 가중치 경로 또는 허깅페이스 모델명")
    parser.add_argument("--dpi", type=int, default=300)
    parser.add_argument("--num_beams", type=int, default=3)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    processor = TrOCRProcessor.from_pretrained(args.model)
    model = VisionEncoderDecoderModel.from_pretrained(args.model).to(device)
    if device == "cuda":
        model.half()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    pages = convert_from_path(args.pdf, dpi=args.dpi)
    all_results = []

    for i, page in enumerate(pages, 1):
        text = trocr_infer_image(page.convert("RGB"), processor, model, device, num_beams=args.num_beams)
        (out_dir / f"page_{i:04d}.txt").write_text(text, encoding="utf-8")
        all_results.append({"page": i, "text": text})

    (out_dir / "result.json").write_text(json.dumps(all_results, ensure_ascii=False, indent=2), encoding="utf-8")

if __name__ == "__main__" and False:
    main_pdf()
