import argparse
import random
import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageFilter
from pathlib import Path


def rand_affine(img: Image.Image):
    w, h = img.size
    dx = random.randint(-5, 5)
    dy = random.randint(-3, 3)
    return img.transform((w, h), Image.AFFINE, (1, dx*0.005, dx, dy*0.005, 1, dy), resample=Image.BICUBIC)


def rand_noise(img: Image.Image):
    arr = np.array(img).astype(np.int16)
    noise = np.random.normal(0, 6, arr.shape)
    arr = np.clip(arr + noise, 0, 255).astype(np.uint8)
    out = Image.fromarray(arr)
    if random.random() < 0.3:
        out = out.filter(ImageFilter.GaussianBlur(radius=0.6))
    return out


def render_text_line(text: str, font: ImageFont.FreeTypeFont, pad=12):
    dummy = Image.new("RGB", (10, 10), (255, 255, 255))
    d = ImageDraw.Draw(dummy)
    w, h = d.textsize(text, font=font)
    img = Image.new("RGB", (w + pad*2, h + pad*2), (255, 255, 255))
    d = ImageDraw.Draw(img)
    d.text((pad, pad), text, fill=(0, 0, 0), font=font)
    return img


def main_synth():
    parser = argparse.ArgumentParser()
    parser.add_argument("--texts", required=True)
    parser.add_argument("--fonts", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--n_per", type=int, default=3)
    parser.add_argument("--font_size", type=int, default=36)
    args = parser.parse_args()

    texts = [t.strip() for t in Path(args.texts).read_text(encoding="utf-8").splitlines() if t.strip()]
    fonts = [str(p) for p in Path(args.fonts).glob("*.ttf")]

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    rows = []
    idx = 0
    for t in texts:
        for _ in range(args.n_per):
            fpath = random.choice(fonts)
            font = ImageFont.truetype(fpath, args.font_size)
            img = render_text_line(t, font)
            if random.random() < 0.8:
                img = rand_affine(img)
            if random.random() < 0.8:
                img = rand_noise(img)
            fp = out / f"{idx:06d}.jpg"
            img.save(fp, quality=95)
            rows.append(f"{fp.name},{t}")
            idx += 1

    # manifest 저장
    (out.parent / "train_manifest.csv").write_text("image_path,text\n" + "\n".join(rows), encoding="utf-8")

if __name__ == "__main__" and False:
    main_synth()