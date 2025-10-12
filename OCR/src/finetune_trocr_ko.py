import os
import argparse
from dataclasses import dataclass
from typing import Dict, Any
import pandas as pd
from PIL import Image
import torch
from torch.utils.data import Dataset
from transformers import (
    TrOCRProcessor,
    VisionEncoderDecoderModel,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
)

class OCRLineDataset(Dataset):
    def __init__(self, csv_path: str, images_dir: str, processor: TrOCRProcessor, max_target_len: int = 256):
        self.df = pd.read_csv(csv_path)
        self.images_dir = images_dir
        self.processor = processor
        self.max_target_len = max_target_len

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_fp = os.path.join(self.images_dir, row["image_path"]) if not os.path.isabs(row["image_path"]) else row["image_path"]
        text = str(row["text"]) if pd.notna(row["text"]) else ""
        image = Image.open(img_fp).convert("RGB")
        pixel_values = self.processor(images=image, return_tensors="pt").pixel_values.squeeze(0)
        with self.processor.as_target_processor():
            labels = self.processor(text, max_length=self.max_target_len, return_tensors="pt", truncation=True).input_ids.squeeze(0)
        return {"pixel_values": pixel_values, "labels": labels}

@dataclass
class DataCollatorOCR:
    processor: TrOCRProcessor

    def __call__(self, features):
        pixel_values = torch.stack([f["pixel_values"] for f in features])
        labels = torch.nn.utils.rnn.pad_sequence(
            [f["labels"] for f in features], batch_first=True, padding_value=self.processor.tokenizer.pad_token_id
        )
        labels[labels == self.processor.tokenizer.pad_token_id] = -100
        return {"pixel_values": pixel_values, "labels": labels}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="microsoft/trocr-base-printed")
    parser.add_argument("--train_csv", required=True)
    parser.add_argument("--valid_csv", required=True)
    parser.add_argument("--images_dir", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--per_device_batch_size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--fp16", action="store_true")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    processor = TrOCRProcessor.from_pretrained(args.model)
    model = VisionEncoderDecoderModel.from_pretrained(args.model)

    model.config.decoder_start_token_id = processor.tokenizer.cls_token_id
    model.config.pad_token_id = processor.tokenizer.pad_token_id
    model.config.vocab_size = model.config.decoder.vocab_size

    train_set = OCRLineDataset(args.train_csv, args.images_dir, processor)
    valid_set = OCRLineDataset(args.valid_csv, args.images_dir, processor)
    collator = DataCollatorOCR(processor)

    training_args = Seq2SeqTrainingArguments(
        output_dir=args.output_dir,
        evaluation_strategy="steps",
        save_strategy="steps",
        save_steps=1000,
        eval_steps=1000,
        logging_steps=100,
        learning_rate=args.lr,
        per_device_train_batch_size=args.per_device_batch_size,
        per_device_eval_batch_size=args.per_device_batch_size,
        num_train_epochs=args.epochs,
        weight_decay=0.01,
        predict_with_generate=True,
        fp16=args.fp16,
        push_to_hub=False,
        report_to=["none"],
    )

    def compute_metrics(eval_pred):
        # Optional: CER/WER 측정 가능 (여기서는 생략)
        return {}

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_set,
        eval_dataset=valid_set,
        data_collator=collator,
        tokenizer=processor.feature_extractor,
        compute_metrics=compute_metrics,
    )

    trainer.train()
    trainer.save_model(args.output_dir)
    processor.save_pretrained(args.output_dir)

if __name__ == "__main__" and False:
    main()