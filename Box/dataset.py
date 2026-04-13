import ast

import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset

from utils.preprocessing import find_image_path, preprocess_vietnamese


class ViVQADataset(Dataset):
    """Dataset cho giai đoạn huấn luyện đa nhiệm (VQA + BBox)."""

    def __init__(self, dataframe, image_dir, blip_processor, tokenizer, label_encoder, unk_token_id, config):
        self.data = dataframe
        self.image_dir = image_dir
        self.blip_processor = blip_processor
        self.tokenizer = tokenizer
        self.label_encoder = label_encoder
        self.unk_token_id = unk_token_id
        self.known_classes = set(label_encoder.classes_)
        self.config = config

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]

        # 1. XỬ LÝ ẢNH
        img_id = row.get('img_id', row.get('image'))
        img_path = find_image_path(self.image_dir, img_id)
        try:
            image = Image.open(img_path).convert("RGB") if img_path else Image.new('RGB', (224, 224))
        except Exception:
            image = Image.new('RGB', (224, 224))

        orig_w, orig_h = image.size
        pixel_values = self.blip_processor(images=image, return_tensors="pt")['pixel_values'].squeeze(0)

        # 2. XỬ LÝ TỌA ĐỘ BOX (TARGET CHO MULTI-TASK)
        target_box = torch.zeros(4, dtype=torch.float32)
        has_box = torch.tensor(0.0, dtype=torch.float32)

        box_str = row.get('merged_box', None)
        if pd.notna(box_str):
            try:
                box = ast.literal_eval(str(box_str))
                xmin, ymin, xmax, ymax = box
                target_box[0] = xmin / orig_w
                target_box[1] = ymin / orig_h
                target_box[2] = xmax / orig_w
                target_box[3] = ymax / orig_h
                target_box = torch.clamp(target_box, 0.0, 1.0)
                has_box = torch.tensor(1.0, dtype=torch.float32)
            except Exception:
                pass

        # 3. XỬ LÝ TEXT & LABEL
        question = preprocess_vietnamese(row['question'])
        text_encoding = self.tokenizer(
            question, return_tensors="pt", padding="max_length",
            truncation=True, max_length=self.config['max_length'], add_special_tokens=True
        )

        answer = str(row['answer']).lower().strip()
        label = self.label_encoder.transform([answer])[0] if answer in self.known_classes else self.unk_token_id

        return {
            "pixel_values": pixel_values,
            "input_ids": text_encoding['input_ids'].squeeze(0),
            "attention_mask": text_encoding['attention_mask'].squeeze(0),
            "q_type": torch.tensor(int(row.get('type', 0)), dtype=torch.long),
            "labels": torch.tensor(label, dtype=torch.long),
            "target_box": target_box,
            "has_box": has_box,
        }


class ViVQAMultiTaskInferenceDataset(Dataset):
    """Dataset cho giai đoạn inference đa nhiệm."""

    def __init__(self, dataframe, image_dir, blip_processor, tokenizer, config):
        self.data = dataframe
        self.image_dir = image_dir
        self.blip_processor = blip_processor
        self.tokenizer = tokenizer
        self.config = config

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]

        img_id = str(row.get('img_id', row.get('image')))
        question_text = str(row['question'])
        gt_answer = str(row['answer']) if 'answer' in row else ""
        q_type = int(row.get('type', 0))

        img_path = find_image_path(self.image_dir, img_id)
        try:
            image = Image.open(img_path).convert("RGB") if img_path else Image.new('RGB', (224, 224))
        except Exception:
            image = Image.new('RGB', (224, 224))

        orig_w, orig_h = image.size
        pixel_values = self.blip_processor(images=image, return_tensors="pt")['pixel_values'].squeeze(0)

        processed_q = preprocess_vietnamese(question_text)
        text_encoding = self.tokenizer(
            processed_q, return_tensors="pt", padding="max_length",
            truncation=True, max_length=self.config['max_length'], add_special_tokens=True
        )

        return {
            "pixel_values": pixel_values,
            "input_ids": text_encoding['input_ids'].squeeze(0),
            "attention_mask": text_encoding['attention_mask'].squeeze(0),
            "q_type": torch.tensor(q_type, dtype=torch.long),
            "meta_img_id": img_id,
            "meta_question": question_text,
            "meta_gt_answer": gt_answer,
            "meta_orig_w": orig_w,
            "meta_orig_h": orig_h,
        }
