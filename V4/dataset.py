import torch
from PIL import Image
from torch.utils.data import Dataset

from V4.config import CONFIG
from utils.preprocessing import find_image_path, preprocess_vietnamese


class ViVQADataset(Dataset):
    """Dataset cho giai đoạn huấn luyện V4 (Hierarchical VQA)."""

    def __init__(self, dataframe, image_dir, blip_processor, tokenizer, label_encoder, unk_token_id):
        self.data = dataframe
        self.image_dir = image_dir
        self.blip_processor = blip_processor
        self.tokenizer = tokenizer
        self.label_encoder = label_encoder
        self.unk_token_id = unk_token_id
        self.known_classes = set(label_encoder.classes_)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]

        img_id = row.get('img_id', row.get('image'))
        img_path = find_image_path(self.image_dir, img_id)
        try:
            image = Image.open(img_path).convert("RGB") if img_path else Image.new('RGB', (224, 224))
        except Exception:
            image = Image.new('RGB', (224, 224))

        pixel_values = self.blip_processor(images=image, return_tensors="pt")['pixel_values'].squeeze(0)

        question = preprocess_vietnamese(row['question'])
        text_encoding = self.tokenizer(
            question, return_tensors="pt", padding="max_length",
            truncation=True, max_length=CONFIG['max_length'], add_special_tokens=True
        )

        answer = str(row['answer']).lower().strip()
        label = self.label_encoder.transform([answer])[0] if answer in self.known_classes else self.unk_token_id

        return {
            "pixel_values": pixel_values,
            "input_ids": text_encoding['input_ids'].squeeze(0),
            "attention_mask": text_encoding['attention_mask'].squeeze(0),
            "q_type": torch.tensor(int(row.get('type', 0)), dtype=torch.long),
            "labels": torch.tensor(label, dtype=torch.long),
        }


class ViVQAInferenceDataset(Dataset):
    """Dataset cho giai đoạn inference V4."""

    def __init__(self, dataframe, image_dir, blip_processor, tokenizer):
        self.data = dataframe
        self.image_dir = image_dir
        self.blip_processor = blip_processor
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]

        img_id = str(row.get('img_id', row.get('image')))
        question_text = row['question']
        gt_answer = str(row['answer'])
        gt_type = int(row.get('type', 0))

        img_path = find_image_path(self.image_dir, img_id)
        try:
            image = Image.open(img_path).convert("RGB") if img_path else Image.new('RGB', (224, 224))
        except Exception:
            image = Image.new('RGB', (224, 224))

        pixel_values = self.blip_processor(images=image, return_tensors="pt")['pixel_values'].squeeze(0)

        processed_q = preprocess_vietnamese(question_text)
        text_encoding = self.tokenizer(
            processed_q, return_tensors="pt", padding="max_length",
            truncation=True, max_length=CONFIG['max_length'], add_special_tokens=True
        )

        return {
            "pixel_values": pixel_values,
            "input_ids": text_encoding['input_ids'].squeeze(0),
            "attention_mask": text_encoding['attention_mask'].squeeze(0),
            "meta_img_id": img_id,
            "meta_question": question_text,
            "meta_gt_answer": gt_answer,
            "meta_gt_type": gt_type,
        }
