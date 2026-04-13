import torch
import torch.nn as nn
import pandas as pd
from transformers import Blip2Model, AutoModel

from utils.fusion import CrossAttentionFusion


def create_answer_type_mask(csv_path, label_encoder, num_q_types):
    """Tạo ma trận mask [num_q_types, num_classes].

    - Giá trị 0.0  : Cho phép (Allow) — đáp án hợp lệ cho loại câu hỏi đó.
    - Giá trị -inf : Chặn (Block)    — đáp án không hợp lệ.
    """
    print(f"Loading mask from {csv_path}...")
    try:
        df = pd.read_csv(csv_path)
    except Exception as e:
        print(f"Error reading mapping file: {e}")
        print("Creating a dummy mask (Allow all) to prevent crash.")
        return torch.zeros((num_q_types, len(label_encoder.classes_)))

    num_classes = len(label_encoder.classes_)
    # Mặc định chặn tất cả
    mask = torch.full((num_q_types, num_classes), float('-inf'))

    count_mapped = 0
    for _, row in df.iterrows():
        ans_text = str(row['answer']).lower().strip()
        try:
            q_type = int(row.get('# type', row.get('type', 0)))
        except Exception:
            continue

        if ans_text in label_encoder.classes_:
            idx = label_encoder.transform([ans_text])[0]
            if 0 <= q_type < num_q_types:
                mask[q_type, idx] = 0.0
                count_mapped += 1

    # Safety net: nếu một loại câu hỏi không có đáp án nào được map, mở khóa toàn bộ
    for i in range(num_q_types):
        if torch.max(mask[i]) == float('-inf'):
            print(f"Warning: Type {i} has no mapped answers. Unmasking all for this type.")
            mask[i] = 0.0

    print(f"Mask created. Mapped {count_mapped} rules.")
    return mask


class PhoBERT_BLIP2_VQA_Hierarchical(nn.Module):
    """Mô hình VQA phân cấp V4 với Hierarchical Masking theo loại câu hỏi.

    Kiến trúc:
        - BLIP-2 Vision Encoder: trích xuất đặc trưng ảnh thô (dim=1408).
        - PhoBERT: mã hóa câu hỏi tiếng Việt.
        - CrossAttentionFusion: kết hợp đặc trưng ảnh và văn bản.
        - Type classifier head: dự đoán loại câu hỏi.
        - Answer classifier head + Type Mask: phân loại câu trả lời có kiểm soát.
    """

    def __init__(self, num_classes, num_q_types, answer_type_mask, config):
        super().__init__()

        # Image Encoder (chỉ dùng vision encoder, không qua Q-Former)
        self.blip = Blip2Model.from_pretrained(config['blip_model'], torch_dtype=torch.float16)
        for param in self.blip.parameters():
            param.requires_grad = False

        # Text Encoder
        self.phobert = AutoModel.from_pretrained(config['text_model'])

        # Fusion (visual_dim=1408 vì dùng raw vision features)
        embed_dim = 768
        self.fusion = CrossAttentionFusion(visual_dim=1408, text_dim=768, embed_dim=embed_dim)

        # Answer classifier head
        self.classifier = nn.Sequential(
            nn.Linear(embed_dim, 512),
            nn.LayerNorm(512),
            nn.GELU(),
            nn.Dropout(0.4),
            nn.Linear(512, num_classes),
        )

        # Question type classifier head
        self.type_classifier = nn.Sequential(
            nn.Linear(embed_dim, 256),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(256, num_q_types),
        )

        # Đăng ký mask như buffer (không tính gradient, nhưng lưu trong state_dict)
        self.register_buffer('type_mask', answer_type_mask)

    def forward(self, pixel_values, input_ids, attention_mask, target_q_type=None):
        # Trích xuất đặc trưng ảnh
        with torch.no_grad():
            vision_outputs = self.blip.vision_model(pixel_values=pixel_values)
            visual_feats = vision_outputs.last_hidden_state

        # Mã hóa văn bản
        text_outputs = self.phobert(input_ids=input_ids, attention_mask=attention_mask)
        text_feats = text_outputs.last_hidden_state

        # Fusion
        fused_vec = self.fusion(visual_feats.to(text_feats.dtype), text_feats)

        # Dự đoán loại câu hỏi và câu trả lời
        type_logits = self.type_classifier(fused_vec)
        raw_ans_logits = self.classifier(fused_vec)

        # Hierarchical Masking
        if self.training and target_q_type is not None:
            # Training: dùng Ground Truth q_type để áp mask
            indices = target_q_type
        else:
            # Inference/Validation: dùng q_type do model dự đoán
            indices = torch.argmax(type_logits, dim=1)

        batch_mask = self.type_mask[indices]
        masked_ans_logits = raw_ans_logits + batch_mask

        return masked_ans_logits, type_logits
