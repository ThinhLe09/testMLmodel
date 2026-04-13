import torch
import torch.nn as nn
from torchvision import models
from transformers import Blip2Model, AutoModel

from Box.config import CONFIG
from utils.fusion import CrossAttentionFusion


class PhoBERT_BLIP2_EfficientNet_MultiTask(nn.Module):
    """Mô hình đa nhiệm: dự đoán câu trả lời VQA và Bounding Box.

    Kiến trúc:
        - BLIP-2 (Q-Former + Vision): trích xuất đặc trưng ảnh toàn cục.
        - EfficientNet-B7: trích xuất đặc trưng ảnh cục bộ.
        - PhoBERT: mã hóa câu hỏi tiếng Việt.
        - CrossAttentionFusion: kết hợp đặc trưng ảnh và văn bản.
        - Classifier head: phân loại câu trả lời.
        - BBox head: hồi quy tọa độ bounding box (chuẩn hóa [0, 1]).
    """

    def __init__(self, num_classes, num_q_types):
        super().__init__()

        # --- BLIP-2 ---
        print("⏳ Đang load BLIP-2...")
        tmp_blip = Blip2Model.from_pretrained(CONFIG['blip_model'], torch_dtype=torch.float16)
        self.vision_model = tmp_blip.vision_model
        self.qformer = tmp_blip.qformer
        self.query_tokens = tmp_blip.query_tokens
        del tmp_blip

        for param in self.vision_model.parameters():
            param.requires_grad = False
        for param in self.qformer.parameters():
            param.requires_grad = False
        self.query_tokens.requires_grad = False

        # --- EfficientNet-B7 ---
        print("⏳ Đang load EfficientNet-B7...")
        try:
            weights = models.EfficientNet_B7_Weights.DEFAULT
            self.efficientnet = models.efficientnet_b7(weights=weights)
        except Exception:
            self.efficientnet = models.efficientnet_b7(pretrained=True)

        self.eff_feature_extractor = self.efficientnet.features
        for param in self.efficientnet.parameters():
            param.requires_grad = False
        self.local_proj = nn.Linear(2560, 768)

        # --- PhoBERT ---
        print("⏳ Đang load PhoBERT...")
        self.phobert = AutoModel.from_pretrained(CONFIG['text_model'])

        # --- Fusion ---
        embed_dim = 768
        self.fusion = CrossAttentionFusion(visual_dim=768, text_dim=768, embed_dim=embed_dim)
        self.type_emb = nn.Embedding(num_q_types + 1, CONFIG['type_embed_dim'])

        in_features = embed_dim + CONFIG['type_embed_dim']

        self.classifier = nn.Sequential(
            nn.Linear(in_features, 512),
            nn.LayerNorm(512),
            nn.GELU(),
            nn.Dropout(0.4),
            nn.Linear(512, num_classes),
        )

        self.bbox_head = nn.Sequential(
            nn.Linear(in_features, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 4),
            nn.Sigmoid(),
        )

    def forward(self, pixel_values, input_ids, attention_mask, q_type):
        # Trích xuất đặc trưng ảnh toàn cục qua BLIP-2 Q-Former
        with torch.no_grad():
            vision_outputs = self.vision_model(pixel_values=pixel_values)
            image_embeds = vision_outputs.last_hidden_state

            image_attention_mask = torch.ones(
                image_embeds.size()[:-1], dtype=torch.long, device=image_embeds.device
            )
            query_tokens = self.query_tokens.expand(image_embeds.shape[0], -1, -1)
            query_outputs = self.qformer(
                query_embeds=query_tokens,
                encoder_hidden_states=image_embeds,
                encoder_attention_mask=image_attention_mask,
                output_attentions=False,
            )
            global_feats = query_outputs.last_hidden_state

        # Trích xuất đặc trưng ảnh cục bộ qua EfficientNet-B7
        with torch.no_grad():
            local_maps = self.eff_feature_extractor(pixel_values)
        B, C, H, W = local_maps.shape
        local_feats = local_maps.view(B, C, -1).permute(0, 2, 1)
        local_feats = self.local_proj(local_feats)

        visual_feats = torch.cat([global_feats, local_feats], dim=1)

        # Mã hóa văn bản
        text_outputs = self.phobert(input_ids=input_ids, attention_mask=attention_mask)
        text_feats = text_outputs.last_hidden_state
        visual_feats = visual_feats.to(text_feats.dtype)

        # Fusion & phân loại
        fused_vec = self.fusion(visual_feats, text_feats)
        type_vec = self.type_emb(q_type)
        final_features = torch.cat([fused_vec, type_vec], dim=-1)

        logits = self.classifier(final_features)
        pred_boxes = self.bbox_head(final_features)

        return logits, pred_boxes
