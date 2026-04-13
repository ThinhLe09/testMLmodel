import gc
import os

import pandas as pd
import torch
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from transformers import AutoTokenizer, Blip2Processor

from V4.config import CONFIG
from V4.dataset import ViVQAInferenceDataset
from V4.model import PhoBERT_BLIP2_VQA_Hierarchical, create_answer_type_mask

INFERENCE_CONFIG = {
    "train_csv": CONFIG['train_csv'],
    "test_csv": CONFIG['test_csv'],
    "img_dir": CONFIG['test_img_dir'],
    "model_path": CONFIG['save_path'],
    "output_file": "ket_qua_test_vivqa.csv",
    "batch_size": 16,
    "device": CONFIG['device'],
}


def run_inference_and_save():
    print(">>> Đang dọn dẹp bộ nhớ GPU...")
    gc.collect()
    torch.cuda.empty_cache()

    print(">>> Đang chuẩn bị dữ liệu cho Inference V4...")

    # 1. Rebuild LabelEncoder (khớp với tập Train)
    train_df = pd.read_csv(INFERENCE_CONFIG['train_csv'])
    train_answers = train_df['answer'].apply(lambda x: str(x).lower().strip()).unique().tolist()
    if 'unknown' not in train_answers:
        train_answers.append('unknown')

    label_encoder = LabelEncoder().fit(train_answers)
    num_classes = len(label_encoder.classes_)

    # 2. Tái tạo Mask (bắt buộc cho V4 Hierarchical)
    num_q_types = int(train_df['type'].max() + 1)
    type_map_path = CONFIG.get('type_mapping_csv', 'answer_type_mapping.csv')

    if os.path.exists(type_map_path):
        type_mask_tensor = create_answer_type_mask(type_map_path, label_encoder, num_q_types)
    else:
        print("!!! Cảnh báo: Không thấy file mapping. Tạo mask rỗng để load weight.")
        type_mask_tensor = torch.zeros((num_q_types, num_classes))

    type_mask_tensor = type_mask_tensor.to(INFERENCE_CONFIG['device'])

    # 3. Khởi tạo Model V4
    print(">>> Khởi tạo model PhoBERT_BLIP2_VQA_Hierarchical...")
    model = PhoBERT_BLIP2_VQA_Hierarchical(num_classes, num_q_types, type_mask_tensor)

    # 4. Load Weights
    if os.path.exists(INFERENCE_CONFIG['model_path']):
        print(f"Loading weights: {INFERENCE_CONFIG['model_path']}")
        state_dict = torch.load(INFERENCE_CONFIG['model_path'], map_location='cpu')
        model.load_state_dict(state_dict)
    else:
        print("!!! LỖI: Không tìm thấy file model checkpoint.")
        return

    model = model.to(INFERENCE_CONFIG['device'])
    model.eval()

    # 5. DataLoader Test
    test_df = pd.read_csv(INFERENCE_CONFIG['test_csv'])
    blip_processor = Blip2Processor.from_pretrained(CONFIG['blip_model'])
    tokenizer = AutoTokenizer.from_pretrained(CONFIG['text_model'], use_fast=False)

    test_dataset = ViVQAInferenceDataset(test_df, INFERENCE_CONFIG['img_dir'], blip_processor, tokenizer)
    test_loader = DataLoader(
        test_dataset, batch_size=INFERENCE_CONFIG['batch_size'], shuffle=False, num_workers=0
    )

    results = []
    print(">>> Bắt đầu chạy dự đoán...")

    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Testing"):
            pixel_values = batch['pixel_values'].to(INFERENCE_CONFIG['device'], dtype=torch.float16)
            input_ids = batch['input_ids'].to(INFERENCE_CONFIG['device'])
            attention_mask = batch['attention_mask'].to(INFERENCE_CONFIG['device'])

            with torch.cuda.amp.autocast(enabled=(INFERENCE_CONFIG['device'] == "cuda")):
                # V4 Inference: model tự dùng type dự đoán để áp mask
                ans_logits, type_logits = model(pixel_values, input_ids, attention_mask)

            pred_ans_indices = ans_logits.argmax(dim=1).cpu().numpy()
            pred_type_indices = type_logits.argmax(dim=1).cpu().numpy()
            pred_ans_text = label_encoder.inverse_transform(pred_ans_indices)

            batch_size = input_ids.size(0)
            for i in range(batch_size):
                results.append({
                    "ID ảnh": batch['meta_img_id'][i],
                    "Câu hỏi": batch['meta_question'][i],
                    "Đáp án (GT)": batch['meta_gt_answer'][i],
                    "Q_Type (GT)": batch['meta_gt_type'][i].item(),
                    "Q_Type Dự đoán": pred_type_indices[i],
                    "Đáp án Dự đoán": pred_ans_text[i],
                })

    # 6. Lưu kết quả
    result_df = pd.DataFrame(results)
    cols = ["ID ảnh", "Câu hỏi", "Đáp án (GT)", "Q_Type (GT)", "Q_Type Dự đoán", "Đáp án Dự đoán"]
    result_df = result_df[cols]
    result_df.to_csv(INFERENCE_CONFIG['output_file'], index=False, encoding='utf-8-sig')
    print(f"\n>>> Đã lưu kết quả: {INFERENCE_CONFIG['output_file']}")

    acc = (
        result_df["Đáp án (GT)"].astype(str).str.lower().str.strip()
        == result_df["Đáp án Dự đoán"].astype(str).str.lower().str.strip()
    ).mean()
    print(f"Độ chính xác Test: {acc:.4f}")
