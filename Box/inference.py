import gc
import os

import pandas as pd
import torch
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from transformers import AutoTokenizer, Blip2Processor

from Box.dataset import ViVQAMultiTaskInferenceDataset
from Box.model import PhoBERT_BLIP2_EfficientNet_MultiTask


def run_inference_and_save(config, output_file="ket_qua_test_multitask.csv", batch_size=16):
    print(">>> Đang dọn dẹp bộ nhớ GPU...")
    gc.collect()
    torch.cuda.empty_cache()

    print(">>> Đang chuẩn bị dữ liệu cho Inference Multi-task...")

    # 1. Rebuild LabelEncoder (khớp với tập Train)
    train_df = pd.read_csv(config['train_csv'])
    train_answers = train_df['answer'].apply(lambda x: str(x).lower().strip()).unique().tolist()
    if 'unknown' not in train_answers:
        train_answers.append('unknown')

    label_encoder = LabelEncoder().fit(train_answers)
    num_classes = len(label_encoder.classes_)

    test_df = pd.read_csv(config['test_csv'])
    num_q_types = int(max(train_df['type'].max(), test_df['type'].max()) + 1)

    # 2. Khởi tạo model đa nhiệm
    print(">>> Khởi tạo model PhoBERT_BLIP2_EfficientNet_MultiTask...")
    model = PhoBERT_BLIP2_EfficientNet_MultiTask(num_classes, num_q_types, config)

    # 3. Load weights
    if os.path.exists(config['save_path']):
        print(f"Loading weights từ: {config['save_path']}")
        state_dict = torch.load(config['save_path'], map_location='cpu')
        model.load_state_dict(state_dict)
    else:
        print("!!! LỖI: Không tìm thấy file checkpoint. Hãy chắc chắn model đã train xong.")
        return

    model = model.to(config['device'])
    model.eval()

    # 4. Chuẩn bị DataLoader Test
    blip_processor = Blip2Processor.from_pretrained(config['blip_model'])
    tokenizer = AutoTokenizer.from_pretrained(config['text_model'], use_fast=False)

    test_dataset = ViVQAMultiTaskInferenceDataset(
        test_df, config['test_img_dir'], blip_processor, tokenizer, config
    )
    num_workers = config.get('num_workers', 2)
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers
    )

    results = []
    print(">>> Bắt đầu chạy dự đoán...")

    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Testing"):
            pixel_values = batch['pixel_values'].to(config['device'], dtype=torch.float16)
            input_ids = batch['input_ids'].to(config['device'])
            attention_mask = batch['attention_mask'].to(config['device'])
            q_type = batch['q_type'].to(config['device'])

            with torch.cuda.amp.autocast(enabled=(config['device'] == "cuda")):
                ans_logits, pred_boxes = model(pixel_values, input_ids, attention_mask, q_type)

            # Xử lý kết quả VQA
            pred_ans_indices = ans_logits.argmax(dim=1).cpu().numpy()
            pred_ans_text = label_encoder.inverse_transform(pred_ans_indices)

            # Xử lý kết quả Bounding Box
            pred_boxes_np = pred_boxes.cpu().numpy()

            batch_size = input_ids.size(0)
            for i in range(batch_size):
                # Khôi phục box từ [0, 1] về pixel ảnh gốc
                w = batch['meta_orig_w'][i].item()
                h = batch['meta_orig_h'][i].item()
                box_norm = pred_boxes_np[i]
                real_box = [
                    int(box_norm[0] * w),
                    int(box_norm[1] * h),
                    int(box_norm[2] * w),
                    int(box_norm[3] * h),
                ]

                results.append({
                    "ID ảnh": batch['meta_img_id'][i],
                    "Câu hỏi": batch['meta_question'][i],
                    "Đáp án (GT)": batch['meta_gt_answer'][i],
                    "Đáp án Dự đoán": pred_ans_text[i],
                    "BBox Dự đoán (x1, y1, x2, y2)": str(real_box),
                })

    # 5. Lưu kết quả
    result_df = pd.DataFrame(results)
    cols = ["ID ảnh", "Câu hỏi", "Đáp án (GT)", "Đáp án Dự đoán", "BBox Dự đoán (x1, y1, x2, y2)"]
    result_df = result_df[cols]
    result_df.to_csv(output_file, index=False, encoding='utf-8-sig')
    print(f"\n>>> Đã lưu kết quả thành công tại: {output_file}")

    if result_df["Đáp án (GT)"].iloc[0] != "":
        acc = (
            result_df["Đáp án (GT)"].astype(str).str.lower().str.strip()
            == result_df["Đáp án Dự đoán"].astype(str).str.lower().str.strip()
        ).mean()
        print(f"Độ chính xác Test VQA: {acc:.4f}")
