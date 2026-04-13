import gc
import os

import pandas as pd
import torch
from PIL import Image
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm
from transformers import AutoTokenizer, Blip2Processor

from Box.model import PhoBERT_BLIP2_EfficientNet_MultiTask
from V4.model import PhoBERT_BLIP2_VQA_Hierarchical, create_answer_type_mask
from utils.preprocessing import find_image_path, preprocess_vietnamese


class EnsembleInferenceDataset(Dataset):
    """Dataset dùng chung cho cả hai model trong pipeline Ensemble."""

    def __init__(self, dataframe, image_dir, blip_processor, tokenizer, max_length):
        self.data = dataframe
        self.image_dir = image_dir
        self.blip_processor = blip_processor
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        img_id = str(row.get('img_id', row.get('image')))
        question_text = str(row['question'])
        gt_answer = str(row.get('answer', ''))
        gt_type = int(row.get('type', -1))

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
            truncation=True, max_length=self.max_length, add_special_tokens=True,
        )

        return {
            "pixel_values": pixel_values,
            "input_ids": text_encoding['input_ids'].squeeze(0),
            "attention_mask": text_encoding['attention_mask'].squeeze(0),
            "meta_img_id": img_id,
            "meta_question": question_text,
            "meta_gt_answer": gt_answer,
            "meta_gt_type": gt_type,
            "meta_orig_w": orig_w,
            "meta_orig_h": orig_h,
        }


def _build_label_encoder(train_csv):
    """Khôi phục LabelEncoder và num_q_types từ CSV huấn luyện."""
    train_df = pd.read_csv(train_csv)
    answers = train_df['answer'].apply(lambda x: str(x).lower().strip()).unique().tolist()
    if 'unknown' not in answers:
        answers.append('unknown')
    label_encoder = LabelEncoder().fit(answers)
    num_q_types = int(train_df['type'].max() + 1)
    return label_encoder, num_q_types, train_df


def run_ensemble_inference(
    box_config,
    v4_config,
    output_file="ket_qua_ensemble.csv",
    batch_size=32,
):
    """Pipeline inference Ensemble 3 bước:

    1. Model V4 dự đoán **q_type** và câu trả lời.
    2. Model Box nhận **q_type từ V4** (không dùng ground truth) để dự đoán
       câu trả lời + bounding box.
    3. Routing: q_type == 1 → dùng V4 answer; còn lại → dùng Box answer.
    """
    device = box_config['device']
    gc.collect()
    torch.cuda.empty_cache()

    # --- Chuẩn bị DataLoader test (dùng chung cho cả 2 model) ---
    print(">>> Đang chuẩn bị DataLoader test...")
    test_df = pd.read_csv(v4_config['test_csv'])
    blip_processor = Blip2Processor.from_pretrained(v4_config['blip_model'])
    tokenizer = AutoTokenizer.from_pretrained(v4_config['text_model'], use_fast=False)
    num_workers = box_config.get('num_workers', 0)

    test_dataset = EnsembleInferenceDataset(
        test_df, v4_config['test_img_dir'], blip_processor, tokenizer, v4_config['max_length']
    )
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    # =========================================================
    # BƯỚC 1: MODEL V4 → dự đoán q_type + câu trả lời
    # =========================================================
    print("\n>>> Bước 1: Load Model V4...")
    v4_le, v4_num_q_types, _ = _build_label_encoder(v4_config['train_csv'])
    v4_num_classes = len(v4_le.classes_)

    type_mask = create_answer_type_mask(v4_config['type_mapping_csv'], v4_le, v4_num_q_types)
    type_mask = type_mask.to(device)
    model_v4 = PhoBERT_BLIP2_VQA_Hierarchical(v4_num_classes, v4_num_q_types, type_mask, v4_config)

    if not os.path.exists(v4_config['save_path']):
        print(f"!!! LỖI: Không tìm thấy checkpoint V4: {v4_config['save_path']}")
        return
    model_v4.load_state_dict(torch.load(v4_config['save_path'], map_location='cpu'))
    model_v4.to(device).eval()

    v4_results = {}
    print(">>> Đang chạy V4 predict...")
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="V4 Predict"):
            pv = batch['pixel_values'].to(device, dtype=torch.float16)
            ids = batch['input_ids'].to(device)
            mask_t = batch['attention_mask'].to(device)

            with torch.cuda.amp.autocast(enabled=(device == "cuda")):
                ans_logits, type_logits = model_v4(pv, ids, mask_t)

            pred_ans = v4_le.inverse_transform(ans_logits.argmax(dim=1).cpu().numpy())
            pred_type = type_logits.argmax(dim=1).cpu().numpy()

            for i in range(len(pred_ans)):
                key = batch['meta_img_id'][i] + "||" + batch['meta_question'][i]
                v4_results[key] = {
                    "v4_ans": pred_ans[i],
                    "v4_pred_type": int(pred_type[i]),
                }

    del model_v4
    gc.collect()
    torch.cuda.empty_cache()

    # =========================================================
    # BƯỚC 2: MODEL BOX → câu trả lời + bounding box
    #         q_type lấy từ V4, KHÔNG lấy từ ground truth
    # =========================================================
    print("\n>>> Bước 2: Load Model Box...")
    box_le, box_num_q_types, _ = _build_label_encoder(box_config['train_csv'])
    box_num_classes = len(box_le.classes_)

    model_box = PhoBERT_BLIP2_EfficientNet_MultiTask(box_num_classes, box_num_q_types, box_config)

    if not os.path.exists(box_config['save_path']):
        print(f"!!! LỖI: Không tìm thấy checkpoint Box: {box_config['save_path']}")
        return
    model_box.load_state_dict(torch.load(box_config['save_path'], map_location='cpu'))
    model_box.to(device).eval()

    box_results = {}
    print(">>> Đang chạy Box predict (dùng q_type từ V4)...")
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Box Predict"):
            pv = batch['pixel_values'].to(device, dtype=torch.float16)
            ids = batch['input_ids'].to(device)
            mask_t = batch['attention_mask'].to(device)

            # Lấy q_type từ kết quả V4 (không dùng ground truth)
            batch_q_types = [
                v4_results.get(
                    batch['meta_img_id'][i] + "||" + batch['meta_question'][i],
                    {"v4_pred_type": 0}
                )["v4_pred_type"]
                for i in range(len(batch['meta_img_id']))
            ]
            q_type_tensor = torch.tensor(batch_q_types, dtype=torch.long).to(device)

            with torch.cuda.amp.autocast(enabled=(device == "cuda")):
                ans_logits, box_preds = model_box(pv, ids, mask_t, q_type_tensor)

            pred_ans = box_le.inverse_transform(ans_logits.argmax(dim=1).cpu().numpy())
            box_np = box_preds.cpu().numpy()

            for i in range(len(pred_ans)):
                key = batch['meta_img_id'][i] + "||" + batch['meta_question'][i]
                w = batch['meta_orig_w'][i].item()
                h = batch['meta_orig_h'][i].item()
                real_box = [
                    int(box_np[i][0] * w), int(box_np[i][1] * h),
                    int(box_np[i][2] * w), int(box_np[i][3] * h),
                ]
                box_results[key] = {
                    "box_ans": pred_ans[i],
                    "box_coords": str(real_box),
                }

    del model_box
    gc.collect()
    torch.cuda.empty_cache()

    # =========================================================
    # BƯỚC 3: ROUTING ENSEMBLE
    #   q_type đã dự đoán bởi V4 (KHÔNG dùng ground truth)
    #   type == 1  →  V4 answer
    #   còn lại    →  Box answer
    # =========================================================
    print("\n>>> Bước 3: Routing Ensemble (type==1 → V4, còn lại → Box)...")
    final_results = []
    for _, row in test_df.iterrows():
        img_id = str(row.get('img_id', row.get('image')))
        q = str(row['question'])
        key = img_id + "||" + q

        v4_data = v4_results.get(key, {"v4_ans": "", "v4_pred_type": -1})
        box_data = box_results.get(key, {"box_ans": "", "box_coords": ""})

        pred_type = v4_data["v4_pred_type"]
        final_ans = v4_data["v4_ans"] if pred_type == 1 else box_data["box_ans"]

        final_results.append({
            "img_id": img_id,
            "question": q,
            "ground_truth": str(row.get('answer', '')),
            "q_type_gt": int(row.get('type', -1)),
            "v4_pred_type": pred_type,
            "v4_ans": v4_data["v4_ans"],
            "box_ans": box_data["box_ans"],
            "final_prediction": final_ans,
            "box_coords": box_data["box_coords"],
        })

    # Lưu kết quả
    df_res = pd.DataFrame(final_results)
    cols = [
        "img_id", "question", "ground_truth", "q_type_gt",
        "v4_pred_type", "v4_ans", "box_ans", "final_prediction", "box_coords",
    ]
    df_res = df_res[cols]
    df_res.to_csv(output_file, index=False, encoding='utf-8-sig')
    print(f"\n>>> Đã lưu kết quả ensemble tại: {output_file}")

    # --- Đánh giá ---
    gt_clean = df_res['ground_truth'].astype(str).str.lower().str.strip()
    df_res['v4_correct'] = df_res['v4_ans'].str.lower().str.strip() == gt_clean
    df_res['box_correct'] = df_res['box_ans'].str.lower().str.strip() == gt_clean
    df_res['ens_correct'] = df_res['final_prediction'].str.lower().str.strip() == gt_clean

    print("\n" + "=" * 50)
    print("TỔNG QUAN ACCURACY")
    print("=" * 50)
    print(f"  V4  accuracy:       {df_res['v4_correct'].mean():.4f}")
    print(f"  Box accuracy:       {df_res['box_correct'].mean():.4f}")
    print(f"  Ensemble accuracy:  {df_res['ens_correct'].mean():.4f}")

    print("\nACCURACY THEO TỪNG Q_TYPE (Ground Truth)")
    print("=" * 50)
    for qt, grp in df_res.groupby('q_type_gt'):
        print(
            f"  Type {qt} (n={len(grp):4d}): "
            f"V4={grp['v4_correct'].mean():.4f} | "
            f"Box={grp['box_correct'].mean():.4f} | "
            f"Ens={grp['ens_correct'].mean():.4f}"
        )

    valid_gt = df_res[df_res['q_type_gt'] >= 0]
    if len(valid_gt) > 0:
        type_acc = (valid_gt['v4_pred_type'] == valid_gt['q_type_gt']).mean()
        print(f"\n  Q-Type prediction accuracy (V4): {type_acc:.4f}")
