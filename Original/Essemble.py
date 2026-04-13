
# ==============================================================================
# CELL 4: INFERENCE PIPELINE (ENSEMBLE)
# q_type duoc lay tu ket qua du doan cua Model V4, KHONG dung tu file test!
# ==============================================================================
import pickle

# --- Dataset cho Inference ---
class ViVQAInferenceDataset(Dataset):
    def __init__(self, df, img_dir, blip_processor, tokenizer):
        self.data = df
        self.img_dir = img_dir
        self.blip_processor = blip_processor
        self.tokenizer = tokenizer

    def __len__(self): return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        img_id = row.get('img_id', row.get('image'))
        img_path = find_image_path(self.img_dir, img_id)
        try:
            img = Image.open(img_path).convert("RGB") if img_path else Image.new('RGB', (224, 224))
        except:
            img = Image.new('RGB', (224, 224))

        orig_w, orig_h = img.size
        pv = self.blip_processor(images=img, return_tensors="pt")['pixel_values'].squeeze(0)
        txt_enc = self.tokenizer(
            preprocess_vietnamese(row['question']),
            return_tensors="pt", padding="max_length", truncation=True, max_length=CONFIG['max_length']
        )
        return {
            "pv": pv,
            "ids": txt_enc['input_ids'].squeeze(0),
            "mask": txt_enc['attention_mask'].squeeze(0),
            "img_id": str(img_id),
            "q": str(row['question']),
            "gt": str(row.get('answer', '')),
            "gt_type": int(row.get('type', -1)),
            "orig_w": orig_w,
            "orig_h": orig_h
        }

def run_inference():
    print("="*60)
    print(">>> GIAI DOAN 3: INFERENCE PIPELINE (ENSEMBLE)")
    print(">>> q_type se duoc du doan boi Model V4, KHONG lay tu file test!")
    print("="*60)

    # Load test data
    test_df = pd.read_csv(CONFIG['test_csv'])
    test_loader = DataLoader(
        ViVQAInferenceDataset(test_df, CONFIG['test_img_dir'], blip_processor, tokenizer),
        batch_size=16, shuffle=False, num_workers=0
    )

    # =========================================================
    # GIAI DOAN 3a: CHAY MODEL V4 => lay predicted q_type + answer
    # =========================================================
    print("\n>>> 3a. Loading Model V4...")
    with open("v4_label_encoder.pkl", "rb") as f:
        v4_label_encoder = pickle.load(f)
    with open("v4_metadata.pkl", "rb") as f:
        v4_meta = pickle.load(f)

    v4_num_classes = v4_meta["num_classes"]
    v4_num_q_types = v4_meta["num_q_types"]

    v4_mask = create_answer_type_mask(CONFIG['type_mapping_csv'], v4_label_encoder, v4_num_q_types).to(CONFIG['device'])
    model_v4 = PhoBERT_BLIP2_VQA_Hierarchical(v4_num_classes, v4_num_q_types, v4_mask)
    model_v4.load_state_dict(torch.load(CONFIG['save_path_v4'], map_location='cpu'), strict=False)
    model_v4.to(CONFIG['device']).eval()

    v4_results = {}
    print(">>> Dang chay V4 Predict...")
    with torch.no_grad():
        for b in tqdm(test_loader, desc="V4 Predict"):
            with autocast(enabled=(CONFIG['device'] == "cuda")):
                ans_logits, type_logits = model_v4(
                    b['pv'].to(CONFIG['device'], dtype=torch.float16),
                    b['ids'].to(CONFIG['device']),
                    b['mask'].to(CONFIG['device'])
                )
            p_ans = v4_label_encoder.inverse_transform(ans_logits.argmax(dim=1).cpu().numpy())
            p_type = type_logits.argmax(dim=1).cpu().numpy()

            for i in range(len(p_ans)):
                key = b['img_id'][i] + "_" + b['q'][i]
                v4_results[key] = {"v4_ans": p_ans[i], "v4_pred_type": int(p_type[i])}

    clear_memory(model_v4)

    # =========================================================
    # GIAI DOAN 3b: CHAY MODEL BOX voi q_type tu V4
    # =========================================================
    print(">>> 3b. Loading Model Box...")
    with open("box_label_encoder.pkl", "rb") as f:
        box_label_encoder = pickle.load(f)
    with open("box_metadata.pkl", "rb") as f:
        box_meta = pickle.load(f)

    box_num_classes = box_meta["num_classes"]
    box_num_q_types = box_meta["num_q_types"]

    model_box = PhoBERT_BLIP2_EfficientNet_MultiTask(box_num_classes, box_num_q_types)
    model_box.load_state_dict(torch.load(CONFIG['save_path_box'], map_location='cpu'), strict=False)
    model_box.to(CONFIG['device']).eval()

    box_results = {}
    print(">>> Dang chay Box Predict (dung q_type tu V4)...")
    with torch.no_grad():
        for b in tqdm(test_loader, desc="Box Predict"):
            # === LAY q_type TU KET QUA V4, KHONG DUNG TU FILE TEST ===
            batch_q_types = []
            for i in range(len(b['img_id'])):
                key = b['img_id'][i] + "_" + b['q'][i]
                v4_data = v4_results.get(key, {"v4_pred_type": 0})
                batch_q_types.append(v4_data["v4_pred_type"])
            q_type_tensor = torch.tensor(batch_q_types, dtype=torch.long).to(CONFIG['device'])

            with autocast(enabled=(CONFIG['device'] == "cuda")):
                ans_logits, box_preds = model_box(
                    b['pv'].to(CONFIG['device'], dtype=torch.float16),
                    b['ids'].to(CONFIG['device']),
                    b['mask'].to(CONFIG['device']),
                    q_type_tensor
                )
            p_ans = box_label_encoder.inverse_transform(ans_logits.argmax(dim=1).cpu().numpy())
            box_np = box_preds.cpu().numpy()

            for i in range(len(p_ans)):
                key = b['img_id'][i] + "_" + b['q'][i]
                w, h = b['orig_w'][i].item(), b['orig_h'][i].item()
                real_box = [int(box_np[i][0]*w), int(box_np[i][1]*h), int(box_np[i][2]*w), int(box_np[i][3]*h)]
                box_results[key] = {"box_ans": p_ans[i], "box_coords": str(real_box)}

    clear_memory(model_box)

    # =========================================================
    # GIAI DOAN 3c: KET HOP KET QUA (ROUTING ENSEMBLE)
    # =========================================================
    print("="*50)
    print(">>> GIAI DOAN 3c: KET HOP KET QUA (ROUTING ENSEMBLE)")
    print("="*50)

    final_results = []
    for _, row in test_df.iterrows():
        img_id = str(row.get('img_id', row.get('image')))
        q = str(row['question'])
        key = img_id + "_" + q

        gt_ans = str(row.get('answer', ''))
        gt_type = int(row.get('type', -1))

        v4_data = v4_results.get(key, {"v4_ans": "", "v4_pred_type": -1})
        box_data = box_results.get(key, {"box_ans": "", "box_coords": ""})

        # ROUTING: Neu V4 du doan type==1 -> dung V4, con lai dung Box
        predicted_type = v4_data["v4_pred_type"]
        final_ans = v4_data["v4_ans"] if predicted_type == 1 else box_data["box_ans"]

        final_results.append({
            "img_id": img_id,
            "question": q,
            "ground_truth": gt_ans,
            "q_type_gt": gt_type,
            "v4_pred_type": predicted_type,
            "v4_ans": v4_data["v4_ans"],
            "box_ans": box_data["box_ans"],
            "final_prediction": final_ans,
            "box_coords": box_data["box_coords"]
        })

    df_res = pd.DataFrame(final_results)
    df_res.to_csv(CONFIG['output_file'], index=False, encoding='utf-8-sig')
    print(f"\n>>> HOAN TAT! Da luu ket qua tai: {CONFIG['output_file']}")

    # --- DANH GIA ---
    if df_res['ground_truth'].str.len().mean() > 0:
        gt_clean = df_res['ground_truth'].str.lower().str.strip()
        df_res['v4_correct'] = df_res['v4_ans'].str.lower().str.strip() == gt_clean
        df_res['box_correct'] = df_res['box_ans'].str.lower().str.strip() == gt_clean
        df_res['ens_correct'] = df_res['final_prediction'].str.lower().str.strip() == gt_clean

        print("\n" + "=" * 50)
        print("TONG QUAN ACCURACY (OVERALL)")
        print("=" * 50)
        print(f"V4 accuracy:       {df_res['v4_correct'].mean():.4f}")
        print(f"Box accuracy:      {df_res['box_correct'].mean():.4f}")
        print(f"ENSEMBLE accuracy: {df_res['ens_correct'].mean():.4f}")

        print("\n" + "=" * 50)
        print("ACCURACY THEO TUNG Q_TYPE (GT)")
        print("=" * 50)
        grouped_acc = df_res.groupby('q_type_gt')[['v4_correct', 'box_correct', 'ens_correct']].mean()
        counts = df_res.groupby('q_type_gt').size()
        for q_type, row in grouped_acc.iterrows():
            count = counts[q_type]
            print(f"--- Q_TYPE: {q_type} (So luong mau: {count}) ---")
            print(f"  V4 accuracy:       {row['v4_correct']:.4f}")
            print(f"  Box accuracy:      {row['box_correct']:.4f}")
            print(f"  ENSEMBLE accuracy: {row['ens_correct']:.4f}")
            print("-" * 30)

        # Q-type prediction accuracy
        print("\n" + "=" * 50)
        print("DO CHINH XAC DU DOAN Q_TYPE (V4 vs GT)")
        print("=" * 50)
        valid_gt = df_res[df_res['q_type_gt'] >= 0]
        if len(valid_gt) > 0:
            type_acc = (valid_gt['v4_pred_type'] == valid_gt['q_type_gt']).mean()
            print(f"Q-Type prediction accuracy: {type_acc:.4f}")

run_inference()