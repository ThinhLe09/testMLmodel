import torch
import torch.nn as nn
import pandas as pd
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from transformers import Blip2Processor, AutoTokenizer, get_linear_schedule_with_warmup
from torch.cuda.amp import autocast, GradScaler

from V4.config import CONFIG
from V4.dataset import ViVQADataset
from V4.model import PhoBERT_BLIP2_VQA_Hierarchical, create_answer_type_mask
from utils.early_stopping import EarlyStopping


def train():
    print(f"Device: {CONFIG['device']}")
    blip_processor = Blip2Processor.from_pretrained(CONFIG['blip_model'])
    tokenizer = AutoTokenizer.from_pretrained(CONFIG['text_model'], use_fast=False)

    # --- Load và split dữ liệu ---
    print(">>> Đang tải dữ liệu Train gốc...")
    full_train_df = pd.read_csv(CONFIG['train_csv'])

    # Tách 10% tập train gốc làm Validation, stratify theo 'type'
    try:
        train_df, val_df = train_test_split(
            full_train_df, test_size=0.1, random_state=42, stratify=full_train_df['type']
        )
    except Exception:
        train_df, val_df = train_test_split(full_train_df, test_size=0.1, random_state=42)

    print(f"Train size: {len(train_df)} | Validation size (Split): {len(val_df)}")

    # --- Encode Answers (trên toàn bộ tập gốc để đủ class) ---
    all_answers = full_train_df['answer'].apply(lambda x: str(x).lower().strip()).unique().tolist()
    if 'unknown' not in all_answers:
        all_answers.append('unknown')

    label_encoder = LabelEncoder().fit(all_answers)
    unk_token_id = label_encoder.transform(['unknown'])[0]

    # --- DataLoaders ---
    train_loader = DataLoader(
        ViVQADataset(train_df, CONFIG['train_img_dir'], blip_processor, tokenizer, label_encoder, unk_token_id),
        batch_size=CONFIG['batch_size'], shuffle=True, num_workers=0, pin_memory=True,
    )
    # Val dùng ảnh từ train_img_dir vì được tách từ train ra
    val_loader = DataLoader(
        ViVQADataset(val_df, CONFIG['train_img_dir'], blip_processor, tokenizer, label_encoder, unk_token_id),
        batch_size=CONFIG['batch_size'], shuffle=False, num_workers=0, pin_memory=True,
    )

    # --- Tạo Mask (đặc trưng V4) ---
    num_q_types = int(full_train_df['type'].max() + 1)
    type_mask_tensor = create_answer_type_mask(CONFIG['type_mapping_csv'], label_encoder, num_q_types)
    type_mask_tensor = type_mask_tensor.to(CONFIG['device'])

    # --- Khởi tạo Model ---
    print("Initializing Hierarchical Model V4...")
    model = PhoBERT_BLIP2_VQA_Hierarchical(
        len(label_encoder.classes_), num_q_types, type_mask_tensor
    ).to(CONFIG['device'])

    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=CONFIG['lr'], weight_decay=CONFIG['weight_decay'],
    )
    total_steps = len(train_loader) * CONFIG['epochs']
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(0.1 * total_steps),
        num_training_steps=total_steps,
    )

    # Loss: per-sample với reduction='none' để áp trọng số theo q_type
    criterion_ans = nn.CrossEntropyLoss(reduction='none', label_smoothing=0.1)
    criterion_type = nn.CrossEntropyLoss()

    type_weight_tensor = torch.ones(num_q_types, device=CONFIG['device'])
    for t, w in CONFIG['type_loss_weights'].items():
        if t < num_q_types:
            type_weight_tensor[t] = w

    scaler = GradScaler()
    early_stopping = EarlyStopping(patience=CONFIG['patience'], path=CONFIG['save_path'])

    print(f"Type loss weights: {CONFIG['type_loss_weights']}")
    print("Bắt đầu huấn luyện...")

    for epoch in range(CONFIG['epochs']):
        # --- TRAIN ---
        model.train()
        train_ans_correct, train_type_correct, train_total = 0, 0, 0
        train_type_ans_correct = {t: 0 for t in range(num_q_types)}
        train_type_ans_total = {t: 0 for t in range(num_q_types)}
        train_loss_avg = 0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch + 1}")
        for batch in pbar:
            pixel_values = batch['pixel_values'].to(CONFIG['device'], dtype=torch.float16)
            input_ids = batch['input_ids'].to(CONFIG['device'])
            attention_mask = batch['attention_mask'].to(CONFIG['device'])
            labels = batch['labels'].to(CONFIG['device'])
            q_type_labels = batch['q_type'].to(CONFIG['device'])

            optimizer.zero_grad()
            with autocast():
                ans_logits, type_logits = model(
                    pixel_values, input_ids, attention_mask, target_q_type=q_type_labels
                )

                # Per-sample loss nhân trọng số theo q_type
                per_sample_loss = criterion_ans(ans_logits, labels)
                sample_weights = type_weight_tensor[q_type_labels]
                loss_ans = (per_sample_loss * sample_weights).mean()

                loss_type = criterion_type(type_logits, q_type_labels)
                total_loss = loss_ans + (CONFIG['lambda_type'] * loss_type)

            scaler.scale(total_loss).backward()
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

            train_loss_avg += total_loss.item()
            preds = ans_logits.argmax(1)
            train_ans_correct += (preds == labels).sum().item()
            train_type_correct += (type_logits.argmax(1) == q_type_labels).sum().item()
            train_total += labels.size(0)

            for t in range(num_q_types):
                mask = (q_type_labels == t)
                if mask.sum() > 0:
                    train_type_ans_correct[t] += (preds[mask] == labels[mask]).sum().item()
                    train_type_ans_total[t] += mask.sum().item()

            pbar.set_postfix({'Loss': f'{total_loss.item():.2f}'})

        # --- VALIDATION ---
        model.eval()
        val_ans_correct, val_type_correct, val_total = 0, 0, 0
        val_type_ans_correct = {t: 0 for t in range(num_q_types)}
        val_type_ans_total = {t: 0 for t in range(num_q_types)}

        with torch.no_grad(), autocast():
            for batch in val_loader:
                pixel_values = batch['pixel_values'].to(CONFIG['device'], dtype=torch.float16)
                input_ids = batch['input_ids'].to(CONFIG['device'])
                attention_mask = batch['attention_mask'].to(CONFIG['device'])
                labels = batch['labels'].to(CONFIG['device'])
                q_type_labels = batch['q_type'].to(CONFIG['device'])

                ans_logits, type_logits = model(pixel_values, input_ids, attention_mask)

                preds = ans_logits.argmax(1)
                val_ans_correct += (preds == labels).sum().item()
                val_type_correct += (type_logits.argmax(1) == q_type_labels).sum().item()
                val_total += labels.size(0)

                for t in range(num_q_types):
                    mask = (q_type_labels == t)
                    if mask.sum() > 0:
                        val_type_ans_correct[t] += (preds[mask] == labels[mask]).sum().item()
                        val_type_ans_total[t] += mask.sum().item()

        val_ans_acc = val_ans_correct / val_total
        val_type_acc = val_type_correct / val_total

        print(f"\n>>> Epoch {epoch + 1} Summary:")
        print(f"    Train Loss: {train_loss_avg / len(train_loader):.4f}")
        print(f"    Train Ans Acc: {train_ans_correct / train_total:.4f} | Val Ans Acc: {val_ans_acc:.4f}")
        print(f"    Train Type Acc: {train_type_correct / train_total:.4f} | Val Type Acc: {val_type_acc:.4f}")

        print("    --- Per Q-Type Answer Accuracy ---")
        for t in range(num_q_types):
            tr_acc = train_type_ans_correct[t] / train_type_ans_total[t] if train_type_ans_total[t] > 0 else 0
            vl_acc = val_type_ans_correct[t] / val_type_ans_total[t] if val_type_ans_total[t] > 0 else 0
            w = CONFIG['type_loss_weights'].get(t, 1.0)
            marker = " <<<" if w > 1.0 else ""
            print(f"    Type {t} (w={w}): Train={tr_acc:.4f} | Val={vl_acc:.4f}{marker}")

        # Early stopping theo accuracy của q_type=1
        val_type_1_acc = val_type_ans_correct[1] / val_type_ans_total[1] if val_type_ans_total[1] > 0 else 0
        print(f"    => Đang theo dõi Early Stopping theo q_type=1 Val Acc: {val_type_1_acc:.4f}")

        early_stopping(val_type_1_acc, model)
        if early_stopping.early_stop:
            print("Early stopping triggered dựa trên điểm số của q_type=1!")
            break
