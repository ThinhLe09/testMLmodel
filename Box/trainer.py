import torch
import torch.nn as nn
import pandas as pd
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from sklearn.preprocessing import LabelEncoder
from transformers import Blip2Processor, AutoTokenizer, get_linear_schedule_with_warmup
from torch.cuda.amp import autocast, GradScaler

from Box.dataset import ViVQADataset
from Box.model import PhoBERT_BLIP2_EfficientNet_MultiTask
from utils.early_stopping import EarlyStopping


def train(config):
    blip_processor = Blip2Processor.from_pretrained(config['blip_model'])
    tokenizer = AutoTokenizer.from_pretrained(config['text_model'], use_fast=False)

    train_df = pd.read_csv(config['train_csv'])
    val_df = pd.read_csv(config['test_csv'])

    print("⏳ Đang chuẩn bị dữ liệu...")

    # Chỉ load Box cho tập Train
    df_box_train = pd.read_csv(config['train_box_csv'])

    # Chuẩn hóa tên cột question_id
    for df in [train_df, val_df, df_box_train]:
        if 'Unnamed: 0' in df.columns:
            df.rename(columns={'Unnamed: 0': 'question_id'}, inplace=True)
        elif df.columns[0] == 'Unnamed: 0' or 'question_id' not in df.columns:
            df.columns = ['question_id'] + list(df.columns[1:])

    # Gộp dữ liệu Box vào train_df. val_df sẽ không có cột owl_box
    train_df = pd.merge(train_df, df_box_train[['question_id', 'owl_box']], on='question_id', how='left')

    train_answers = train_df['answer'].apply(lambda x: str(x).lower().strip()).unique().tolist()
    if 'unknown' not in train_answers:
        train_answers.append('unknown')

    label_encoder = LabelEncoder().fit(train_answers)
    unk_token_id = label_encoder.transform(['unknown'])[0]

    num_workers = config.get('num_workers', 0)
    train_loader = DataLoader(
        ViVQADataset(train_df, config['train_img_dir'], blip_processor, tokenizer, label_encoder, unk_token_id, config),
        batch_size=config['batch_size'], shuffle=True, num_workers=num_workers, pin_memory=True,
    )
    val_loader = DataLoader(
        ViVQADataset(val_df, config['test_img_dir'], blip_processor, tokenizer, label_encoder, unk_token_id, config),
        batch_size=config['batch_size'], shuffle=False, num_workers=num_workers, pin_memory=True,
    )

    num_q_types = int(max(train_df['type'].max(), val_df['type'].max()) + 1)
    model = PhoBERT_BLIP2_EfficientNet_MultiTask(
        len(label_encoder.classes_), num_q_types, config
    ).to(config['device'])

    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=config['lr'], weight_decay=config['weight_decay'],
    )
    total_steps = len(train_loader) * config['epochs']
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(0.1 * total_steps),
        num_training_steps=total_steps,
    )

    # Weighted per-sample VQA loss (reduction='none')
    criterion_vqa = nn.CrossEntropyLoss(reduction='none', label_smoothing=0.1)
    criterion_bbox = nn.SmoothL1Loss(reduction='none')

    # Tạo weight lookup tensor trên GPU
    type_weight_tensor = torch.ones(num_q_types, device=config['device'])
    for t, w in config['type_loss_weights'].items():
        if t < num_q_types:
            type_weight_tensor[t] = w

    scaler = GradScaler()
    early_stopping = EarlyStopping(patience=config['patience'], path=config['save_path'])

    LAMBDA_BBOX = config.get('lambda_bbox', 2.0)

    print(f"Type loss weights: {config['type_loss_weights']}")
    for epoch in range(config['epochs']):
        model.train()
        train_correct, train_total = 0, 0
        train_type_correct = {t: 0 for t in range(num_q_types)}
        train_type_total = {t: 0 for t in range(num_q_types)}

        pbar = tqdm(train_loader, desc=f"Epoch {epoch + 1}")

        for batch in pbar:
            pixel_values = batch['pixel_values'].to(config['device'], dtype=torch.float16)
            input_ids = batch['input_ids'].to(config['device'])
            attention_mask = batch['attention_mask'].to(config['device'])
            q_type = batch['q_type'].to(config['device'])
            labels = batch['labels'].to(config['device'])
            target_box = batch['target_box'].to(config['device'])
            has_box = batch['has_box'].to(config['device'])

            optimizer.zero_grad()
            with autocast():
                logits, pred_boxes = model(pixel_values, input_ids, attention_mask, q_type)

                # Weighted per-sample VQA loss theo q_type
                per_sample_loss = criterion_vqa(logits, labels)
                sample_weights = type_weight_tensor[q_type]
                loss_vqa = (per_sample_loss * sample_weights).mean()

                loss_bbox_raw = criterion_bbox(pred_boxes, target_box).mean(dim=1)
                # has_box = 0 với mẫu không có nhãn box, giúp loại bỏ nhiễu
                loss_bbox = (loss_bbox_raw * has_box).sum() / (has_box.sum() + 1e-6)

                loss = loss_vqa + (LAMBDA_BBOX * loss_bbox)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

            preds = logits.argmax(1)
            train_correct += (preds == labels).sum().item()
            train_total += labels.size(0)

            for t in range(num_q_types):
                mask = (q_type == t)
                if mask.sum() > 0:
                    train_type_correct[t] += (preds[mask] == labels[mask]).sum().item()
                    train_type_total[t] += mask.sum().item()

            pbar.set_postfix({
                'acc': f'{train_correct / train_total:.4f}',
                'L_vqa': f'{loss_vqa.item():.3f}',
                'L_box': f'{loss_bbox.item():.3f}',
            })

        # --- VALIDATION ---
        model.eval()
        val_correct, val_total = 0, 0
        val_type_correct = {t: 0 for t in range(num_q_types)}
        val_type_total = {t: 0 for t in range(num_q_types)}

        with torch.no_grad(), autocast():
            for batch in val_loader:
                pv = batch['pixel_values'].to(config['device'], dtype=torch.float16)
                ii = batch['input_ids'].to(config['device'])
                am = batch['attention_mask'].to(config['device'])
                qt = batch['q_type'].to(config['device'])
                lb = batch['labels'].to(config['device'])

                logits, _ = model(pv, ii, am, qt)
                preds = logits.argmax(1)
                val_correct += (preds == lb).sum().item()
                val_total += lb.size(0)

                for t in range(num_q_types):
                    mask = (qt == t)
                    if mask.sum() > 0:
                        val_type_correct[t] += (preds[mask] == lb[mask]).sum().item()
                        val_type_total[t] += mask.sum().item()

        val_acc = val_correct / val_total
        print(f"Epoch {epoch + 1} | Train Acc: {train_correct / train_total:.4f} | Val Acc: {val_acc:.4f}")

        print("    --- Per Q-Type Answer Accuracy ---")
        for t in range(num_q_types):
            tr_acc = train_type_correct[t] / train_type_total[t] if train_type_total[t] > 0 else 0
            vl_acc = val_type_correct[t] / val_type_total[t] if val_type_total[t] > 0 else 0
            w = config['type_loss_weights'].get(t, 1.0)
            marker = " <<<" if w > 1.0 else ""
            print(f"    Type {t} (w={w}): Train={tr_acc:.4f} | Val={vl_acc:.4f}{marker}")

        early_stopping(val_acc, model)
        if early_stopping.early_stop:
            print("Early stopping triggered")
            break
