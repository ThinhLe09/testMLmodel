import torch

CONFIG = {
    "train_csv": "/kaggle/input/datasets/hydrogenhydrogen/vivqa-1/ViVQA-main/ViVQA-main/train.csv",
    "val_csv": "/kaggle/input/datasets/hydrogenhydrogen/vivqa-1/ViVQA-main/ViVQA-main/test.csv",
    "train_img_dir": "/kaggle/input/datasets/hydrogenhydrogen/vivqa-1/train",
    "val_img_dir": "/kaggle/input/datasets/hydrogenhydrogen/vivqa-1/test",
    "train_box_csv": "/kaggle/input/datasets/hydrogenhydrogen/vivqa-1/merged_train.csv",

    "save_path": "/kaggle/working/multitask_box_vivqa.pth",
    "label_encoder_path": "label_encoder.pkl",

    "blip_model": "Salesforce/blip2-opt-2.7b",
    "text_model": "vinai/phobert-base",

    "batch_size": 32,
    "epochs": 25,
    "lr": 5e-5,
    "weight_decay": 0.05,
    "device": "cuda" if torch.cuda.is_available() else "cpu",

    "type_embed_dim": 64,
    "max_length": 50,
    "patience": 5,
    "delta": 0.001,

    # Weighted per-sample answer loss theo q_type (boost type 1)
    "type_loss_weights": {0: 4.0, 1: 0.5, 2: 4.0, 3: 4.0},
}
