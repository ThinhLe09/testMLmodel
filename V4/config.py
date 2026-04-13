import torch

CONFIG = {
    # File dữ liệu huấn luyện gốc (sẽ được tách thành Train + Val nội bộ)
    "train_csv": "/kaggle/input/datasets/hydrogenhydrogen/vivqa-1/ViVQA-main/ViVQA-main/train.csv",

    # File Test (Hold-out), chỉ dùng để kiểm tra sau cùng
    "test_csv": "/kaggle/input/datasets/hydrogenhydrogen/vivqa-1/ViVQA-main/ViVQA-main/test.csv",

    "train_img_dir": "/kaggle/input/datasets/hydrogenhydrogen/vivqa-1/train",
    "test_img_dir": "/kaggle/input/datasets/hydrogenhydrogen/vivqa-1/test",

    # File mapping loại câu hỏi → đáp án (đặc trưng V4 Hierarchical)
    "type_mapping_csv": "/kaggle/input/datasets/hydrogenhydrogen/vivqa-1/answer_type_mapping.csv",

    "save_path": "phobert_blip2_hierarchical_v4_fixed.pth",

    "blip_model": "Salesforce/blip2-opt-2.7b",
    "text_model": "vinai/phobert-base",

    "batch_size": 32,
    "epochs": 25,
    "lr": 5e-5,
    "weight_decay": 0.05,
    "device": "cuda" if torch.cuda.is_available() else "cpu",
    "max_length": 50,
    "patience": 5,
    "lambda_type": 0.5,

    # Weighted per-sample answer loss theo q_type (boost type 1)
    "type_loss_weights": {0: 1.0, 1: 3.0, 2: 1.0, 3: 1.0},
}
