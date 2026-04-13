import os
from underthesea import word_tokenize


def preprocess_vietnamese(text):
    text = str(text).lower().strip()
    tokens = word_tokenize(text, format="text")
    return tokens


def find_image_path(folder, img_id):
    valid_exts = ['.jpg', '.png', '.jpeg']
    img_id_str = str(img_id)
    if img_id_str.lower().endswith(tuple(valid_exts)):
        path = os.path.join(folder, img_id_str)
        if os.path.exists(path):
            return path
    for ext in valid_exts:
        path = os.path.join(folder, f"{img_id_str}{ext}")
        if os.path.exists(path):
            return path
    return None
