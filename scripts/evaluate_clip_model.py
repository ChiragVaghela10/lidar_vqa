import os
import torch
import clip
from PIL import Image, ImageDraw, ImageFont
from models.fusion_head import CLIPFusionHead
from train.dataset_qa import KittiQADataset
from torch.utils.data import Subset
from sklearn.model_selection import train_test_split
import torch.nn.functional as F

# --- Setup ---
device = "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"
clip_model, preprocess = clip.load("ViT-B/32", device=device)

# --- Load model head ---
head = CLIPFusionHead(num_classes=5)
head.load_state_dict(torch.load("../train/checkpoints/clip_head.pth", map_location=device))
head.eval().to(device).float()

# --- Load dataset and split ---
candidates = ["A car", "A pedestrian", "A cyclist", "No object", "Unknown"]
dataset = KittiQADataset("../data/training/generated_qa.json", "../data/training/image_2/", candidates, preprocess)

_, val_idx = train_test_split(list(range(len(dataset))), test_size=0.2, random_state=42)
val_subset = Subset(dataset, val_idx)

# --- Inference + Visualization ---
os.makedirs("outputs/eval", exist_ok=True)

img_list = [6, 13, 15]
for i in img_list:  # First 20 samples
    sample = val_subset[i]
    image = sample["image"].unsqueeze(0).to(device)
    question = sample["question"]
    true_label = candidates[sample["label"]]

    with torch.no_grad():
        image_feat = clip_model.encode_image(image).float()
        text_feat = clip_model.encode_text(clip.tokenize([question]).to(device)).float()
        logits = head(image_feat, text_feat)
        probs = F.softmax(logits, dim=1)
        pred_idx = logits.argmax(dim=1).item()
        pred_label = candidates[pred_idx]
        confidence = probs[0, pred_idx].item()

    # --- Load and annotate original image with bounding box ---
    image_path = sample["image_path"]
    original = Image.open(image_path).convert("RGB")
    draw = ImageDraw.Draw(original)

    x1, y1, x2, y2 = map(int, sample["bbox"])
    draw.rectangle([(x1, y1), (x2, y2)], outline="red", width=3)

    text_lines = [
        f"Q: {question}",
        f"Pred: {pred_label} ({confidence * 100:.1f}%)",
        f"GT: {true_label}"
    ]

    y_text = y1 - 60 if y1 - 60 > 0 else y2 + 10
    for idx, line in enumerate(text_lines):
        draw.text((x1, y_text + idx * 20), line, fill="red")

    # Console output
    print(f"Q: {question}")
    print(f"Predicted: {pred_label} ({confidence*100:.1f}%) | Ground Truth: {true_label}")
    print("-" * 50)
    original.save(f"outputs/eval/sample_{i:02d}.png")
