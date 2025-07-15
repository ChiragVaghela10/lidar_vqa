import json
import os
import torch
from tqdm import tqdm
from models.clip_wrapper import CLIPWrapper

qa_path = "../data/training/generated_qa.json"
image_root = "../data/training/image_2/"

clip_model = CLIPWrapper()
correct, total = 0, 0

with open(qa_path) as f:
    samples = json.load(f)

# Define possible answer candidates
candidates = ["A car", "A pedestrian", "A cyclist", "No object", "Unknown"]
text_feats = torch.cat([clip_model.encode_text(c) for c in candidates])

for sample in tqdm(samples):
    image_path = os.path.join(image_root, os.path.basename(sample["image_path"]))

    for qa in sample["qas"]:
        question = qa["question"]
        gt_answer = qa["answer"]
        bbox = qa["bbox"]

        # Crop region and encode
        region_feat = clip_model.encode_image(image_path, bbox=bbox)

        # Compute cosine similarity to answer candidates
        sims = (region_feat @ text_feats.T).squeeze()
        best_idx = sims.argmax().item()
        pred_answer = candidates[best_idx]

        # Logging
        #print(f"Q: {question}")
        #print(f"Predicted: {pred_answer}, GT: {gt_answer}")
        #print("---")

        if pred_answer == gt_answer:
            correct += 1
        total += 1

print(f"Accuracy: {correct / total:.2f}")
