import os
import json
from PIL import Image
from torchvision import transforms
from torch.utils.data import Dataset

class KittiQADataset(Dataset):
    def __init__(self, qa_path, image_root, candidates, clip_preprocess):
        with open(qa_path) as f:
            self.data = json.load(f)
        self.image_root = image_root
        self.candidates = candidates
        self.preprocess = clip_preprocess
        self.augment = transforms.Compose([
            transforms.RandomResizedCrop(224, scale=(0.9, 1.0)),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
            transforms.RandomHorizontalFlip(),
            self.preprocess  # CLIP’s normalize and resize
        ])

        self.samples = []
        for item in self.data:
            for qa in item["qas"]:
                self.samples.append({
                    "image_path": os.path.join(image_root, os.path.basename(item["image_path"])),
                    "bbox": qa["bbox"],
                    "question": qa["question"],
                    "answer": qa["answer"]
                })

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        item = self.samples[idx]
        image = Image.open(item["image_path"]).convert("RGB")
        x1, y1, x2, y2 = map(int, item["bbox"])
        image = image.crop((x1, y1, x2, y2))
        image = self.augment(image)

        return {
            "image_path": item["image_path"],
            "bbox": item["bbox"],
            "image": image,
            "question": item["question"],
            "label": self.candidates.index(item["answer"])
        }
