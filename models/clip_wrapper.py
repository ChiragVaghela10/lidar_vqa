import torch
import clip
from PIL import Image

class CLIPWrapper:
    def __init__(self, device="cuda" if torch.cuda.is_available() else "cpu"):
        self.device = device
        print(f"Using device: {self.device}")
        self.model, self.preprocess = clip.load("ViT-B/32", device=self.device)

    def encode_image(self, image_path, bbox=None):
        image = Image.open(image_path).convert("RGB")
        if bbox:
            x1, y1, x2, y2 = map(int, bbox)
            image = image.crop((x1, y1, x2, y2))
        image_tensor = self.preprocess(image).unsqueeze(0).to(self.device)
        with torch.no_grad():
            image_feat = self.model.encode_image(image_tensor)
        return image_feat / image_feat.norm(dim=-1, keepdim=True)

    def encode_text(self, text):
        tokens = clip.tokenize([text]).to(self.device)
        with torch.no_grad():
            text_feat = self.model.encode_text(tokens)
        return text_feat / text_feat.norm(dim=-1, keepdim=True)
