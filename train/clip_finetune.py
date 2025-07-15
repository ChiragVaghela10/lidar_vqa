import torch
import clip
import os
import wandb
from pathlib import Path
import pandas as pd
from sklearn.metrics import confusion_matrix, classification_report
from tqdm import tqdm
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import train_test_split

from models.cross_attention_fusion import CrossAttentionFusionHead
from train.dataset_qa import KittiQADataset
from config.constants import num_epochs, learning_rate, wandb_config
from utils.visualizations import log_attention_heatmap


if torch.backends.mps.is_available():
    device = "mps"
elif torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"
print(f"Using device: {device}")
torch.set_default_dtype(torch.float32)


base_dir = Path(__file__).parent.parent
generated_qa_path = base_dir / Path("data/training/generated_qa.json")
training_images_path = base_dir / Path("data/training/image_2/")
checkpoints_path = base_dir / Path("outputs/checkpoints")
attention_weights_path = base_dir / Path("outputs/attention_weights/")
wandb_dir = base_dir / Path("outputs/logs/")
candidates = ["A car", "A pedestrian", "A cyclist", "No object", "Unknown"]


# --- W&B Init ---
wandb.init(
    project="clip-vqa-kitti",
    name="phase-3-cross-attention",
    dir=wandb_dir,
    config= wandb_config
)


# --- Model ---
clip_model, preprocess = clip.load("ViT-B/32", device=device)
clip_model.float()
for param in clip_model.parameters():
    param.requires_grad = False     # Freeze all parameters


# --- Dataset ---
dataset = KittiQADataset(
    qa_path= generated_qa_path,
    image_root= training_images_path,
    candidates=candidates,
    clip_preprocess=preprocess
)


indices = list(range(len(dataset)))
train_idx, val_idx = train_test_split(indices, test_size=0.3, random_state=42)
train_loader = DataLoader(Subset(dataset, train_idx), batch_size=32, shuffle=True, pin_memory=False)
val_loader = DataLoader(Subset(dataset, val_idx), batch_size=32, shuffle=False, pin_memory=False)


# --- Fusion Head ---
# head = CLIPFusionHead(num_classes=len(candidates)).to(device, dtype=torch.float32)
head = CrossAttentionFusionHead(num_classes=len(candidates)).to(device, dtype=torch.float32)
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.AdamW(
    #filter(lambda p: p.requires_grad, list(clip_model.parameters()) + list(head.parameters())),
    head.parameters(),
    lr=learning_rate
)


global_step = 0
all_preds = []
all_labels = []

for epoch in range(num_epochs):
    # --- Training ---
    head.train()
    total, correct, training_loss = 0, 0, 0
    loop = tqdm(train_loader, desc=f"Epoch {epoch + 1} [Train]")

    for batch in loop:
        images = batch["image"].to(device, dtype=torch.float32)
        questions = clip.tokenize(batch["question"]).to(device, dtype=torch.long)
        labels = batch["label"].to(device, dtype=torch.long)

        with torch.no_grad():
            image_feats = clip_model.encode_image(images)
            text_feats = clip_model.encode_text(questions)

        logits, txt_weights, img_weights = head(image_feats, text_feats)
        loss = criterion(logits, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        preds = logits.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)
        training_loss += loss.item()

        acc = correct / total
        avg_loss = training_loss / total

        loop.set_postfix({"Acc": f"{acc:.4f}", "Loss": f"{avg_loss:.4f}"})

        # --- Log Train Summary ---
        wandb.log({
            "train/loss": loss.item(),
            "train/acc": acc
        })
        if global_step % 100 == 0:
            os.makedirs(attention_weights_path, exist_ok=True)
            torch.save(txt_weights.cpu(), attention_weights_path / Path(f"t2i_step{global_step}.pt"))
            torch.save(img_weights.cpu(), attention_weights_path / Path(f"i2t_step{global_step}.pt"))
            log_attention_heatmap(txt_weights, "attn_text_to_image", global_step)
            log_attention_heatmap(img_weights, "attn_image_to_text", global_step)
        global_step += 1

    # --- Validation ---
    head.eval()
    total_val, correct_val, val_loss = 0, 0, 0
    loop = tqdm(val_loader, desc=f"Epoch {epoch + 1} [Val]")

    with torch.no_grad():
        for batch in loop:
            images = batch["image"].to(device, dtype=torch.float32)
            questions = clip.tokenize(batch["question"]).to(device, dtype=torch.long)
            labels = batch["label"].to(device, dtype=torch.long)

            image_feats = clip_model.encode_image(images).float()
            text_feats = clip_model.encode_text(questions).float()

            logits, txt_weights, img_weights = head(image_feats, text_feats)
            loss = criterion(logits, labels)

            preds = logits.argmax(dim=1)
            correct_val += (preds == labels).sum().item()
            total_val += labels.size(0)
            val_loss += loss.item()
            all_preds.extend(preds.cpu().tolist())
            all_labels.extend(labels.cpu().tolist())

    class_names = candidates  # same as we used in KittiQADataset
    cm = confusion_matrix(all_labels, all_preds, labels=range(len(class_names)))
    # --- Log Val Summary ---
    wandb.log({
        "epoch/val_loss": val_loss / len(val_loader),
        "epoch/val_acc": correct_val / total_val
    })

    wandb.log({
        "val/confusion_matrix": wandb.plot.confusion_matrix(
            probs=None,
            y_true=all_labels,
            preds=all_preds,
            class_names=class_names
        )
    })

    print(f"Epoch {epoch+1} [Train]: Acc={correct/total:.4f}, Loss={training_loss / len(train_loader):.4f}")
    print(f"Epoch {epoch+1} [Val]: Acc={correct_val / total_val:.4f}, Loss={val_loss / len(val_loader):.4f}")


# --- Classification Report ---
report = classification_report(all_labels, all_preds, output_dict=True, zero_division=0)
df_report = pd.DataFrame(report).transpose()
wandb.log({f"val/classification_report": wandb.Table(dataframe=df_report)})


# --- Save Model ---
os.makedirs(checkpoints_path, exist_ok=True)
torch.save(head.state_dict(), checkpoints_path / Path("clip_head.pth"))


# --- W&B Stopped ---
wandb.finish()
