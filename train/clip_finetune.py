import torch
import clip
import os
import wandb
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, classification_report
from tqdm import tqdm
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import train_test_split

from models.clip_fusion import CLIPFusionHead
from models.cross_attention_fusion import CrossAttentionFusionHead
from train.dataset_qa import KittiQADataset

#if torch.backends.mps.is_available():
#    device = "mps"
if torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"
print(f"Using device: {device}")

num_epochs = 1 # 10
batch_size = 32
learning_rate = 1e-3

# --- W&B Init ---
wandb.init(
    project="clip-vqa-kitti",
    name="phase-2-late-fusion",
    config={
        "model": "ViT-B/32 + MLP head",
        "fusion": "late",
        "freeze_clip": True,
        "dropout": 0.3,
        "hidden_dim": 256,
        "lr": learning_rate,
        "batch_size": batch_size,
        "epochs": num_epochs
    }
)


def log_attention_heatmap(attn_weights, name, step):
    """
    attn_weights: [B, num_heads, Q, K] or [B, Q, K]
    name: wandb key name
    step: training step
    """
    if attn_weights.dim() == 4:
        attn_weights = attn_weights.mean(1)  # avg over heads

    fig, ax = plt.subplots()
    ax.imshow(attn_weights[0].cpu().detach().numpy(), cmap='viridis')
    ax.set_title(name)
    wandb.log({name: wandb.Image(fig)})
    plt.close(fig)


clip_model, preprocess = clip.load("ViT-B/32", device=device)
# Freeze all parameters
for param in clip_model.parameters():
    param.requires_grad = False

#if device == "mps":
#    print("Warming up MPS backend...")
#    dummy = torch.randn(1, 512).to("mps")  # force early compile
#    _ = dummy @ dummy.T

# Freeze all layers except last transformer block
#for name, param in clip_model.visual.named_parameters():
#    if "transformer.resblocks.11" not in name:
#        param.requires_grad = False
#for name, param in clip_model.transformer.named_parameters():
#    if "resblocks.11" not in name:
#        param.requires_grad = False

# --- Dataset ---
candidates = ["A car", "A pedestrian", "A cyclist", "No object", "Unknown"]
dataset = KittiQADataset(
    qa_path="../data/training/generated_qa.json",
    image_root="../data/training/image_2/",
    candidates=candidates,
    clip_preprocess=preprocess
)

indices = list(range(len(dataset)))
train_idx, val_idx = train_test_split(indices, test_size=0.3, random_state=42)
train_loader = DataLoader(Subset(dataset, train_idx), batch_size=32, shuffle=True, pin_memory=False)
val_loader = DataLoader(Subset(dataset, val_idx), batch_size=32, shuffle=False, pin_memory=False)

# head = CLIPFusionHead(num_classes=len(candidates)).to(device)
head = CrossAttentionFusionHead(num_classes=len(candidates)).to(device)
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.AdamW(
    #filter(lambda p: p.requires_grad, list(clip_model.parameters()) + list(head.parameters())),
    head.parameters(),
    lr=learning_rate
)


# --- Training ---
global_step = 0
all_preds = []
all_labels = []

for epoch in range(num_epochs):
    head.train()
    total, correct, running_loss = 0, 0, 0
    loop = tqdm(train_loader, desc=f"Epoch {epoch + 1} [Train]")

    for batch in loop:
        images = batch["image"].to(device)
        questions = clip.tokenize(batch["question"]).to(device)
        labels = batch["label"].to(device)

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
        running_loss += loss.item()

        acc = correct / total
        avg_loss = running_loss / total

        loop.set_postfix({"Acc": f"{acc:.4f}", "Loss": f"{avg_loss:.4f}"})

        # Log Train Summary
        wandb.log({
            "train/loss": loss.item(),
            "train/acc": acc
        })
        if global_step % 100 == 0:
            os.makedirs("attention_weights", exist_ok=True)
            torch.save(txt_weights.cpu(), f"attention_weights/t2i_step{global_step}.pt")
            torch.save(img_weights.cpu(), f"attention_weights/i2t_step{global_step}.pt")
            log_attention_heatmap(txt_weights, "attn_text_to_image", global_step)
            log_attention_heatmap(img_weights, "attn_image_to_text", global_step)
        global_step += 1

    # --- Validation ---
    head.eval()
    total_val, correct_val, val_loss = 0, 0, 0
    loop = tqdm(val_loader, desc=f"Epoch {epoch + 1} [Val]")

    with torch.no_grad():
        for batch in loop:
            images = batch["image"].to(device)
            questions = clip.tokenize(batch["question"]).to(device)
            labels = batch["label"].to(device)

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

    class_names = candidates  # same as you used in KittiQADataset
    cm = confusion_matrix(all_labels, all_preds, labels=range(len(class_names)))
    # Log Val Summary
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

    print(f"Epoch {epoch+1} [Train]: Acc={correct/total:.4f}, Loss={running_loss/len(train_loader):.4f}")
    print(f"Epoch {epoch+1} [Val]: Acc={correct_val / total_val:.4f}, Loss={val_loss / len(val_loader):.4f}")

# Classification Report
report = classification_report(all_labels, all_preds, output_dict=True, zero_division=0)
df_report = pd.DataFrame(report).transpose()
wandb.log({f"val/classification_report": wandb.Table(dataframe=df_report)})

# --- Save Model ---
os.makedirs("checkpoints", exist_ok=True)
torch.save(head.state_dict(), "checkpoints/clip_head.pth")

# --- Finish W&B Run ---
wandb.finish()
