import torch
import clip
import os
import wandb
from pathlib import Path
from tqdm import tqdm
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import train_test_split

from models.cross_attention_fusion import CrossAttentionFusionHead
from train.dataset_qa import KittiQADataset
from config.constants import num_epochs, learning_rate, wandb_config
from utils.visualizations import log_attention_heatmap, log_confusion_matrix, log_classification_report


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
# clip_head = CLIPFusionHead(num_classes=len(candidates)).to(device, dtype=torch.float32)
attention_head = CrossAttentionFusionHead(num_classes=len(candidates)).to(device, dtype=torch.float32)
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.AdamW(
    #filter(lambda p: p.requires_grad, list(clip_model.parameters()) + list(head.parameters())),
    params=attention_head.parameters(),
    lr=learning_rate
)


def run_batch(batch, clip_model, head, criterion, device):
    images = batch["image"].to(device, dtype=torch.float32)
    questions = clip.tokenize(batch["question"]).to(device, dtype=torch.long)
    labels = batch["label"].to(device, dtype=torch.long)

    with torch.no_grad():
        image_feats = clip_model.encode_image(images).float()
        text_feats = clip_model.encode_text(questions).float()

    logits, txt_weights, img_weights = head(image_feats, text_feats)
    loss = criterion(logits, labels)
    preds = logits.argmax(dim=1)

    return loss, preds, labels, txt_weights, img_weights


def train_clip_vqa(
    model=clip_model,
    head=attention_head,
    train_loader=train_loader,
    val_loader=val_loader,
    optimizer=optimizer,
    criterion=criterion,
    device=device,
    candidates=candidates,
    # attention_weights_path=attention_weights_path,
    # log_attention_heatmap=log_attention_heatmap,
    epochs=num_epochs
):
    # train_step, val_step = 0, 0

    for epoch in range(epochs):
        head.train()
        total_train, correct_train, train_loss = 0, 0, 0
        loop = tqdm(train_loader, desc=f"Epoch {epoch + 1} [Train]")

        for batch in loop:
            loss, preds, labels, txt_weights, img_weights = run_batch(batch, clip_model, head, criterion, device)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            correct_train += (preds == labels).sum().item()
            total_train += labels.size(0)
            train_loss += loss.item()

            acc_train = correct_train / total_train
            avg_train_loss = train_loss / total_train
            loop.set_postfix({"Train Acc": f"{acc_train:.4f}", "Train Loss": f"{avg_train_loss:.4f}"})

            wandb.log({
                "train/loss": loss.item(),
                "train/acc": acc_train
            })

            # if train_step % 300 == 0:
            #     os.makedirs(attention_weights_path, exist_ok=True)
            #     torch.save(txt_weights.cpu(), attention_weights_path / Path(f"t2i_step{train_step}.pt"))
            #     torch.save(img_weights.cpu(), attention_weights_path / Path(f"i2t_step{train_step}.pt"))
            #     log_attention_heatmap(txt_weights, "attn_text_to_image", train_step)
            #     log_attention_heatmap(img_weights, "attn_image_to_text", train_step)

            # train_step += 1

        # --- Validation ---
        head.eval()
        total_val, correct_val, val_loss = 0, 0, 0
        all_preds, all_labels = [], []
        loop = tqdm(val_loader, desc=f"Epoch {epoch + 1} [Val]")

        with torch.no_grad():
            for batch in loop:
                loss, preds, labels, txt_weights, img_weights = run_batch(batch, clip_model, head, criterion, device)

                correct_val += (preds == labels).sum().item()
                total_val += labels.size(0)
                val_loss += loss.item()
                all_preds.extend(preds.cpu().tolist())
                all_labels.extend(labels.cpu().tolist())

                acc_val = correct_val / total_val
                avg_val_loss = val_loss / total_val
                loop.set_postfix({"Val Acc": f"{acc_val:.4f}", "Val Loss": f"{avg_val_loss:.4f}"})

                wandb.log({
                    "val/loss": loss.item(),
                    "val/acc": acc_val
                })

                # val_step += 1

        wandb.log({
            "epoch/val_loss": val_loss / len(val_loader),
            "epoch/val_acc": correct_val / total_val,
        })

        log_confusion_matrix(all_labels, all_preds, candidates)#, val_step)
        log_classification_report(all_labels, all_preds)#, val_step)

        print(f"Epoch {epoch+1} [Train]: Acc={correct_train/total_train:.4f}, Loss={train_loss / len(train_loader):.4f}")
        print(f"Epoch {epoch+1} [Val]: Acc={correct_val / total_val:.4f}, Loss={val_loss / len(val_loader):.4f}")

    # --- Save Model ---
    os.makedirs(checkpoints_path, exist_ok=True)
    torch.save(head.state_dict(), checkpoints_path / Path("clip_head.pth"))

    # --- W&B Stopped ---
    wandb.finish()
