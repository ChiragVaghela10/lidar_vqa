import wandb
import pandas as pd
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt


def log_attention_heatmap(attn_weights, name):#, step):
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


def log_confusion_matrix(all_labels, all_preds, class_names):#, step):
    wandb.log({
        "val/confusion_matrix": wandb.plot.confusion_matrix(
            probs=None,
            y_true=all_labels,
            preds=all_preds,
            class_names=class_names
        )
    })


def log_classification_report(all_labels, all_preds):#, step):
    report = classification_report(all_labels, all_preds, output_dict=True, zero_division=0)
    df_report = pd.DataFrame(report).transpose()
    wandb.log({"val/classification_report": wandb.Table(dataframe=df_report)})