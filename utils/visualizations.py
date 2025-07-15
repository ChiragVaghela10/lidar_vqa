import wandb
import matplotlib.pyplot as plt

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
