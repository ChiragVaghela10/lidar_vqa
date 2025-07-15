num_epochs = 1 # 10
batch_size = 32
learning_rate = 1e-3

wandb_config = {
    "model": "ViT-B/32 + MLP head",
    "fusion": "late",
    "freeze_clip": True,
    "dropout": 0.3,
    "hidden_dim": 256,
    "lr": learning_rate,
    "batch_size": batch_size,
    "epochs": num_epochs
}

