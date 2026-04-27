"""
"""
import torch
import os

def save_checkpoint(state, filename="checkpoint.pth"):
    """
    state: dict, usually contains:
        {
            'epoch': epoch,
            'model_state': model.state_dict(),
            'optimizer_state': optimizer.state_dict(),
            'loss': loss
        }
    """
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    torch.save(state, filename)
    print(f"=> Checkpoint saved to {filename}")

def load_checkpoint(checkpoint_path, model, optimizer=None):
    """
    Load model (and optimizer) from checkpoint
    """
    if not os.path.isfile(checkpoint_path):
        raise FileNotFoundError(f"No checkpoint found at '{checkpoint_path}'")

    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    model.load_state_dict(checkpoint['model_state'])

    if optimizer is not None and 'optimizer_state' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state'])

    epoch = checkpoint.get('epoch', 0)
    loss = checkpoint.get('loss', None)

    print(f"=> Checkpoint loaded from {checkpoint_path} (epoch {epoch})")
    return epoch, loss
