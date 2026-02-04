import torch
import os
import logging

logger = logging.getLogger(__name__)

def save_checkpoint(dir_path, state, step, keep_latest=5):
    """
    Save checkpoint with step info.
    """
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
    
    filename = os.path.join(dir_path, f"checkpoint_{step}.pth")
    torch.save(state, filename)
    logger.info(f"Saved checkpoint to {filename}")

def load_checkpoint(model, optimizer=None, ema=None, path=None, strict=False):
    """
    Robust checkpoint loading handling 'module.' prefix issues.
    """
    if not path or not os.path.exists(path):
        logger.warning(f"Checkpoint path {path} does not exist. Skipping.")
        return 0 # return start_step

    logger.info(f"Loading checkpoint from {path}...")
    checkpoint = torch.load(path, map_location='cpu')

    # 1. Load Model
    if 'model' in checkpoint:
        state_dict = checkpoint['model']
    else:
        state_dict = checkpoint # Handle raw weight files
        
    # Fix DDP 'module.' prefix mismatch
    new_state_dict = {}
    for k, v in state_dict.items():
        name = k[7:] if k.startswith('module.') else k
        new_state_dict[name] = v
        
    missing, unexpected = model.load_state_dict(new_state_dict, strict=strict)
    if missing:
        logger.warning(f"Missing keys: {missing}")
    if unexpected:
        logger.warning(f"Unexpected keys: {unexpected}")

    # 2. Load Optimizer & EMA (Only for resuming training)
    if optimizer and 'optimizer' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer'])
        
    if ema and 'ema' in checkpoint:
        ema.load_state_dict(checkpoint['ema'])

    # 3. Return step
    return checkpoint.get('step', 0)
