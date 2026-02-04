import logging
import os
import random
import numpy as np
import torch
import sys

def setup_logger(output_dir, name="ADBM"):
    """
    Sets up a logger that outputs to console and a file.
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    logger.propagate = False

    # Console Handler
    ch = logging.StreamHandler(stream=sys.stdout)
    ch.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s %(name)s: %(message)s", datefmt="%H:%M:%S")
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    # File Handler
    fh = logging.FileHandler(os.path.join(output_dir, "log.txt"), mode='a')
    fh.setLevel(logging.INFO)
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    return logger

def set_seed(seed):
    """
    Set seed for reproducibility.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    
    # Deterministic behavior for 4090/cuDNN
    torch.backends.cudnn.benchmark = True # 4090 上开启 benchmark 通常更快
    # torch.backends.cudnn.deterministic = True # 只有在极度需要完全一致时才开启，会牺牲性能
