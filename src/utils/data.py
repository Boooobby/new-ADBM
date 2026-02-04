import os
import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, DistributedSampler

# 尝试导入 RobustBench，如果没装也不报错
try:
    from robustbench.data import load_cifar10
    ROBUSTBENCH_AVAILABLE = True
except ImportError:
    ROBUSTBENCH_AVAILABLE = False

def get_data_scaler(config):
    """
    返回一个函数，将 [0, 1] 的数据转换到 UNet 需要的范围 (通常是 [-1, 1])
    """
    def scale_fn(x):
        # [0, 1] -> [-1, 1]
        return x * 2.0 - 1.0
    return scale_fn

def get_data_inverse_scaler(config):
    """
    返回一个函数，将 UNet 输出的 [-1, 1] 数据还原回 [0, 1] (用于分类器或可视化)
    """
    def inverse_scale_fn(x):
        # [-1, 1] -> [0, 1]
        return (x + 1.0) / 2.0
    return inverse_scale_fn

def get_dataloader(config, split='train', distributed=False):
    """
    统一的数据加载接口，支持自动下载。
    """
    dataset_name = config.dataset.lower()
    data_dir = os.path.join('data', dataset_name) # 数据保存在 ./data/cifar10
    
    # 1. 预处理逻辑 (ADBM 需要 [-1, 1] 还是 [0, 1]?)
    # 根据之前的 generator.py 逻辑 (eps=8/255)，我们暂时保持 [0, 1]
    # 如果旧代码依赖 [-1, 1]，可以在这里加 Normalize((0.5,), (0.5,))
    # 关键修改：transform 只负责 ToTensor，输出 [0, 1]
    if split == 'train':
        transform = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            # 删除了 Normalize，确保 DataLoader 输出纯净的 [0, 1]
        ])
    else:
        transform = transforms.Compose([
            transforms.ToTensor(),
        ])

    # 2. 数据集加载
    if dataset_name == 'cifar10':
        if split == 'train':
            # torchvision 会自动检查并下载
            dataset = torchvision.datasets.CIFAR10(
                root='./data', 
                train=True, 
                download=True,  # <--- 自动下载开关
                transform=transform
            )
        else:
            dataset = torchvision.datasets.CIFAR10(
                root='./data', 
                train=False, 
                download=True, 
                transform=transform
            )
            
    elif dataset_name == 'robustbench' and ROBUSTBENCH_AVAILABLE:
        # RobustBench 通常用于测试集，它会自动下载模型和数据
        # 注意：RobustBench 默认返回的是 Tensor，不需要 transform
        print("Loading data via RobustBench...")
        x_test, y_test = load_cifar10(n_examples=config.eval.n_examples, data_dir='./data')
        dataset = torch.utils.data.TensorDataset(x_test, y_test)
        
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")

    # 3. DDP 采样器 (多卡训练必备)
    sampler = None
    if distributed and split == 'train':
        sampler = DistributedSampler(dataset)

    # 4. 创建 DataLoader
    dataloader = DataLoader(
        dataset, 
        batch_size=config.training.batch_size if split=='train' else config.eval.batch_size,
        shuffle=(split == 'train' and sampler is None),
        sampler=sampler,
        num_workers=4,
        pin_memory=True,
        drop_last=(split == 'train')
    )
    
    return dataloader
