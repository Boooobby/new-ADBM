import sys
import os
# 添加项目根目录到 pythonpath
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from accelerate import Accelerator, DistributedDataParallelKwargs
from src.utils.config import parse_args_and_config
from src.utils.misc import setup_logger, set_seed
from src.utils.checkpoint import load_checkpoint
from src.models.wrapper import get_unet, get_classifier
from src.engine.loss import ADBMLoss
from src.engine.generator import AdversarialGenerator
from src.engine.trainer import ADBMTrainer
from torch_ema import ExponentialMovingAverage # 建议使用 pip install torch_ema 替代旧代码
from src.utils.data import get_dataloader # [新增引用]

# 你需要一个简单的 DataLoader (可以使用 torchvision)
import torchvision
import torchvision.transforms as transforms

def main():
    args, config = parse_args_and_config()
    
    # 1. Setup
    ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
    accelerator = Accelerator(kwargs_handlers=[ddp_kwargs])
    
    logger = setup_logger(config.output_dir)
    logger.info(f"Running with config: {config}")
    set_seed(args.seed)
    
    # 2. Data
    # 简单的 CIFAR-10 加载示例 (注意 ADBM 需要 scale 到 [-1, 1] 还是 [0, 1] 取决于 Generator 实现)
    # 我们在 Generator 里假设了 epsilon=8/255，所以这里建议使用 [0, 1]
    transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor() 
    ])
    train_set = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    train_loader = get_dataloader(config, split='train', distributed=accelerator.use_distributed)
    
    # 3. Models
    unet = get_unet(config, ckpt_path=config.model.init_ckpt, device=accelerator.device)
    classifier = get_classifier(dataset=config.dataset, ckpt_path=config.model.classifier_ckpt, device=accelerator.device)
    
    # 4. Components
    optimizer = torch.optim.Adam(unet.parameters(), lr=config.optim.lr, betas=(config.optim.beta1, 0.999))
    ema = ExponentialMovingAverage(unet.parameters(), decay=config.model.ema_rate)
    
    loss_fn = ADBMLoss(config)
    generator = AdversarialGenerator(eps=config.attack.eps, num_steps=config.attack.steps)
    
    # 5. Prepare (Accelerate)
    # 注意：不要 prepare classifier，因为它不需要梯度更新也不需要 DDP 包装
    unet, optimizer, train_loader = accelerator.prepare(unet, optimizer, train_loader)
    
    # 6. Resume
    start_step = 0
    if args.resume:
        start_step = load_checkpoint(unet, optimizer, ema, args.resume)
    
    # 7. Train
    trainer = ADBMTrainer(
        config=config,
        accelerator=accelerator,
        model=unet,
        classifier=classifier,
        optimizer=optimizer,
        ema=ema,
        loss_fn=loss_fn,
        generator=generator,
        train_loader=train_loader,
        val_loader=None
    )
    
    trainer.train(start_step=start_step, max_steps=config.training.total_iterations)

if __name__ == "__main__":
    main()
