import torch
import logging
from tqdm import tqdm
from accelerate import Accelerator
from src.utils.checkpoint import save_checkpoint
from src.utils.data import get_data_scaler

logger = logging.getLogger(__name__)

class ADBMTrainer:
    def __init__(self, 
                 config, 
                 accelerator: Accelerator,
                 model, 
                 classifier,
                 optimizer, 
                 ema,
                 loss_fn, 
                 generator,
                 train_loader, 
                 val_loader):
        
        self.config = config
        self.accelerator = accelerator
        self.model = model
        self.classifier = classifier
        self.optimizer = optimizer
        self.ema = ema
        self.loss_fn = loss_fn
        self.generator = generator
        self.train_loader = train_loader
        self.val_loader = val_loader
        
        self.tune_T_min = config.training.tune_T_min
        self.tune_T_max = config.training.tune_T_max
        self.epsilon_t = 1e-5
        self.scaler = get_data_scaler(config) # 获取 [0,1]->[-1,1] 转换器

    def train(self, start_step=0, max_steps=100000):
        
        logger.info(f"Starting training from step {start_step}...")
        progress_bar = tqdm(total=max_steps, initial=start_step, disable=not self.accelerator.is_local_main_process)
        
        step = start_step
        train_iter = iter(self.train_loader)
        
        while step < max_steps:
            # [修改点 1] 处理 DataLoader 耗尽的情况 (Infinite Loop)
            try:
                images, labels = next(train_iter)
            except StopIteration:
                train_iter = iter(self.train_loader)
                images, labels = next(train_iter)
            
            # --- 核心逻辑开始 ---
            
            # 1. 显式采样 (Explicit Sampling)
            B = images.shape[0]
            device = images.device

            # 1. [新增] 动态采样当前的 T (Bridge 的起点)
            # 在 [0.1, 0.2] 之间均匀采样一个标量
            current_T_val = (torch.rand(1, device=device).item() * (self.tune_T_max - self.tune_T_min) + 
                            self.tune_T_min)
            
            t = torch.rand(B, device=device) * (current_T_val - self.epsilon_t) + self.epsilon_t
            noise = torch.randn_like(images)
            
            # 2. 生成对抗扰动 (Generator)
            # 这里的 images 是 [0, 1]，Generator 内部会自己处理缩放
            if self.config.training.advtrain:
                adv_perturbation = self.generator.generate(
                    classifier=self.classifier,
                    unet=self.model,
                    loss_module=self.loss_fn,
                    clean_images=images, # 传入 [0, 1]
                    labels=labels,
                    fixed_t=t,
                    fixed_noise=noise, 
                    current_T_val=current_T_val
                )
            else:
                adv_perturbation = None
                
            # 3. ADBM 训练更新
            self.model.train()
            self.optimizer.zero_grad()
            
            # [修改点 2] 关键数据范围转换 [0, 1] -> [-1, 1]
            # 计算 Loss 时，UNet 需要 [-1, 1] 的输入
            images_norm = self.scaler(images) 
            
            loss = self.loss_fn(
                net=self.model,
                clean_images=images_norm, # <--- 必须传入 [-1, 1] 的数据
                t=t,
                noise=noise,
                adv_perturbation=adv_perturbation, # Generator 已经返回 [-1, 1] 尺度的扰动
                tune_T_val=current_T_val # <--- 关键传参：告诉 Loss 这一步的终点是哪里
            )
            
            self.accelerator.backward(loss)
            
            if self.config.optim.grad_clip > 0:
                self.accelerator.clip_grad_norm_(self.model.parameters(), self.config.optim.grad_clip)
                
            self.optimizer.step()
            self.ema.update(self.model.parameters())
            
            # --- 核心逻辑结束 ---
            
            # Logging
            loss_val = loss.item()
            progress_bar.set_postfix({"loss": f"{loss_val:.4f}"})
            progress_bar.update(1)
            
            if self.accelerator.is_main_process:
                # TODO: 日志记录逻辑 ...
                if step % self.config.training.snapshot_freq == 0 and step > 0:
                    save_checkpoint(
                        dir_path=self.config.output_dir,
                        state={
                            'model': self.accelerator.unwrap_model(self.model).state_dict(),
                            'optimizer': self.optimizer.state_dict(),
                            'ema': self.ema.state_dict(),
                            'step': step
                        },
                        step=step
                    )
            
            step += 1
            
        progress_bar.close()
        logger.info("Training finished.")