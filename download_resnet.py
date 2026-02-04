import torch
from robustbench.utils import load_model
import os

# 1. 确保 weights 文件夹存在
os.makedirs('./weights', exist_ok=True)

print("正在通过 RobustBench 下载 WRN-70-16 (Gowal2020Uncovering)...")

# [修正点] 修改 model_name 为 RobustBench 数据库中的正确 ID
# 对应论文: "Uncovering the Limits of Adversarial Training against Norm-Bounded Adversarial Examples"
model = load_model(model_name='Gowal2020Uncovering_70_16_extra', dataset='cifar10', threat_model='Linf')

# 2. 保存为 .pth 文件
save_path = './weights/WideResNet_70_16_dropout_cifar10.pth'
print(f"正在保存到 {save_path} ...")

# 提取 state_dict
state_dict = model.state_dict()

# 保存
torch.save(state_dict, save_path)
print("完成！你现在可以运行 train.py 了。")