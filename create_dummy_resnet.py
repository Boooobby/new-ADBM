import torch
import os
# 从 robustbench 导入网络架构定义 (本地库，无需联网)
from robustbench.model_zoo.architectures.wide_resnet import WideResNet

def create_fake_weights():
    print("正在创建本地 WideResNet-70-16 模型结构...")
    
    # 1. 初始化一个随机权重的模型
    # 参数必须与原论文一致，否则加载时会报错 key mismatch
    model = WideResNet(
        depth=70, 
        widen_factor=16, 
        dropRate=0.3, 
        num_classes=10
    )
    
    # 2. 准备保存路径
    save_dir = "weights"
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, "WideResNet_70_16_dropout_cifar10.pth")
    
    # 3. 保存权重
    print(f"正在保存 '替身' 权重到: {save_path}")
    torch.save(model.state_dict(), save_path)
    print("✅ 成功！")
    print("注意：这是一个随机初始化的模型，仅用于跑通代码流程，没有分类能力。")

if __name__ == "__main__":
    create_fake_weights()
