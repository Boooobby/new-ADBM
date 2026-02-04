import torch
import torch.nn as nn
import logging
from src.models.backbones.score_sde.models import utils as scoreutils
# 动态导入分类器，防止路径依赖问题
from src.models.backbones.classifiers.diffpure_resnet import WideResNet

logger = logging.getLogger(__name__)

def get_unet(config, ckpt_path=None, device="cuda"):
    """
    Load the Score-SDE UNet.
    """
    logger.info(f"Creating Score-SDE Model...")
    
    # 这里的 config 需要兼容 score_sde 的读取方式 (attribute access)
    # 我们之前的 Config 类已经支持了 .attribute 访问
    model = scoreutils.create_model(config)
    
    if ckpt_path:
        logger.info(f"Loading UNet weights from {ckpt_path}")
        # 处理 Score-SDE 特有的 EMA shadow_params 加载逻辑
        state = torch.load(ckpt_path, map_location='cpu')
        
        if 'ema' in state and 'shadow_params' in state['ema']:
            # 加载 EMA 权重 (通常用于 Inference/Fine-tuning)
            logger.info("Loading EMA shadow params...")
            state_dict = state['ema']['shadow_params']
            parameters = [p for p in model.parameters() if p.requires_grad]
            # Score-SDE 的存储方式是 list of tensors，不是 dict
            for s_param, param in zip(state_dict, parameters):
                param.data.copy_(s_param.data)
        else:
            # 标准加载
            model.load_state_dict(state)
            
    model = model.to(device)
    return model

def get_classifier(dataset, depth=70, widen=16, ckpt_path=None, device="cuda"):
    """
    Load the WideResNet Classifier.
    """
    logger.info(f"Creating WideResNet-{depth}-{widen} for {dataset}...")
    
    model = WideResNet(depth=depth, widen_factor=widen, dropRate=0.3, dataset=dataset)
    
    if ckpt_path:
        logger.info(f"Loading Classifier weights from {ckpt_path}")
        state = torch.load(ckpt_path, map_location='cpu')
        
        # 处理可能的 key 前缀
        state_dict = state['ema'] if 'ema' in state else state
        # 如果是 state_dict 嵌套
        if 'state_dict' in state_dict:
            state_dict = state_dict['state_dict']
            
        new_state_dict = {}
        for k, v in state_dict.items():
            # 移除 module. 前缀
            k = k.replace('module.', '')
            new_state_dict[k] = v
            
        model.load_state_dict(new_state_dict)
        
    model = model.to(device)
    model.eval() # 分类器通常在 ADBM 训练中是冻结的
    return model
