import torch
import torch.nn as nn
import logging
from src.models.backbones.score_sde.models import utils as scoreutils
from src.models.backbones.classifiers.diffpure_resnet import WideResNet

logger = logging.getLogger(__name__)

def get_unet(config, ckpt_path=None, device="cuda"):
    """
    加载 Score-SDE UNet 模型
    """
    logger.info(f"Creating Score-SDE Model...")
    model = scoreutils.create_model(config)
    
    if ckpt_path:
        logger.info(f"Loading UNet weights from {ckpt_path}")
        state = torch.load(ckpt_path, map_location='cpu')
        
        # 处理 EMA 权重的特殊逻辑
        if 'ema' in state and 'shadow_params' in state['ema']:
            logger.info("Loading EMA shadow params...")
            state_dict = state['ema']['shadow_params']
            parameters = [p for p in model.parameters() if p.requires_grad]
            for s_param, param in zip(state_dict, parameters):
                param.data.copy_(s_param.data)
        else:
            model.load_state_dict(state)
            
    model = model.to(device)
    return model

def get_classifier(config, device="cuda"):
    """
    直接从 config 读取参数加载 WideResNet 分类器
    """
    # 1. 直接读取 Config (简单粗暴，不搞默认值)
    # 如果 YAML 里没写这些参数，这里直接报错，让你知道去补 YAML
    depth = config.model.depth
    widen_factor = config.model.widen_factor
    dropRate = config.model.dropRate
    dataset = config.dataset  # 比如 "CIFAR10"
    ckpt_path = config.model.classifier_ckpt

    logger.info(f"Creating WideResNet-{depth}-{widen_factor} (Dropout={dropRate}) for {dataset}...")
    
    # 2. 初始化模型
    model = WideResNet(
        depth=depth, 
        widen_factor=widen_factor, 
        dropRate=dropRate, 
        num_classes=10 if dataset.lower() == 'cifar10' else 100, 
        dataset=dataset
    )
    
    # 3. 加载权重
    if ckpt_path:
        logger.info(f"Loading Classifier weights from {ckpt_path}")
        try:
            state = torch.load(ckpt_path, map_location='cpu')
            
            # 兼容性处理：提取 state_dict
            state_dict = state['ema'] if 'ema' in state else state
            if 'state_dict' in state_dict:
                state_dict = state_dict['state_dict']
            
            # 去掉 'module.' 前缀
            new_state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
                
            model.load_state_dict(new_state_dict)
            logger.info("✅ Classifier weights loaded successfully!")
        except Exception as e:
            logger.error(f"❌ Failed to load classifier: {e}")
            raise e
        
    model = model.to(device)
    model.eval()
    return model