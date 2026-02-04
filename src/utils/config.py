import yaml
import argparse
import os

class Config(dict):
    """
    A dictionary that allows access via dot notation (config.model.name),
    replacing the old 'dict2namespace'.
    """
    def __getattr__(self, name):
        value = self.get(name)
        if isinstance(value, dict):
            return Config(value)
        return value

    def __setattr__(self, name, value):
        self[name] = value

    @classmethod
    def load_from_yaml(cls, path):
        with open(path, 'r') as f:
            config_dict = yaml.safe_load(f)
        return cls(config_dict)

def parse_args_and_config():
    parser = argparse.ArgumentParser(description="ADBM Training")
    
    # 核心参数
    parser.add_argument('--config', type=str, required=True, help="Path to config file")
    parser.add_argument('--output_dir', type=str, default="./outputs", help="Output directory")
    parser.add_argument('--seed', type=int, default=42, help="Random seed")
    parser.add_argument('--resume', type=str, default=None, help="Path to checkpoint to resume from")
    
    # 允许命令行覆盖 Config (高级用法)
    # 比如: python train.py --opts training.batch_size=64
    parser.add_argument("opts", default=None, nargs=argparse.REMAINDER, 
                        help="Modify config options from command line")

    args = parser.parse_args()
    
    # 加载 YAML
    config = Config.load_from_yaml(args.config)
    
    return args, config
