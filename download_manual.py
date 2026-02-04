import os
import shutil

# 1. 强制设置国内镜像环境变量 (写在代码里最保险)
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

try:
    from huggingface_hub import hf_hub_download
except ImportError:
    print("错误: 找不到 huggingface_hub 库。请先运行: pip install huggingface_hub")
    exit(1)

print("🚀 开始从国内镜像下载 WRN-70-16 模型...")
print("这可能需要几分钟 (约 500MB)...")

try:
    # 2. 下载文件到缓存区
    # 使用 Python API 下载，这比命令行工具更稳定
    cached_file_path = hf_hub_download(
        repo_id="croce/robustbench-models",
        filename="cifar10/Linf/Gowal2020Uncovering_70_16_extra.pt",
        repo_type="model"
    )
    print(f"✅ 下载成功！文件暂存路径: {cached_file_path}")

    # 3. 移动并重命名文件
    target_dir = "weights"
    target_name = "WideResNet_70_16_dropout_cifar10.pth"
    target_path = os.path.join(target_dir, target_name)

    # 确保 weights 文件夹存在
    os.makedirs(target_dir, exist_ok=True)

    print(f"📂 正在移动文件到: {target_path}")
    shutil.copy(cached_file_path, target_path)
    
    print("-" * 30)
    print("🎉 搞定！所有权重文件已就绪。")
    print("现在你可以运行 train.py 了！")
    print("-" * 30)

except Exception as e:
    print(f"❌ 下载失败: {e}")