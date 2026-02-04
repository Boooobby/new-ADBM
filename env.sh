pip install torch==2.1.2 torchvision==0.16.2 torchaudio==2.1.2 --index-url https://download.pytorch.org/whl/cu118

pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple

python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}'); print(f'Version: {torch.version.cuda}')"

# 预期输出：True 和 11.8
