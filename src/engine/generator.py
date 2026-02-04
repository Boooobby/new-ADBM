import torch
import torch.nn as nn
import torch.nn.functional as F
from src.utils.data import get_data_scaler, get_data_inverse_scaler

class AdversarialGenerator:
    def __init__(self, eps, num_steps, step_size=None):
        # 直接用传进来的参数 eps (通常是8)
        self.eps = eps / 255.0
        self.num_steps = num_steps
        
        # 这里的 step_size 如果没有传，通常设为 eps (或者 eps/4)
        # 简单起见，我们假设 step_size 等于 eps (这也是常见做法)
        if step_size is None:
             self.step_size = self.eps
        else:
             self.step_size = step_size / 255.0
                
        # 获取缩放器
        self.scaler = get_data_scaler(self.eps)          # [0,1] -> [-1,1]
        self.inverse_scaler = get_data_inverse_scaler(self.eps) # [-1,1] -> [0,1]
    
    def generate(self, classifier, unet, loss_module, 
                 clean_images, labels, fixed_t, fixed_noise, current_T_val):
        """
        clean_images: [0, 1] 范围
        """
        classifier.eval()
        unet.eval()
        
        # 1. 初始化扰动 delta (在 [0, 1] 空间)
        delta = torch.empty_like(clean_images).uniform_(-self.eps, self.eps)
        # 确保初始 x_adv 也在 [0, 1]
        delta = torch.clamp(clean_images + delta, 0.0, 1.0) - clean_images
        delta.requires_grad = True

        for _ in range(self.num_steps):
            # 2. 构造攻击样本 (在 [0, 1] 空间)
            x_adv_01 = torch.clamp(clean_images + delta, 0.0, 1.0)
            
            # 3. 准备进入 UNet (转换为 [-1, 1])
            # 注意：ADBM 的 Bridge 公式是在 [-1, 1] 空间计算的
            x_adv_norm = self.scaler(x_adv_01)         # [-1, 1]
            clean_norm = self.scaler(clean_images)     # [-1, 1]
            
            # 计算 adversarial perturbation 在 [-1, 1] 空间的数值
            # 因为 x_norm = 2*x - 1, 所以 delta_norm = 2 * delta
            adv_perturbation_norm = x_adv_norm - clean_norm 

            # --- UNet Forward (Bridge Logic) ---
            # 复用 Loss 模块中的逻辑来预测 x0
            # 这里的输入必须都是 [-1, 1] 范围
            
            # 计算系数 (基于 fixed_t)
            alpha_t = loss_module.get_alpha(fixed_t).view(-1, 1, 1, 1)
            T_tensor = torch.full_like(fixed_t, current_T_val)
            alpha_T = loss_module.get_alpha(T_tensor).view(-1, 1, 1, 1)

            # 计算 Bridge 系数
            coeff_input = (alpha_T * (1 - alpha_t)) / ((1 - alpha_T) * torch.sqrt(alpha_t))
            input_adv_term = adv_perturbation_norm * coeff_input

            # 加噪: x_t = sqrt(alpha)*clean + sqrt(1-alpha)*noise
            # 注意：这里的 clean 是 [-1, 1] 的
            x_t_clean = torch.sqrt(alpha_t) * clean_norm + torch.sqrt(1 - alpha_t) * fixed_noise
            
            # 加上 Bridge 项
            model_input = x_t_clean + input_adv_term
            
            # UNet 预测噪声
            time_cond = fixed_t * 999
            noise_pred = unet(model_input, time_cond)
            
            # 恢复 x_0_hat (仍在 [-1, 1] 空间)
            # x0 = (x_t - sqrt(1-alpha)*eps) / sqrt(alpha)
            x_0_hat_norm = (model_input - torch.sqrt(1 - alpha_t) * noise_pred) / torch.sqrt(alpha_t)
            # -----------------------------------

            # 4. 准备进入分类器 (还原回 [0, 1])
            x_0_hat_01 = self.inverse_scaler(x_0_hat_norm)
            
            # 5. 计算分类 Loss
            logits = classifier(x_0_hat_01)
            loss = F.cross_entropy(logits, labels)
            
            # 6. PGD 更新 delta
            grad = torch.autograd.grad(loss, [delta])[0]
            with torch.no_grad():
                delta = delta + self.step_size * torch.sign(grad)
                delta = torch.clamp(delta, -self.eps, self.eps)
                # 再次截断确保 x_adv 在 [0, 1]
                delta = torch.clamp(clean_images + delta, 0.0, 1.0) - clean_images
                
            delta.requires_grad = True

        # 返回 [-1, 1] 尺度的扰动，因为 Trainer 里的 Loss 计算是在 [-1, 1] 空间进行的
        # 或者是返回 [0, 1] 尺度的 delta，让 Trainer 自己去缩放？
        # 为了逻辑清晰，Generator 最好返回 "Loss 函数需要的格式"
        # ADBM Loss 需要的是 epsilon_a (在 UNet 输入空间的扰动)
        # 所以我们返回 delta * 2
        final_delta_01 = delta.detach()
        return final_delta_01 * 2.0
