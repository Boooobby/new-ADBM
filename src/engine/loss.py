import torch
import torch.nn as nn
import torch.nn.functional as F

class ADBMLoss(nn.Module):
    def __init__(self, config):
        """
        ADBM Loss Module for VP-SDE.
        Args:
            config: The configuration object containing 'training' parameters.
        """
        super().__init__()
        # [修改点] 恰如其分的修改：
        # 直接按照 YAML 配置文件的结构读取参数 (config.training.beta_d)
        # 这样代码逻辑和配置文件是完全对应的。
        self.beta_d = config.training.beta_d
        self.beta_min = config.training.beta_min

        # [新增] 读取 tune_T 参数并存为 tune_T_val
        # Generator 指名道姓要找 'tune_T_val'，所以我们必须存成这个名字
        self.tune_T_val = config.training.tune_T

    def get_log_mean_coeff(self, t):
        return -0.25 * t ** 2 * (self.beta_d) - 0.5 * t * self.beta_min

    def get_alpha(self, t):
        # Calculates \bar{\alpha}_t
        return (2 * self.get_log_mean_coeff(t)).exp()

    def forward(self, 
                net, 
                clean_images: torch.Tensor, 
                t: torch.Tensor, 
                noise: torch.Tensor, 
                adv_perturbation: torch.Tensor = None, 
                tune_T_val: float = 0.1):
        """
        Args:
            net: The diffusion UNet.
            clean_images: x_0 [B, C, H, W]
            t: Continuous time steps [B]
            noise: Sampled Gaussian noise \epsilon [B, C, H, W]
            adv_perturbation: \epsilon_a (The adversarial noise) [B, C, H, W]
            tune_T_val: The max timestep T used for purification (e.g., 0.1 or 0.2)
        """
        
        # 1. Calculate coefficients based on t and T
        alpha_t = self.get_alpha(t).view(-1, 1, 1, 1)  # \bar{\alpha}_t
        
        # T is usually a scalar constant for the batch during training
        # We handle T as a tensor of same shape as t for broadcasting
        T_tensor = torch.full_like(t, tune_T_val)
        alpha_T = self.get_alpha(T_tensor).view(-1, 1, 1, 1) # \bar{\alpha}_T

        # 2. Calculate Bridge Coefficients (Eq. 22 & Eq. 9 derived)
        if adv_perturbation is not None:
            # Term added to input: x_t^d input
            # Coeff: (a_T * (1 - a_t)) / ((1 - a_T) * sqrt(a_t))
            coeff_input = (alpha_T * (1 - alpha_t)) / ((1 - alpha_T) * torch.sqrt(alpha_t))
            input_adv_term = adv_perturbation * coeff_input

            # Term added to target: expected output noise
            # Coeff: (a_T * sqrt(1 - a_t)) / ((1 - a_T) * sqrt(a_t))
            # Note: sqrt(1/a_t - 1) = sqrt(1-a_t)/sqrt(a_t)
            coeff_target = (alpha_T * torch.sqrt(1 - alpha_t)) / ((1 - alpha_T) * torch.sqrt(alpha_t))
            target_adv_term = adv_perturbation * coeff_target
        else:
            input_adv_term = 0.0
            target_adv_term = 0.0

        # 3. Diffuse inputs
        # x_t = sqrt(a_t) * x_0 + sqrt(1-a_t) * \epsilon
        diffused_clean = torch.sqrt(alpha_t) * clean_images + torch.sqrt(1 - alpha_t) * noise
        
        # Add bridge term: x_t^d = x_t + coeff * \epsilon_a
        model_input = diffused_clean + input_adv_term
        
        # 4. Model Prediction
        # VP-SDE usually conditions on t \in [0, 1], scaled to 999 for discrete compatibility if needed
        # The original code did `time_step = (t * 999)`
        time_cond = t * 999
        
        # Predict noise
        noise_pred = net(model_input, time_cond)

        # 5. Calculate Target
        # We want model to predict: \epsilon + coeff * \epsilon_a
        target = noise + target_adv_term

        loss = F.mse_loss(noise_pred, target)
        return loss