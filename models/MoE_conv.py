import torch
import torch.nn as nn
import torch.nn.functional as F
# 对每个batch分别选取专家，同时专家网络选取为卷积的神经网络
class ConvFeatureExtractor(nn.Module):
    def __init__(self, input_channels, output_channels):
        super(ConvFeatureExtractor, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(input_channels, output_channels, kernel_size=1, stride=1, padding=0),
            nn.GELU(),
        )

    def forward(self, x):
        return self.conv(x)

class GlobalTopKGating(nn.Module):
    def __init__(self, input_dim, num_experts, top_k=2, initial_temperature=2.0, is_finetune=False): # 载入模型，finetune的时候这里一定要改成0.5
        super().__init__()
        self.num_experts = num_experts
        self.top_k = top_k
        self.temperature = initial_temperature
        self.min_temperature = 0.5
        self.temperature_decay = 0.99
        
        if is_finetune: # 第二阶段不在改变温度
            self.temperature = 0.5
            
        # 全局特征提取
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        
        # 增强的门控网络
        self.gate = nn.Sequential(
            # 特征变换
            nn.Conv2d(input_dim, input_dim*2, 1),
            nn.BatchNorm2d(input_dim*2),   # 理论上不应该选择batchnorm的，因为会使得不同数据之间产生关联
            nn.GELU(),
            
            # 通道注意力
            ChannelAttention(input_dim*2),
            
            # 预测层
            nn.Conv2d(input_dim*2, input_dim, 1),
            nn.BatchNorm2d(input_dim),
            nn.GELU(),
            nn.Conv2d(input_dim, num_experts, 1)
        )
    
    def update_temperature(self):
        self.temperature = max(
            self.min_temperature,
            self.temperature * self.temperature_decay
        )
    
    def forward(self, x):
        # 提取全局特征
        global_feat = self.global_pool(x)  # [B, C, 1, 1]
        gating_scores = self.gate(global_feat).squeeze(-1).squeeze(-1)  # [B, num_experts]
        
        # 选择top-k专家
        top_k_values, top_k_indices = torch.topk(gating_scores, self.top_k, dim=1)  # [B, top_k]
        top_k_values = F.softmax(top_k_values / self.temperature, dim=1)
        
        return top_k_indices, top_k_values

class Expert(nn.Module):
    def __init__(self, input_channels, output_channels):
        super(Expert, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(input_channels, output_channels, kernel_size=1, stride=1, padding=0),
            nn.GELU()
        )

    def forward(self, x):
        return self.conv(x)

class ChannelAttention(nn.Module):
    def __init__(self, channels, reduction=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        
        self.fc = nn.Sequential(
            nn.Conv2d(channels, channels // reduction, 1),
            nn.GELU(),
            nn.Conv2d(channels // reduction, channels, 1)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        out = self.sigmoid(avg_out + max_out)
        return x * out

class MoEImage(nn.Module):
    def __init__(self, input_channels, hidden_channels, output_channels, 
                 num_experts, shared_experts_num=2, top_k=4 ,is_finetune=False):
        super().__init__()
        self.input_channels = input_channels
        self.hidden_channels = hidden_channels
        self.output_channels = output_channels
        self.num_experts = num_experts
        self.shared_experts_num = shared_experts_num
        self.top_k = top_k
        self.is_finetune = is_finetune

        # 基础网络组件
        self.feature_extractor = ConvFeatureExtractor(input_channels, hidden_channels)
        self.gating = GlobalTopKGating(hidden_channels, num_experts, top_k, is_finetune=self.is_finetune)
        
        # 共享专家网络
        self.shared_experts = nn.ModuleList([
            Expert(hidden_channels, output_channels) 
            for _ in range(shared_experts_num)
        ])
        
        # 专家网络
        self.experts = nn.ModuleList([
            Expert(hidden_channels, output_channels) 
            for _ in range(num_experts)
        ])

    def freeze_feature_and_gating(self, freeze=True):
        """
        冻结或解冻特征提取器和门控网络的参数
        Args:
            freeze (bool): True表示冻结参数，False表示解冻参数
        """
        for param in self.feature_extractor.parameters():
            param.requires_grad = not freeze
        for param in self.gating.parameters():
            param.requires_grad = not freeze
            
    def forward(self, x):
        features = self.feature_extractor(x)
        
        # 1. 共享专家的输出
        shared_output = torch.zeros_like(x)
        for expert in self.shared_experts:
            shared_output += expert(features) / self.shared_experts_num
        
        # 2. 专家网络的输出
        output = torch.zeros_like(x)
        
        # 获取专家分配
        top_k_indices, top_k_values = self.gating(features)
        
        # 对每个专家处理数据
        for expert_idx in range(self.num_experts):
            mask = (top_k_indices == expert_idx)
            weights = top_k_values * mask
            expert_output = self.experts[expert_idx](features)
            output += expert_output * weights.sum(dim=1).view(-1, 1, 1, 1)
        
        # 更新专家使用统计，finetune阶段不在计算门控损失
        if self.training and not self.is_finetune:
            # 计算平衡损失
            loss_gate = self.compute_balance_loss(top_k_values, top_k_indices)
            # 更新温度
            self.gating.update_temperature()
        else:
            loss_gate = 0

        # 最终输出是共享专家和专家网络输出的组合
        return shared_output + output, loss_gate

    def compute_balance_loss(self, gates, indices):
        """计算负载均衡损失"""
        importance = torch.zeros(self.num_experts, device=gates.device)
        for i in range(self.num_experts):
            mask = (indices == i)
            importance[i] = (gates * mask).sum()
        
        # 计算理想负载
        ideal_load = gates.sum() / self.num_experts
        # 计算负载均衡损失
        balance_loss = torch.pow(importance - ideal_load, 2).mean() # 理论上的CV还应该除以理想负载
            
        return balance_loss

# Example usage
if __name__ == "__main__":
    # 初始化模型
    model = MoEImage(
        input_channels=3,
        hidden_channels=16,
        output_channels=3,
        num_experts=4,
        top_k=2
    )

    # 模拟训练过程
    model.train()
    x = torch.randn(8, 3, 32, 32)
    
    for epoch in range(10):
        output, loss_gate = model(x)
        print(f"Epoch {epoch}, Gate Loss: {loss_gate.item():.4f}")
        print(f"Current temperature: {model.gating.temperature:.4f}")
