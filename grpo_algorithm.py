"""
Group Relative Policy Optimization (GRPO) Algorithm Implementation

GRPO是一种用于训练大语言模型的强化学习算法，最初在DeepSeekMath论文中提出。
它通过群组相对优势估计来避免使用价值函数，从而减少内存使用并提高训练效率。

核心思想：
1. 对每个prompt生成多个响应
2. 使用奖励模型对响应进行评分
3. 计算群组内的相对优势
4. 使用裁剪目标函数更新策略

论文: DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
import math


@dataclass
class GRPOConfig:
    """GRPO算法配置参数"""
    group_size: int = 16  # 每个prompt的响应数量
    epsilon: float = 0.2  # PPO裁剪参数
    beta_kl: float = 0.04  # KL散度系数
    learning_rate: float = 1e-6  # 学习率
    max_length: int = 1024  # 最大序列长度
    batch_size: int = 32  # 批次大小
    num_epochs: int = 1  # 训练轮数
    gamma: float = 1.0  # 折扣因子（通常为1，因为是文本生成任务）


class GRPOTrainer:
    """
    GRPO训练器实现
    
    GRPO算法的核心步骤：
    1. 生成响应群组
    2. 计算奖励
    3. 计算群组相对优势
    4. 更新策略
    """
    
    def __init__(
        self,
        policy_model: nn.Module,
        reward_model: nn.Module,
        tokenizer,
        config: GRPOConfig
    ):
        self.policy_model = policy_model
        self.reward_model = reward_model
        self.tokenizer = tokenizer
        self.config = config
        
        # 保存旧策略用于重要性采样
        self.old_policy_model = None
        
        # 优化器
        self.optimizer = torch.optim.Adam(
            self.policy_model.parameters(),
            lr=config.learning_rate
        )
        
    def generate_responses(
        self,
        prompts: List[str],
        temperature: float = 0.7,
        top_p: float = 0.9
    ) -> Dict[str, List[str]]:
        """
        为每个prompt生成多个响应
        
        Args:
            prompts: 输入提示列表
            temperature: 采样温度
            top_p: nucleus采样参数
            
        Returns:
            字典，键为prompt，值为响应列表
        """
        responses = {}
        
        for prompt in prompts:
            prompt_responses = []
            
            # 对每个prompt生成group_size个响应
            for _ in range(self.config.group_size):
                # 编码输入
                inputs = self.tokenizer(
                    prompt,
                    return_tensors="pt",
                    truncation=True,
                    max_length=self.config.max_length
                )
                
                # 生成响应
                with torch.no_grad():
                    outputs = self.policy_model.generate(
                        inputs.input_ids,
                        max_length=self.config.max_length,
                        temperature=temperature,
                        top_p=top_p,
                        do_sample=True,
                        pad_token_id=self.tokenizer.pad_token_id
                    )
                
                # 解码响应
                response = self.tokenizer.decode(
                    outputs[0][inputs.input_ids.shape[1]:],
                    skip_special_tokens=True
                )
                prompt_responses.append(response)
            
            responses[prompt] = prompt_responses
        
        return responses
    
    def compute_rewards(
        self,
        prompts: List[str],
        responses: Dict[str, List[str]]
    ) -> Dict[str, List[float]]:
        """
        使用奖励模型计算响应的奖励
        
        Args:
            prompts: 输入提示列表
            responses: 响应字典
            
        Returns:
            奖励字典，键为prompt，值为奖励列表
        """
        rewards = {}
        
        for prompt in prompts:
            prompt_rewards = []
            
            for response in responses[prompt]:
                # 构造奖励模型的输入
                reward_input = f"{prompt}\n{response}"
                
                # 编码输入
                inputs = self.tokenizer(
                    reward_input,
                    return_tensors="pt",
                    truncation=True,
                    max_length=self.config.max_length
                )
                
                # 计算奖励
                with torch.no_grad():
                    reward = self.reward_model(**inputs).logits.squeeze().item()
                
                prompt_rewards.append(reward)
            
            rewards[prompt] = prompt_rewards
        
        return rewards
    
    def compute_advantages(
        self,
        rewards: Dict[str, List[float]]
    ) -> Dict[str, List[float]]:
        """
        计算群组相对优势
        
        GRPO的核心创新：使用群组内奖励的均值作为基线
        
        优势 = 奖励 - 群组均值奖励
        
        Args:
            rewards: 奖励字典
            
        Returns:
            优势字典
        """
        advantages = {}
        
        for prompt, prompt_rewards in rewards.items():
            # 计算群组内奖励均值
            mean_reward = np.mean(prompt_rewards)
            
            # 计算每个响应的优势
            prompt_advantages = [
                reward - mean_reward for reward in prompt_rewards
            ]
            
            advantages[prompt] = prompt_advantages
        
        return advantages
    
    def compute_log_probabilities(
        self,
        prompts: List[str],
        responses: Dict[str, List[str]],
        model: nn.Module
    ) -> Dict[str, List[torch.Tensor]]:
        """
        计算模型对响应的对数概率
        
        Args:
            prompts: 输入提示列表
            responses: 响应字典
            model: 策略模型
            
        Returns:
            对数概率字典
        """
        log_probs = {}
        
        for prompt in prompts:
            prompt_log_probs = []
            
            for response in responses[prompt]:
                # 构造完整序列
                full_text = f"{prompt}{response}"
                
                # 编码
                inputs = self.tokenizer(
                    full_text,
                    return_tensors="pt",
                    truncation=True,
                    max_length=self.config.max_length
                )
                
                # 计算对数概率
                with torch.no_grad():
                    outputs = model(**inputs)
                    logits = outputs.logits
                    
                    # 计算每个token的对数概率
                    log_probs_per_token = F.log_softmax(logits, dim=-1)
                    
                    # 获取实际token的对数概率
                    target_log_probs = log_probs_per_token[0, :-1].gather(
                        1, inputs.input_ids[0, 1:].unsqueeze(1)
                    ).squeeze(1)
                    
                    # 只考虑响应部分的token
                    prompt_length = len(self.tokenizer(prompt).input_ids)
                    response_log_probs = target_log_probs[prompt_length-1:]
                    
                    # 求和得到整个响应的对数概率
                    total_log_prob = response_log_probs.sum()
                
                prompt_log_probs.append(total_log_prob)
            
            log_probs[prompt] = prompt_log_probs
        
        return log_probs
    
    def compute_grpo_loss(
        self,
        prompts: List[str],
        responses: Dict[str, List[str]],
        advantages: Dict[str, List[float]]
    ) -> torch.Tensor:
        """
        计算GRPO损失函数
        
        GRPO目标函数：
        L = E[min(r(θ) * A, clip(r(θ), 1-ε, 1+ε) * A)] - β * KL(π_θ || π_old)
        
        其中：
        - r(θ) = π_θ(a|s) / π_old(a|s) 是重要性比率
        - A 是优势函数
        - ε 是裁剪参数
        - β 是KL散度系数
        
        Args:
            prompts: 输入提示列表
            responses: 响应字典
            advantages: 优势字典
            
        Returns:
            损失张量
        """
        # 计算新策略和旧策略的对数概率
        new_log_probs = self.compute_log_probabilities(
            prompts, responses, self.policy_model
        )
        old_log_probs = self.compute_log_probabilities(
            prompts, responses, self.old_policy_model
        )
        
        total_loss = 0.0
        total_samples = 0
        
        for prompt in prompts:
            for i in range(len(responses[prompt])):
                # 重要性比率
                ratio = torch.exp(
                    new_log_probs[prompt][i] - old_log_probs[prompt][i]
                )
                
                # 优势
                advantage = torch.tensor(
                    advantages[prompt][i], dtype=torch.float32
                )
                
                # 裁剪的重要性比率
                clipped_ratio = torch.clamp(
                    ratio,
                    1.0 - self.config.epsilon,
                    1.0 + self.config.epsilon
                )
                
                # GRPO目标（取最小值以保守更新）
                policy_loss = -torch.min(
                    ratio * advantage,
                    clipped_ratio * advantage
                )
                
                total_loss += policy_loss
                total_samples += 1
        
        # 平均损失
        avg_loss = total_loss / total_samples
        
        return avg_loss
    
    def compute_kl_divergence(
        self,
        prompts: List[str],
        responses: Dict[str, List[str]]
    ) -> torch.Tensor:
        """
        计算新策略和旧策略之间的KL散度
        
        Args:
            prompts: 输入提示列表
            responses: 响应字典
            
        Returns:
            KL散度
        """
        new_log_probs = self.compute_log_probabilities(
            prompts, responses, self.policy_model
        )
        old_log_probs = self.compute_log_probabilities(
            prompts, responses, self.old_policy_model
        )
        
        total_kl = 0.0
        total_samples = 0
        
        for prompt in prompts:
            for i in range(len(responses[prompt])):
                kl = old_log_probs[prompt][i] - new_log_probs[prompt][i]
                total_kl += kl
                total_samples += 1
        
        return total_kl / total_samples
    
    def train_step(self, prompts: List[str]) -> Dict[str, float]:
        """
        执行一步GRPO训练
        
        Args:
            prompts: 训练提示列表
            
        Returns:
            训练统计信息
        """
        # 保存旧策略
        if self.old_policy_model is None:
            self.old_policy_model = type(self.policy_model)(
                self.policy_model.config
            )
        
        self.old_policy_model.load_state_dict(
            self.policy_model.state_dict()
        )
        self.old_policy_model.eval()
        
        # 1. 生成响应群组
        responses = self.generate_responses(prompts)
        
        # 2. 计算奖励
        rewards = self.compute_rewards(prompts, responses)
        
        # 3. 计算群组相对优势
        advantages = self.compute_advantages(rewards)
        
        # 4. 计算损失
        policy_loss = self.compute_grpo_loss(prompts, responses, advantages)
        kl_loss = self.compute_kl_divergence(prompts, responses)
        
        # 总损失
        total_loss = policy_loss + self.config.beta_kl * kl_loss
        
        # 5. 反向传播和优化
        self.optimizer.zero_grad()
        total_loss.backward()
        
        # 梯度裁剪
        torch.nn.utils.clip_grad_norm_(
            self.policy_model.parameters(), max_norm=1.0
        )
        
        self.optimizer.step()
        
        # 返回统计信息
        stats = {
            "policy_loss": policy_loss.item(),
            "kl_loss": kl_loss.item(),
            "total_loss": total_loss.item(),
            "avg_reward": np.mean([
                np.mean(prompt_rewards) 
                for prompt_rewards in rewards.values()
            ]),
            "avg_advantage": np.mean([
                np.mean(prompt_advantages)
                for prompt_advantages in advantages.values()
            ])
        }
        
        return stats
    
    def train(
        self,
        train_prompts: List[str],
        num_iterations: int = 100
    ) -> List[Dict[str, float]]:
        """
        执行完整的GRPO训练
        
        Args:
            train_prompts: 训练提示列表
            num_iterations: 训练迭代次数
            
        Returns:
            训练历史统计
        """
        training_history = []
        
        for iteration in range(num_iterations):
            # 随机采样批次
            batch_prompts = np.random.choice(
                train_prompts,
                size=min(self.config.batch_size, len(train_prompts)),
                replace=False
            ).tolist()
            
            # 执行训练步骤
            stats = self.train_step(batch_prompts)
            stats["iteration"] = iteration
            
            training_history.append(stats)
            
            # 打印进度
            if iteration % 10 == 0:
                print(f"Iteration {iteration}: "
                      f"Loss = {stats['total_loss']:.4f}, "
                      f"Reward = {stats['avg_reward']:.4f}")
        
        return training_history


def demonstrate_grpo():
    """
    演示GRPO算法的使用
    """
    print("GRPO (Group Relative Policy Optimization) 算法演示")
    print("=" * 60)
    
    # 模拟数据
    print("\n1. 算法原理：")
    print("GRPO是一种用于训练大语言模型的强化学习算法")
    print("核心创新：使用群组内奖励均值作为基线，避免训练价值函数")
    print("优势：减少内存使用，提高训练效率")
    
    print("\n2. 算法步骤：")
    steps = [
        "生成响应群组：对每个prompt生成多个响应",
        "计算奖励：使用奖励模型评估每个响应",
        "计算优势：优势 = 奖励 - 群组均值奖励",
        "策略更新：使用裁剪目标函数更新策略参数"
    ]
    
    for i, step in enumerate(steps, 1):
        print(f"   {i}. {step}")
    
    print("\n3. 数学公式：")
    print("目标函数：")
    print("L = E[min(r(θ) * A, clip(r(θ), 1-ε, 1+ε) * A)] - β * KL(π_θ || π_old)")
    print("\n其中：")
    print("- r(θ) = π_θ(a|s) / π_old(a|s) 是重要性比率")
    print("- A = R - R̄ 是群组相对优势")
    print("- R̄ = (1/K) Σ R_i 是群组内奖励均值")
    print("- ε 是裁剪参数")
    print("- β 是KL散度系数")
    
    print("\n4. 模拟训练过程：")
    
    # 模拟参数
    config = GRPOConfig(
        group_size=4,
        epsilon=0.2,
        beta_kl=0.04,
        learning_rate=1e-6
    )
    
    # 模拟prompt和响应
    prompt = "解决数学问题：2 + 3 = ?"
    responses = [
        "让我计算：2 + 3 = 5",
        "2加3等于5",
        "2 + 3 = 6",  # 错误答案
        "首先，2 + 3 = 5"
    ]
    
    # 模拟奖励（正确答案获得高奖励）
    rewards = [1.0, 0.8, -0.5, 0.9]
    
    print(f"\nPrompt: {prompt}")
    print("响应和奖励：")
    for i, (response, reward) in enumerate(zip(responses, rewards)):
        print(f"  响应{i+1}: {response} (奖励: {reward})")
    
    # 计算群组相对优势
    mean_reward = np.mean(rewards)
    advantages = [reward - mean_reward for reward in rewards]
    
    print(f"\n群组平均奖励: {mean_reward:.2f}")
    print("群组相对优势：")
    for i, advantage in enumerate(advantages):
        print(f"  响应{i+1}: {advantage:.2f}")
    
    print("\n5. 算法优势：")
    advantages_list = [
        "内存效率：不需要训练价值函数网络",
        "计算效率：群组相对优势计算简单",
        "训练稳定：裁剪机制防止过大的策略更新",
        "效果显著：在数学推理任务上表现优异"
    ]
    
    for advantage in advantages_list:
        print(f"   • {advantage}")
    
    print("\n6. 应用场景：")
    applications = [
        "数学问题求解",
        "代码生成",
        "逻辑推理",
        "需要精确答案的任务"
    ]
    
    for app in applications:
        print(f"   • {app}")


if __name__ == "__main__":
    demonstrate_grpo()