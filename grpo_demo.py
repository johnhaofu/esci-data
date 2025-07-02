"""
GRPO (Group Relative Policy Optimization) 算法演示
简化版本，使用模拟数据展示算法核心思想
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import List, Dict, Tuple
import random


class SimpleGRPODemo:
    """
    简化的GRPO算法演示
    使用模拟数据展示核心概念
    """
    
    def __init__(self, group_size: int = 8, epsilon: float = 0.2):
        self.group_size = group_size
        self.epsilon = epsilon
        self.training_history = []
        
    def simulate_response_generation(self, prompt: str) -> List[str]:
        """
        模拟为prompt生成多个响应
        """
        base_responses = [
            f"解答：{prompt}的答案是5",
            f"计算：{prompt} = 5",
            f"让我算一下：{prompt}等于5",
            f"简单计算：{prompt}的结果是5",
            f"数学计算：{prompt} = 6",  # 错误答案
            f"我觉得{prompt}等于4",    # 错误答案
            f"根据计算：{prompt} = 5",
            f"答案：{prompt}是5"
        ]
        
        # 随机选择group_size个响应
        return random.sample(base_responses, self.group_size)
    
    def compute_rewards(self, prompt: str, responses: List[str]) -> List[float]:
        """
        模拟奖励计算
        正确答案（包含"5"）获得高奖励，错误答案获得低奖励
        """
        rewards = []
        
        for response in responses:
            if "= 5" in response or "是5" in response or "等于5" in response:
                # 正确答案，添加一些随机性
                reward = np.random.normal(1.0, 0.1)
            elif "= 6" in response or "是6" in response or "等于6" in response:
                # 错误答案
                reward = np.random.normal(-0.5, 0.1)
            elif "= 4" in response or "是4" in response or "等于4" in response:
                # 错误答案
                reward = np.random.normal(-0.3, 0.1)
            else:
                # 其他情况
                reward = np.random.normal(0.0, 0.2)
            
            rewards.append(reward)
        
        return rewards
    
    def compute_group_advantages(self, rewards: List[float]) -> Tuple[float, List[float]]:
        """
        计算群组相对优势
        
        GRPO核心：优势 = 奖励 - 群组平均奖励
        
        Returns:
            群组平均奖励, 优势列表
        """
        mean_reward = np.mean(rewards)
        advantages = [reward - mean_reward for reward in rewards]
        return mean_reward, advantages
    
    def compute_importance_ratios(self, advantages: List[float]) -> List[float]:
        """
        模拟重要性比率计算
        在实际实现中，这是新策略概率/旧策略概率
        这里用简化的模拟
        """
        # 模拟重要性比率，基于优势大小
        ratios = []
        for advantage in advantages:
            # 正优势的响应有更高的比率
            if advantage > 0:
                ratio = np.random.uniform(1.0, 1.5)
            else:
                ratio = np.random.uniform(0.7, 1.0)
            ratios.append(ratio)
        
        return ratios
    
    def compute_clipped_objective(
        self, 
        ratios: List[float], 
        advantages: List[float]
    ) -> List[float]:
        """
        计算GRPO的裁剪目标函数
        
        目标 = min(ratio * advantage, clip(ratio, 1-ε, 1+ε) * advantage)
        """
        objectives = []
        
        for ratio, advantage in zip(ratios, advantages):
            # 裁剪重要性比率
            clipped_ratio = np.clip(ratio, 1 - self.epsilon, 1 + self.epsilon)
            
            # 计算两个目标值并取最小值
            obj1 = ratio * advantage
            obj2 = clipped_ratio * advantage
            
            objective = min(obj1, obj2)
            objectives.append(objective)
        
        return objectives
    
    def demonstrate_single_step(self, prompt: str) -> Dict:
        """
        演示单步GRPO训练过程
        """
        print(f"\n{'='*60}")
        print(f"GRPO训练步骤演示 - Prompt: {prompt}")
        print(f"{'='*60}")
        
        # 1. 生成响应群组
        print("\n1. 生成响应群组:")
        responses = self.simulate_response_generation(prompt)
        for i, response in enumerate(responses):
            print(f"   响应{i+1}: {response}")
        
        # 2. 计算奖励
        print("\n2. 计算奖励:")
        rewards = self.compute_rewards(prompt, responses)
        for i, (response, reward) in enumerate(zip(responses, rewards)):
            print(f"   响应{i+1}: {reward:.3f}")
        
        # 3. 计算群组相对优势
        print("\n3. 计算群组相对优势:")
        mean_reward, advantages = self.compute_group_advantages(rewards)
        print(f"   群组平均奖励: {mean_reward:.3f}")
        print("   各响应优势:")
        for i, advantage in enumerate(advantages):
            status = "正优势" if advantage > 0 else "负优势"
            print(f"     响应{i+1}: {advantage:+.3f} ({status})")
        
        # 4. 计算重要性比率
        print("\n4. 计算重要性比率:")
        ratios = self.compute_importance_ratios(advantages)
        for i, ratio in enumerate(ratios):
            print(f"   响应{i+1}: {ratio:.3f}")
        
        # 5. 计算裁剪目标
        print("\n5. 计算GRPO裁剪目标:")
        objectives = self.compute_clipped_objective(ratios, advantages)
        total_objective = np.mean(objectives)
        
        for i, (ratio, advantage, objective) in enumerate(zip(ratios, advantages, objectives)):
            clipped_ratio = np.clip(ratio, 1 - self.epsilon, 1 + self.epsilon)
            print(f"   响应{i+1}: ratio={ratio:.3f}, clipped={clipped_ratio:.3f}, "
                  f"advantage={advantage:+.3f}, objective={objective:+.3f}")
        
        print(f"\n   平均目标值: {total_objective:.3f}")
        
        # 记录训练历史
        step_info = {
            'mean_reward': mean_reward,
            'total_objective': total_objective,
            'positive_advantages': sum(1 for a in advantages if a > 0),
            'negative_advantages': sum(1 for a in advantages if a < 0)
        }
        
        return step_info
    
    def run_training_simulation(self, num_steps: int = 10):
        """
        运行多步训练模拟
        """
        print(f"\n{'='*80}")
        print("GRPO训练模拟 - 多步训练过程")
        print(f"{'='*80}")
        
        prompts = [
            "2 + 3",
            "1 + 4", 
            "3 + 2",
            "4 + 1",
            "0 + 5"
        ]
        
        training_stats = []
        
        for step in range(num_steps):
            # 随机选择一个prompt
            prompt = random.choice(prompts)
            
            print(f"\n第{step+1}步训练:")
            print(f"Prompt: {prompt}")
            
            # 简化的单步演示
            responses = self.simulate_response_generation(prompt)
            rewards = self.compute_rewards(prompt, responses)
            mean_reward, advantages = self.compute_group_advantages(rewards)
            
            # 统计信息
            pos_advantages = sum(1 for a in advantages if a > 0)
            neg_advantages = sum(1 for a in advantages if a < 0)
            
            print(f"  平均奖励: {mean_reward:.3f}")
            print(f"  正优势响应: {pos_advantages}/{self.group_size}")
            print(f"  负优势响应: {neg_advantages}/{self.group_size}")
            
            training_stats.append({
                'step': step + 1,
                'mean_reward': mean_reward,
                'positive_ratio': pos_advantages / self.group_size
            })
        
        self.training_history = training_stats
        return training_stats
    
    def plot_training_progress(self):
        """
        绘制训练进度
        """
        if not self.training_history:
            print("没有训练历史数据可绘制")
            return
        
        steps = [stat['step'] for stat in self.training_history]
        mean_rewards = [stat['mean_reward'] for stat in self.training_history]
        positive_ratios = [stat['positive_ratio'] for stat in self.training_history]
        
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
        
        # 平均奖励趋势
        ax1.plot(steps, mean_rewards, 'b-o', linewidth=2, markersize=6)
        ax1.set_title('GRPO训练过程 - 平均奖励变化', fontsize=14, fontweight='bold')
        ax1.set_xlabel('训练步数')
        ax1.set_ylabel('平均奖励')
        ax1.grid(True, alpha=0.3)
        ax1.axhline(y=0, color='r', linestyle='--', alpha=0.5)
        
        # 正优势比例趋势
        ax2.plot(steps, positive_ratios, 'g-s', linewidth=2, markersize=6)
        ax2.set_title('GRPO训练过程 - 正优势响应比例', fontsize=14, fontweight='bold')
        ax2.set_xlabel('训练步数')
        ax2.set_ylabel('正优势响应比例')
        ax2.set_ylim(0, 1)
        ax2.grid(True, alpha=0.3)
        ax2.axhline(y=0.5, color='r', linestyle='--', alpha=0.5, label='50%基线')
        ax2.legend()
        
        plt.tight_layout()
        plt.show()
    
    def explain_algorithm(self):
        """
        解释GRPO算法原理
        """
        print("\n" + "="*80)
        print("GRPO (Group Relative Policy Optimization) 算法原理详解")
        print("="*80)
        
        print("\n🎯 核心思想:")
        print("   GRPO通过群组内的相对比较来估计优势，避免了训练价值函数的需要")
        
        print("\n📊 算法步骤:")
        steps = [
            "生成群组响应：对每个prompt生成K个不同的响应",
            "计算奖励：使用奖励模型评估每个响应的质量",
            "计算群组基线：R̄ = (1/K) Σ R_i，使用群组平均奖励作为基线",
            "计算相对优势：A_i = R_i - R̄，每个响应相对于群组的优势",
            "策略更新：使用裁剪目标函数更新策略参数"
        ]
        
        for i, step in enumerate(steps, 1):
            print(f"   {i}. {step}")
        
        print("\n🧮 数学公式:")
        print("   目标函数：L = E[min(r(θ)·A, clip(r(θ), 1-ε, 1+ε)·A)]")
        print("   其中：")
        print("     • r(θ) = π_θ(a|s) / π_old(a|s) 是重要性比率")
        print("     • A = R - R̄ 是群组相对优势") 
        print("     • R̄ = (1/K) Σ R_i 是群组平均奖励")
        print("     • ε 是裁剪参数，防止过大的策略更新")
        
        print("\n✅ 算法优势:")
        advantages = [
            "内存效率：不需要训练单独的价值函数网络",
            "计算简单：群组相对优势计算直观且高效", 
            "训练稳定：裁剪机制确保策略更新的稳定性",
            "效果显著：在数学推理等任务上表现优异"
        ]
        
        for advantage in advantages:
            print(f"   • {advantage}")
        
        print("\n🎯 应用场景:")
        applications = [
            "数学问题求解",
            "代码生成",
            "逻辑推理",
            "需要精确答案验证的任务"
        ]
        
        for app in applications:
            print(f"   • {app}")
        
        print("\n💡 与PPO的区别:")
        print("   PPO: 需要价值函数估计优势 → A = Q(s,a) - V(s)")
        print("   GRPO: 使用群组均值估计优势 → A = R - R̄")
        print("   结果: GRPO内存使用更少，计算更简单")


def main():
    """
    主函数：运行GRPO算法演示
    """
    print("🚀 GRPO (Group Relative Policy Optimization) 算法复现")
    print("   基于DeepSeekMath论文的强化学习算法")
    
    # 创建演示实例
    demo = SimpleGRPODemo(group_size=8, epsilon=0.2)
    
    # 1. 解释算法原理
    demo.explain_algorithm()
    
    # 2. 单步详细演示
    demo.demonstrate_single_step("2 + 3")
    
    # 3. 多步训练模拟
    print(f"\n{'='*80}")
    print("开始多步训练模拟...")
    training_stats = demo.run_training_simulation(num_steps=15)
    
    # 4. 显示训练统计
    print(f"\n{'='*60}")
    print("训练统计总结:")
    print(f"{'='*60}")
    
    final_stats = training_stats[-5:]  # 最后5步
    avg_reward = np.mean([s['mean_reward'] for s in final_stats])
    avg_positive_ratio = np.mean([s['positive_ratio'] for s in final_stats])
    
    print(f"最后5步平均奖励: {avg_reward:.3f}")
    print(f"最后5步正优势比例: {avg_positive_ratio:.3f}")
    
    if avg_positive_ratio > 0.6:
        print("✅ 训练效果良好！正优势响应比例较高")
    elif avg_positive_ratio > 0.4:
        print("⚠️  训练效果一般，需要更多迭代")
    else:
        print("❌ 训练效果不佳，可能需要调整参数")
    
    # 5. 绘制训练曲线（如果有matplotlib）
    try:
        demo.plot_training_progress()
    except ImportError:
        print("\n注意：需要安装matplotlib来绘制训练曲线")
        print("运行: pip install matplotlib")
    
    print(f"\n{'='*80}")
    print("🎉 GRPO算法演示完成！")
    print("这个演示展示了GRPO的核心思想：")
    print("1. 群组生成多个响应")
    print("2. 使用群组均值作为基线计算相对优势") 
    print("3. 通过裁剪目标函数稳定训练")
    print("4. 避免训练价值函数，提高内存效率")
    print(f"{'='*80}")


if __name__ == "__main__":
    main()