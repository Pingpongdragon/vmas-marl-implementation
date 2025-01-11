from Config import PPOConfig
from ActorCriticNetwork import ActorCriticNetwork
from Scheduler import Scheduler
from VMASTruncWrapper import VMASTruncWrapper
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
import vmas
import time
import os

class IPPO:
    def __init__(self, config):
        # 创建一个 VMASTruncWrapper 包装的环境
        self.env = VMASTruncWrapper(
            vmas.make_env(config.env_name, num_envs=config.num_envs, device=config.device, continuous_actions=False, dict_spaces=True),
            max_steps=config.max_env_steps
        )
        # 设置设备（例如 'cuda' 或 'cpu'）
        self.device = config.device
        # 保存配置
        self.config = config
        # 计算每次 rollout 的步数增量
        self.increment = config.num_envs * config.rollout_steps
        # 计算总的学习步数
        self.total_learn_steps = config.total_steps // self.increment
        # 初始化智能体
        self.initialize_agents()
        
    def initialize_agents(self):
        "初始化环境中的所有智能体。"
        self.agents = {}  # 初始化一个空字典，用于存储所有智能体的信息
        print(f"Number of agents: {len(self.env.agents)}")  # 输出智能体数量
        for agent in self.env.agents:  # 遍历环境中的每个智能体
            state_dim = np.prod(self.env.observation_space[agent.name].shape)  # 计算状态空间的维度

            action_dim = self.env.action_space[agent.name].n  # 获取动作空间的维度
            network = ActorCriticNetwork(state_dim, action_dim, self.config.hidden_dim).to(self.device)  # 创建 Actor-Critic 网络并移动到指定设备
            optimizer = torch.optim.Adam(network.parameters(), lr=self.config.lr_start)  # 创建 Adam 优化器
            lr_scheduler = Scheduler(start=self.config.lr_start, end=self.config.lr_final, duration=self.total_learn_steps, log_decay=True)  # 创建学习率调度器
            buffer = {
                'states'      : torch.zeros((self.config.rollout_steps, self.config.num_envs, state_dim), dtype=torch.float32, device=self.device),  # 初始化状态缓冲区
                'actions'     : torch.zeros((self.config.rollout_steps, self.config.num_envs),            dtype=torch.int64,   device=self.device),  # 初始化动作缓冲区
                'rewards'     : torch.zeros((self.config.rollout_steps, self.config.num_envs),            dtype=torch.float32, device=self.device),  # 初始化奖励缓冲区
                'next_states' : torch.zeros((self.config.rollout_steps, self.config.num_envs, state_dim), dtype=torch.float32, device=self.device),  # 初始化下一状态缓冲区
                'terminations': torch.zeros((self.config.rollout_steps, self.config.num_envs),            dtype=torch.float32, device=self.device),  # 初始化终止信号缓冲区
                'truncations' : torch.zeros((self.config.rollout_steps, self.config.num_envs),            dtype=torch.float32, device=self.device),  # 初始化截断信号缓冲区
                'log_probs'   : torch.zeros((self.config.rollout_steps, self.config.num_envs),            dtype=torch.float32, device=self.device),  # 初始化动作概率缓冲区
                'buffer_idx'  : 0  # 初始化缓冲区索引
            }
            self.agents[agent.name] = {'network': network, 'optimizer': optimizer, 'buffer': buffer, 'lr_scheduler': lr_scheduler}  # 将智能体的信息存储到字典中
        
    def checkpoint(self):
        "保存模型权重到磁盘。"
        if not os.path.exists('models'):
            os.makedirs('models', exist_ok=True)  # 如果 'models' 目录不存在，则创建该目录
        for name, agent in self.agents.items():
            checkpoint_path = f"models/IPPO_{self.config.env_name}_{name}_{self.logs['steps']}.pth"  # 定义保存模型权重的路径
            torch.save(agent['network'].state_dict(), checkpoint_path)  # 保存智能体的网络权重到指定路径
    
    def print_progress(self):
        "打印训练进度。"
        print(f"\r--- {100 * self.logs['steps'] / self.config.total_steps:.1f}%" 
            f"\t Step: {self.logs['steps']:,}"  # 当前训练步数
            f"\t Mean Reward: {np.mean(self.logs['mean_rewards']):.2f}"  # 平均奖励
            f"\t Episode: {self.logs['episode_count']:,}"  # 当前完成的回合数量
            f"\t Duration: {time.time() - self.logs['start_time']:,.1f}s  ---", end='')  # 训练持续时间
        if (self.logs['steps'] % self.increment) == 0:
            #记录到日志文件夹中
            if not os.path.exists('logs'):
                os.makedirs('logs', exist_ok=True) 
            with open(f'logs/{self.config.env_name}_IPPO.log', 'a') as f:
                f.write(f"{np.mean(self.logs['episode_rewards'])}\n")
        if (self.logs['steps'] // self.increment) % (self.total_learn_steps // self.config.num_checkpoints) == 0:
            print()  # 如果满足条件，则换行
            
    def select_action(self, agent, state, return_log_probs=False):
        "使用 Boltzmann 策略选择动作。"
        network = agent['network']  # 获取智能体的网络
        with torch.no_grad():  # 在不计算梯度的上下文中执行
            logits = network.actor(state)  # 通过网络的 actor 部分计算 logits
            distribution = torch.distributions.Categorical(logits=logits)  # 创建一个类别分布
            actions = distribution.sample()  # 从分布中采样动作
            if return_log_probs:
                log_probs = distribution.log_prob(actions)
                return actions, log_probs  # 返回动作和对数概率
            return actions  # 仅返回动作
        
    def calculate_advantage(self, agent, states, next_states, rewards, terminations, truncations):
        "使用 GAE（广义优势估计）并行计算优势。"
        network = agent['network']  # 获取智能体的网络
        with torch.no_grad():  # 在不计算梯度的上下文中执行
            # 获取当前状态和下一状态的价值
            values = network.critic(states.view(-1, states.shape[-1])).view(self.config.rollout_steps, self.config.num_envs)
            next_values = network.critic(next_states.view(-1, states.shape[-1])).view(self.config.rollout_steps, self.config.num_envs)
            
            advantages = torch.zeros_like(rewards, device=self.device)  # 初始化优势张量
            advantage = torch.zeros(self.config.num_envs, device=self.device)  # 初始化单步优势张量
            
            # 计算每个时间步和每个环境的 GAE
            for t in reversed(range(self.config.rollout_steps)):
                non_terminal, non_truncation = 1. - terminations[t], 1. - truncations[t]  # 计算非终止和非截断信号
                delta = rewards[t] + self.config.gamma * next_values[t] * non_terminal - values[t]  # 计算 TD 残差
                advantages[t] = advantage = delta + \
                    self.config.gamma * self.config.gae_lambda * non_terminal * non_truncation * advantage  # 计算优势
            returns = advantages + values  # 计算回报
            return advantages, returns  # 返回优势和回报
            
    def rollout(self):
        "为每个环境中的智能体执行 rollouts，收集经验并将其存储在缓冲区中。"
        # 获取上次 rollout 的环境状态
        observations = self.observations

        self.logs['episodic_reward'] = np.zeros(self.config.num_envs)  # 初始化当前episode的总奖励
        self.logs['episode_rewards'] = []  # 初始化每个回合的总奖励
        
        # 主循环，执行 rollout_steps 步
        for step in range(self.config.rollout_steps):
            
            # 动作选择
            actions, log_probs = {}, {}
            for name, agent in self.agents.items():
                actions[name], log_probs[name] = self.select_action(agent, observations[name], return_log_probs=True)
                
            # 环境步进
            next_observations, rewards, terminations, truncations, infos = self.env.step(actions)
            
            # 处理并存储每个智能体的经验
            for name, agent in self.agents.items():
                buffer_idx = agent['buffer']['buffer_idx']
                agent['buffer']['states'][buffer_idx] = observations[name]
                agent['buffer']['actions'][buffer_idx] = actions[name]
                agent['buffer']['rewards'][buffer_idx] = rewards[name]
                agent['buffer']['next_states'][buffer_idx] = next_observations[name]
                agent['buffer']['terminations'][buffer_idx] = terminations
                agent['buffer']['truncations'][buffer_idx] = truncations
                agent['buffer']['log_probs'][buffer_idx] = log_probs[name]
                agent['buffer']['buffer_idx'] += 1
            
            # 日志记录
            self.logs['steps'] += self.config.num_envs
            self.logs['episodic_reward'] += sum(rewards.values()).cpu().numpy()
            
            flag = False
            # 处理环境重置
            if True in terminations or True in truncations:
                if self.config.rollout_steps - step <= self.config.break_count:
                    flag = True
                dones = terminations + truncations
                for done_idx in dones.argwhere().squeeze(1):
                    next_observations = self.env.reset_at(done_idx)
        
                    # 日志记录
                    self.logs['episode_count'] += 1
                    self.logs['episode_rewards'].append(self.logs['episodic_reward'][done_idx])
                    self.logs['episodic_reward'][done_idx] = 0.

            # 设置下一步的观测值
            observations = next_observations

            if flag:
                break
            
        # 设置下次 rollout 的观测值
        self.observations = observations

        if self.logs['episode_rewards']:
            self.logs['mean_rewards'] = np.mean(self.logs['episode_rewards'])
        else:
            self.logs['mean_rewards'] = 0.

        
        
    def learn(self):
        "使用缓冲区中存储的经验为每个智能体执行一次学习步骤。"
        for agent in self.agents.values():
            # 衰减学习率
            if self.config.lr_decay:
                agent['optimizer'].param_groups[0]['lr'] = agent['lr_scheduler']()
            
            # 从缓冲区解包经验
            states, actions, rewards, next_states, terminations, truncations, old_log_probs, _ = agent['buffer'].values()
            agent['buffer']['buffer_idx'] = 0
            
            # 裁剪和缩放奖励
            if self.config.reward_clip:
                rewards = torch.clamp(rewards, min=-self.config.reward_clip, max=self.config.reward_clip)
            if self.config.scale_rewards:
                rewards /= (rewards.std() + 1e-6)

            # 计算优势
            advantages, returns = self.calculate_advantage(agent, states, next_states, rewards, terminations, truncations)

            # 标准化优势
            if self.config.advantage_norm:
                advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

            # 扁平化张量的环境维度
            states = states.view(self.config.rollout_steps * self.config.num_envs, -1)
            actions, advantages, returns, old_log_probs = \
                (x.view(-1) for x in (actions, advantages, returns, old_log_probs))

            # 生成批次和小批次索引
            b_indices = np.arange(len(states))
            minibatch_size = len(states) // self.config.num_minibatches
            for epoch in range(self.config.num_epochs):
                np.random.shuffle(b_indices)
                for minibatch in range(self.config.num_minibatches):
                    start = minibatch * minibatch_size
                    end = start + minibatch_size
                    mb_indices = b_indices[start:end]

                    # 使用当前参数进行前向传播
                    new_logits, new_values = agent['network'](states[mb_indices])

                    # PPO 策略损失
                    distribution = torch.distributions.Categorical(logits=new_logits)
                    new_log_probs = distribution.log_prob(actions[mb_indices])
                    ratio = (new_log_probs - old_log_probs[mb_indices]).exp()
                    loss_surrogate_unclipped = -advantages[mb_indices] * ratio
                    loss_surrogate_clipped = -advantages[mb_indices] * \
                        torch.clamp(ratio, 1 - self.config.ppo_clip, 1 + self.config.ppo_clip)
                    loss_policy = torch.max(loss_surrogate_unclipped, loss_surrogate_clipped).mean()

                    # 价值损失
                    loss_value = F.mse_loss(new_values.squeeze(1), returns[mb_indices])

                    # 熵损失
                    entropy = distribution.entropy().mean()

                    # 组合加权损失
                    loss = loss_policy + \
                        self.config.value_loss * loss_value + \
                        -self.config.entropy_beta * entropy

                    # 网络更新和全局梯度裁剪
                    agent['optimizer'].zero_grad()
                    loss.backward()
                    nn.utils.clip_grad_norm_(agent['network'].parameters(), self.config.grad_norm_clip)
                    agent['optimizer'].step()
    
    def train(self):
        "根据配置训练 IPPO 智能体。"
        if self.config.verbose: 
            print("Training agent\n")  # 如果配置中设置了 verbose，则打印训练开始信息
        
        # 日志信息
        self.logs = {
            'episode_count': 0,  # 记录完成的回合数量
            'episodic_reward': np.zeros(self.config.num_envs),  # 初始化每个并行环境的累积奖励
            'episode_rewards': [],  # 记录每个回合的总奖励
            'mean_rewards': 0,  # 记录每次rollout的平均奖励
            'steps': 0,  # 记录总的步数
            'start_time': time.time()  # 记录训练开始的时间
        }
        
        # 设置初始观测值
        self.observations = self.env.reset()
        
        # 主训练循环
        while self.logs['steps'] < self.config.total_steps:

            # 在每个环境中执行 rollouts
            self.rollout()
            
            # 使用收集的经验进行学习步骤
            self.learn()
            
            # 如果最近 20 次回合的平均奖励超过目标奖励，则结束训练
            if len(self.logs['episode_rewards']) > 0:
                if self.config.target_reward and np.mean(self.logs['episode_rewards'][-20:]) >= self.config.target_reward:
                    break
                    
            # 如果配置中设置了 verbose，则打印训练信息
            if self.config.verbose:
                self.print_progress()
            
            # 模型检查点
            if self.config.checkpoint:
                if (self.logs['steps'] // self.increment % (self.total_learn_steps // self.config.num_checkpoints) == 0):
                    self.checkpoint()

        # 训练结束
        if self.config.verbose: 
            print("\n\nTraining done")
        self.logs['end_time'] = time.time()  # 记录训练结束的时间
        self.logs['duration'] = self.logs['end_time'] - self.logs['start_time']  # 计算训练持续时间

        # 保存最终权重
        if self.config.checkpoint:
            self.checkpoint()

        return self.logs  # 返回日志信息


if __name__ == '__main__':
    ippo = IPPO(PPOConfig())
    ippo.train()