import os
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
from ActorCriticNetwork import ActorCriticNetwork
from Scheduler import Scheduler
import torch.nn.functional as F
from VMASTruncWrapper import VMASTruncWrapper
from Config import PPOConfig
import vmas


class CPPO:
    def __init__(self, config):
        self.env = VMASTruncWrapper(
            vmas.make_env(config.env_name, num_envs=config.num_envs, device=config.device, continuous_actions=False,
                          dict_spaces=True),
            max_steps=config.max_env_steps
        )
        self.device = config.device
        self.config = config
        self.increment = config.num_envs * config.rollout_steps
        self.total_learn_steps = config.total_steps // self.increment
        self.initialize_agents()

    def initialize_agents(self):
        self.agents = {}
        state_dim = np.prod(self.env.observation_space[self.env.agents[0].name].shape)
        action_dim = self.env.action_space[self.env.agents[0].name].n
        # 创建共享的网络和优化器
        self.shared_network = ActorCriticNetwork(state_dim, action_dim, self.config.hidden_dim).to(self.device)
        self.shared_optimizer = torch.optim.Adam(self.shared_network.parameters(), lr=self.config.lr_start)
        lr_scheduler = Scheduler(start=self.config.lr_start, end=self.config.lr_final, duration=self.total_learn_steps,
                                 log_decay=True)
        for agent in self.env.agents:
            buffer = {
              'states': torch.zeros((self.config.rollout_steps, self.config.num_envs, state_dim), dtype=torch.float32,
                                      device=self.device),
                'actions': torch.zeros((self.config.rollout_steps, self.config.num_envs), dtype=torch.int64,
                                       device=self.device),
              'rewards': torch.zeros((self.config.rollout_steps, self.config.num_envs), dtype=torch.float32,
                                      device=self.device),
                'next_states': torch.zeros((self.config.rollout_steps, self.config.num_envs, state_dim), dtype=torch.float32,
                                           device=self.device),
                'terminations': torch.zeros((self.config.rollout_steps, self.config.num_envs), dtype=torch.float32,
                                            device=self.device),
                'truncations': torch.zeros((self.config.rollout_steps, self.config.num_envs), dtype=torch.float32,
                                           device=self.device),
                'log_probs': torch.zeros((self.config.rollout_steps, self.config.num_envs), dtype=torch.float32,
                                         device=self.device),
                'buffer_idx': 0
            }
            self.agents[agent.name] = {'network': self.shared_network, 'optimizer': self.shared_optimizer,
                                       'buffer': buffer, 'lr_scheduler': lr_scheduler}

    def checkpoint(self):
        if not os.path.exists('models'):
            os.makedirs('models', exist_ok=True)
        for name, agent in self.agents.items():
            checkpoint_path = f"models/CPPO_{self.config.env_name}_{name}_{self.logs['steps']}.pth"
            torch.save(agent['network'].state_dict(), checkpoint_path)

    def print_progress(self):
        print(f"\r--- {100 * self.logs['steps'] / self.config.total_steps:.1f}%"
              f"\t Step: {self.logs['steps']:,}"
              f"\t Mean Reward: {np.mean(self.logs['episode_rewards']):.2f}"
              f"\t Episode: {self.logs['episode_count']:,}"
              f"\t Duration: {time.time() - self.logs['start_time']:,.1f}s  ---", end='')
        if (self.logs['steps'] % self.increment) == 0:
            #记录到日志文件夹中
            if not os.path.exists('logs'):
                os.makedirs('logs', exist_ok=True) 
            with open(f'logs/{self.config.env_name}_CPPO.log', 'a') as f:
                f.write(f"{np.mean(self.logs['episode_rewards'])}\n")
        if (self.logs['steps'] // self.increment) % (self.total_learn_steps // self.config.num_checkpoints) == 0:
            print()

    def select_action(self, agent, state, return_log_probs=False):
        network = self.shared_network
        with torch.no_grad():
            logits, _ = network(state)
            distribution = Categorical(logits=logits)
            actions = distribution.sample()
            if return_log_probs:
                log_probs = distribution.log_prob(actions)
                return actions, log_probs
            return actions

    def calculate_advantage(self, agent, states, next_states, rewards, terminations, truncations):
        network = self.shared_network
        with torch.no_grad():
            values = network.critic(states.view(-1, states.shape[-1])).view(self.config.rollout_steps, self.config.num_envs)
            next_values = network.critic(next_states.view(-1, states.shape[-1])).view(self.config.rollout_steps,
                                                                                     self.config.num_envs)

            advantages = torch.zeros_like(rewards, device=self.device)
            advantage = torch.zeros(self.config.num_envs, device=self.device)

            for t in reversed(range(self.config.rollout_steps)):
                non_terminal, non_truncation = 1. - terminations[t], 1. - truncations[t]
                delta = rewards[t] + self.config.gamma * next_values[t] * non_terminal - values[t]
                advantages[t] = advantage = delta + \
                    self.config.gamma * self.config.gae_lambda * non_terminal * non_truncation * advantage
            returns = advantages + values
            return advantages, returns

    def rollout(self):
        observations = self.observations

        for step in range(self.config.rollout_steps):
            actions, log_probs = {}, {}
            for name, agent in self.agents.items():
                actions[name], log_probs[name] = self.select_action(agent, observations[name], return_log_probs=True)

            next_observations, rewards, terminations, truncations, infos = self.env.step(actions)

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

            self.logs['steps'] += self.config.num_envs
            self.logs['episodic_reward'] += sum(rewards.values()).cpu().numpy()

            if True in terminations or True in truncations:
                dones = terminations + truncations
                for done_idx in dones.argwhere().squeeze(1):
                    next_observations = self.env.reset_at(done_idx)

                    self.logs['episode_count'] += 1
                    self.logs['episode_rewards'] += [self.logs['episodic_reward'][done_idx]]
                    self.logs['episodic_reward'][done_idx] = 0.

            observations = next_observations

        self.observations = observations
        if self.logs['episode_count'] > 0:
            self.logs['mean_reward'] = sum(self.logs['episode_rewards']) / self.logs['episode_count']
        else:
            self.logs['mean_reward'] = 0

    def learn(self):
        # 使用共享的优化器
        optimizer = self.shared_optimizer
        if self.config.lr_decay:
            optimizer.param_groups[0]['lr'] = self.agents[self.env.agents[0].name]['lr_scheduler']()

        for agent in self.agents.values():
            states, actions, rewards, next_states, terminations, truncations, old_log_probs, _ = agent['buffer'].values()
            agent['buffer']['buffer_idx'] = 0

            if self.config.reward_clip:
                rewards = torch.clamp(rewards, min=-self.config.reward_clip, max=self.config.reward_clip)
            if self.config.scale_rewards:
                rewards /= (rewards.std() + 1e-6)

            advantages, returns = self.calculate_advantage(agent, states, next_states, rewards, terminations, truncations)

            if self.config.advantage_norm:
                advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

            states = states.view(self.config.rollout_steps * self.config.num_envs, -1)
            actions, advantages, returns, old_log_probs = \
                (x.view(-1) for x in (actions, advantages, returns, old_log_probs))

            b_indices = np.arange(len(states))
            minibatch_size = len(states) // self.config.num_minibatches
            for epoch in range(self.config.num_epochs):
                np.random.shuffle(b_indices)
                for minibatch in range(self.config.num_minibatches):
                    start = minibatch * minibatch_size
                    end = start + minibatch_size
                    mb_indices = b_indices[start:end]

                    new_logits, new_values = agent['network'](states[mb_indices])

                    distribution = Categorical(logits=new_logits)
                    new_log_probs = distribution.log_prob(actions[mb_indices])
                    ratio = (new_log_probs - old_log_probs[mb_indices]).exp()
                    loss_surrogate_unclipped = -advantages[mb_indices] * ratio
                    loss_surrogate_clipped = -advantages[mb_indices] * \
                        torch.clamp(ratio, 1 - self.config.ppo_clip, 1 + self.config.ppo_clip)
                    loss_policy = torch.max(loss_surrogate_unclipped, loss_surrogate_clipped).mean()

                    loss_value = F.mse_loss(new_values.squeeze(1), returns[mb_indices])

                    entropy = distribution.entropy().mean()

                    loss = loss_policy + \
                           self.config.value_loss * loss_value + \
                          -self.config.entropy_beta * entropy

                    optimizer.zero_grad()
                    loss.backward()
                    nn.utils.clip_grad_norm_(agent['network'].parameters(), self.config.grad_norm_clip)
                    optimizer.step()

    def train(self):
        if self.config.verbose:
            print("Training agent\n")

        self.logs = {
            'episode_count': 0,
            'episodic_reward': np.zeros(self.config.num_envs),
            'episode_rewards': [],
            'mean_reward': 0,
          'steps': 0,
          'start_time': time.time()
        }

        self.observations = self.env.reset()

        while self.logs['steps'] < self.config.total_steps:
            self.rollout()
            self.learn()

            if len(self.logs['episode_rewards']) > 0:
                if self.config.target_reward and np.mean(self.logs['episode_rewards'][-20:]) >= self.config.target_reward:
                    break

            if self.config.verbose and len(self.logs['episode_rewards']) > 0:
                self.print_progress()

            if self.config.checkpoint:
                if (self.logs['steps'] // self.increment % (self.total_learn_steps // self.config.num_checkpoints) == 0):
                    self.checkpoint()

        if self.config.verbose:
            print("\n\nTraining done")
        self.logs['end_time'] = time.time()
        self.logs['duration'] = self.logs['end_time'] - self.logs['start_time']

        if self.config.checkpoint:
            self.checkpoint()

        return self.logs
    
if __name__ == '__main__':
    cppo = CPPO(PPOConfig())
    cppo.train()