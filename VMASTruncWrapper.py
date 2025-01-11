import torch
import numpy as np

class VMASTruncWrapper:
    "允许 VMAS 环境返回单独的终止和截断信号的包装器。"
    def __init__(self, env, max_steps=None):
        assert env.max_steps is None, "基础环境的最大步数应设置为 None。"
        assert env.dict_spaces, "包装器只能包装具有字典空间的环境。"
        assert not env.continuous_actions, "包装器设计用于离散动作空间。"
        self.env = env
        self.steps = np.zeros(env.num_envs, dtype=np.int64)  # 初始化步数计数器
        self.max_steps = max_steps  # 设置最大步数

    def reset(self, **kwargs):
        "使用额外的关键字参数重置环境。"
        return self.env.reset(**kwargs)
    
    def reset_at(self, index, **kwargs):
        "使用额外的关键字参数重置指定索引处的环境。"
        return self.env.reset_at(index, **kwargs)
        
    def step(self, actions, **kwargs):
        "使用自定义截断逻辑通过环境执行一步，使用关键字参数。"
        self.steps += 1  # 增加步数计数器
        next_observations, rewards, terminations, infos = self.env.step(actions, **kwargs)  # 执行一步
        truncations = torch.zeros_like(terminations)  # 初始化截断信号
        
        # 检查是否需要截断任何环境
        if self.max_steps is not None:
            for env_idx in range(self.num_envs):
                if self.steps[env_idx] >= self.max_steps:
                    truncations[env_idx] = True  # 设置截断信号
                    self.steps[env_idx] = 0  # 重置步数计数器
                    
        # 扁平化动作张量，因为 VMAS 的 step 方法会进行就地形状修改
        for name in actions.keys():
            actions[name] = actions[name].view(-1)

        return next_observations, rewards, terminations, truncations, infos

    def __getattr__(self, name):
        "委托访问原始环境的属性。"
        return getattr(self.env, name)