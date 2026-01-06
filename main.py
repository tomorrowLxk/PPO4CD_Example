import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from myEnv import TrafficIntersectionEnv
from PPO import Model
import torch.nn.init as init
import random

def get_advantages(deltas):
    advantages = []
    # 反向遍历deltas
    s = 0.0
    for delta in deltas[::-1]:
        s = 0.9 * 0.9 * s + delta
        advantages.append(s)
    # 逆序
    advantages.reverse()
    return advantages


def valuate():
    state = env.reset(seed=random.randint(0, 60))
    reward_sum = 0
    over = False
    while not over:
        action = get_action(state)
        state, reward, over, _, _ = env.step([action])
        reward_sum += reward
        # env.render()
    return reward_sum


def get_action(state):
    state = torch.FloatTensor(state).reshape(1, 16)
    mu, std = model(state)
    # 根据概率选择一个动作
    # action = random.normalvariate(mu=mu.item(), sigma=std.item())
    action = torch.distributions.Normal(mu, std).sample().item()
    if action >= 2:
        action = 2
    if action >= 10*(20/3.6 - state[0][2].item()):
        action = 10*(20/3.6 - state[0][2].item())
    if action < -2:
        action = -2
    if action < -10*state[0][2].item():
        action = -10*state[0][2].item()
    return action


def get_data():
    states = []
    rewards = []
    actions = []
    next_states = []
    overs = []
    state = env.reset(seed=random.randint(0, 60))
    over = False
    while not over:
        action = get_action(state)
        next_state, reward, over, _, _ = env.step([action])
        states.append(state)
        rewards.append(reward)
        actions.append(action)
        next_states.append(next_state)
        overs.append(over)
        state = next_state
    states = torch.FloatTensor(states).reshape(-1, 16)
    rewards = torch.FloatTensor(rewards).reshape(-1, 1)
    actions = torch.FloatTensor(actions).reshape(-1, 1)
    next_states = torch.FloatTensor(next_states).reshape(-1, 16)
    overs = torch.LongTensor(overs).reshape(-1, 1)
    return states, rewards, actions, next_states, overs


def train():
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-5)
    optimizer_td = torch.optim.Adam(model_td.parameters(), lr=5e-3, weight_decay=1e-5)
    loss_fn = torch.nn.MSELoss()

    for epoch in range(3000):
        states, rewards, actions, next_states, overs = get_data()
        rewards = rewards  # 偏移reward,便于训练
        values = model_td(states)
        targets = model_td(next_states).detach()
        targets = targets * 0.99
        targets *= (1 - overs)
        targets += rewards
        deltas = (targets - values).squeeze(dim=1).tolist()
        advantages = get_advantages(deltas)
        advantages = torch.FloatTensor(advantages).reshape(-1, 1)

        mu, std = model(states)
        old_probs = torch.distributions.Normal(mu, std)
        old_probs = old_probs.log_prob(actions).exp().detach()

        for _ in range(10):
            mu, std = model(states)
            new_probs = torch.distributions.Normal(mu, std)
            new_probs = new_probs.log_prob(actions).exp()

            ratios = new_probs / (old_probs+1e-6)
            # print(ratios)
            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 0.2, 1.2) * advantages
            loss = -torch.min(surr1, surr2)
            loss = loss.mean()
            values = model_td(states)
            loss_td = loss_fn(values, targets)
            # print(loss_td)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            optimizer_td.zero_grad()
            loss_td.backward()
            optimizer_td.step()

        if epoch % 200 == 0:
            torch.save(model.state_dict(), f'./checkpoints/new_model{epoch}.pth')
            test_result = sum([valuate() for _ in range(10)]) / 10
            print(epoch, test_result)


if __name__ == '__main__':
    # 预定义超参数
    n_state = 16  # 状态空间维度
    epsilon = 0.98  # 奖励折扣系数

    epochs = 3000  # 训练多少轮
    train_size = 200  # 一次训练要学习多少次

    evaluate_time = 200  # 每隔多少轮训练评估一次
    test_size = 10  # 一次测试要评估多少次

    # 初始化环境
    env = TrafficIntersectionEnv(render_mode='human')
    env.reset(seed=random.randint(0, 60))
    # 定义神经网络
    model = Model()

    model_td = torch.nn.Sequential(
        torch.nn.Linear(n_state, 128),
        nn.ReLU(),
        torch.nn.Linear(128, 1),
    )

    # 获取数据
    train()
