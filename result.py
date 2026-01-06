import numpy as np
import torch
import pygame
from myEnv import TrafficIntersectionEnv
from PPO import Model
import random

model = Model()
model.load_state_dict(torch.load("./checkpoints/model2800.pth"))
pygame.init()
screen = pygame.display.set_mode((600, 600))  # 设置窗口大小
clock = pygame.time.Clock()
# 创建环境实例
env = TrafficIntersectionEnv(render_mode='human')

# 测试环境
for ca in range(100):
    obs = env.reset()
    done = False
    dlist = []
    while not done:
        obs = torch.FloatTensor(obs).reshape(1, 16)
        dlist.append(
            [env.AV.s, env.AV.v, env.HVs[0].s, env.HVs[0].v, env.HVs[1].s, env.HVs[1].v, env.HVs[2].s, env.HVs[2].v])
        mu, std = model(obs)
        action = torch.distributions.Normal(mu, std).sample().item()
        if action >= 2:
            action = 2
        if action >= 10 * (20 / 3.6 - obs[0][2].item()):
            action = 10 * (20 / 3.6 - obs[0][2].item())
        if action < -2:
            action = -2
        if action < -10 * obs[0][2].item():
            action = -10 * obs[0][2].item()
        obs, reward, done, _, _ = env.step([action])
        # print(reward)
        # env.render()
        if done:
            print("Episode finished after {} timesteps".format(env.t))
            np.save(f"./toliutan/{ca}.npy", np.array(dlist))
            break
    env.close()
