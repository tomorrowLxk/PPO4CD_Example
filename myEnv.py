import numpy as np
import gymnasium as gym  # 注意这里使用gymnasium
from gymnasium import spaces
from myVeh import init_veh, IDM_get_a_next
import pygame

pygame.init()
screen = pygame.display.set_mode((600, 600))  # 设置窗口大小
clock = pygame.time.Clock()

def cal_acc(ego, vehicles):
    ans = 10
    for j in range(len(ego.conflict_point)):
        if j > 0 and ans != 10:
            break
        for veh in vehicles:
            if ego.id == veh.id:
                continue
            for k in range(len(veh.conflict_point)):
                if ego.conflict_point[j] == veh.conflict_point[k]:
                    temp_s = ego.dis2point[j] - veh.dis2point[k]
                    if temp_s > 0:
                        temp_a = IDM_get_a_next(ego.v, veh.v, temp_s, True)
                        if temp_a < ans:
                            ans = temp_a
    if ans == 10:
        ego.a = IDM_get_a_next(ego.v, 0, 0, False)
    else:
        ego.a = ans


def is_collision(ego, vehicles):
    flag = 0
    for veh in vehicles:
        if (ego.x - veh.x) ** 2 + (ego.y - veh.y) ** 2 < 9:
            flag += 1
    return flag


class TrafficIntersectionEnv(gym.Env):
    metadata = {'render_modes': ['human'], 'render_fps': 10}

    def __init__(self, render_mode=None):
        self.t = 0
        super(TrafficIntersectionEnv, self).__init__()
        self.render_mode = render_mode
        self.action_space = spaces.Box(low=-2.0, high=2.0, shape=(1,), dtype=np.float32)
        num_vehicles = 4
        num_features_per_vehicle = 4  # 假设有5个状态特征（如位置x, y, v, a）
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(num_vehicles, num_features_per_vehicle),
                                            dtype=np.float32)
        self.fig, self.ax = None, None

        self.AV = None
        self.HVs = [None, None, None]
        self.VEHs = [None, None, None, None]

    def reset(self, seed=None, options=None):
        self.t = 0
        super().reset(seed=seed)
        av, hv1, hv2, hv3 = init_veh(seed)
        self.AV = av
        self.HVs = [hv1, hv2, hv3]
        self.VEHs = [av, hv1, hv2, hv3]
        observation = self._get_observation()
        return observation  # 注意返回值的变化

    def step(self, action):
        self.t += 1
        don = 0
        rew = 0.0
        self.AV.a = action[0]
        for vehicle in self.HVs:
            cal_acc(vehicle, self.VEHs)

        self.AV.move()
        for vehicle in self.HVs:
            vehicle.move()

        flag = is_collision(self.AV, self.HVs)
        # 计算奖励
        rew = self.AV.v / 20*3.6
        if flag != 0:
            rew -= 100
        if self.AV.is_stop:
            rew += 50

        observation = self._get_observation()

        # if flag != 0:
        #     don = 1
        # if self.AV.is_stop and self.HVs[0].is_stop and self.HVs[1].is_stop and self.HVs[2].is_stop:
        if self.AV.is_stop:
            don = 1
        if self.t >1000:
            don = 1
        return observation, rew, don, None, {}

    def render(self):
        if self.render_mode == 'human':
            screen.fill((255, 255, 255))  # 清屏

            # 绘制交叉路口
            pygame.draw.line(screen, (0, 0, 0), (250, 300), (350, 300), 2)
            pygame.draw.line(screen, (0, 0, 0), (300, 250), (300, 350), 2)

            # 绘制车辆
            pygame.draw.circle(screen, (0, 0, 255), (int(self.AV.x * 10 + 300), int(-self.AV.y * 10 + 300)), 5)  # AV
            for hv in self.HVs:
                pygame.draw.circle(screen, (0, 255, 0), (int(hv.x * 10 + 300), int(-hv.y * 10 + 300)), 5)  # HVs

            pygame.display.flip()  # 更新整个屏幕
            clock.tick(self.metadata['render_fps']) # 控制帧率

    def close(self):
        pygame.quit()

    def _get_observation(self):
        # 获取当前环境的观测值
        obs = np.array(
            [self.AV.x, self.AV.y, self.AV.v, self.AV.a,
             self.HVs[0].x, self.HVs[0].y, self.HVs[0].v, self.HVs[0].a,
             self.HVs[1].x, self.HVs[1].y, self.HVs[1].v, self.HVs[1].a,
             self.HVs[2].x, self.HVs[2].y, self.HVs[2].v, self.HVs[2].a])
        return obs

