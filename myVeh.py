import numpy as np
import matplotlib.pyplot as plt

# 定义路径
AV_path = np.load("D:\研究\基于数值仿真的强化学习环境\AV.npy")
HV_path1 = np.load("D:\研究\基于数值仿真的强化学习环境\HV1.npy")
HV_path2 = np.load("D:\研究\基于数值仿真的强化学习环境\HV2.npy")
HV_path3 = np.load("D:\研究\基于数值仿真的强化学习环境\HV3.npy")


# 车辆状态
class Vehicle:
    def __init__(self, id, s, v, path):
        self.id = id
        self.s = s
        self.v = v
        self.path = path[:, 0:2]
        self.dist = path[:, 2]
        self.a = 0  # 加速度
        self.x = None
        self.y = None
        self.is_stop = False
        self._update_xy()
        self._get_next_conflict_point()
        self.conflict_point = []
        self.dis2point = []

    def _update_xy(self, ):
        p = np.where(self.dist >= self.s)[0]
        if len(p) != 0:
            pos = p[0]
            portion = (self.dist[pos] - self.s) / (self.dist[pos] - self.dist[pos-1])
            self.x = self.path[pos-1, 0] * portion + self.path[pos , 0] * (1 - portion)
            self.y = self.path[pos-1, 1] * portion + self.path[pos , 1] * (1 - portion)
        else:
            print(f"{self.id}车的坐标更新存在问题")

    def _get_next_conflict_point(self, ):
        self.conflict_point = []
        self.dis2point = []
        if self.id == 1:
            if self.s < 26.4347:
                self.conflict_point.append([-2, 0])
                self.dis2point.append(26.4347 - self.s)
            elif self.s < 29.2724:
                self.conflict_point.append([0, 2])
                self.dis2point.append(29.2724 - self.s)
        elif self.id == 2:
            if self.s < 28:
                self.conflict_point.append([0, 2])
                self.dis2point.append(28 - self.s)
            elif self.s < 36:
                self.conflict_point.append([-8, 2])
                self.dis2point.append(36 - self.s)
        elif self.id == 3:
            if self.s < 26.4274:
                self.conflict_point.append([0, 2])
                self.dis2point.append(26.4274 - self.s)
        elif self.id == 4:
            if self.s < 29.2651:
                self.conflict_point.append([-2, 0])
                self.dis2point.append(29.2651 - self.s)
            elif self.s < 35.6998:
                self.conflict_point.append([-8, 2])
                self.dis2point.append(35.6998 - self.s)

    def move(self, dt=0.1):
        if not self.is_stop:
            self.v += self.a * dt
            self.s += self.v * dt + 0.5 * self.a * dt ** 2
        if self.s > self.dist[-1]:
            self.is_stop = True
            self.s = self.dist[-1]
            self.x = self.path[-1, 0]
            self.y = self.path[-1, 1]
            # print(f"{self.id}车到达终点")
        else:
            self._update_xy()
            self._get_next_conflict_point()


def init_veh(seed):
    np.random.seed(seed)
    # 初始行驶距离范围(0.1, 8)  初始速度范围(3, 5.5)
    random_s = numbers = np.random.uniform(0.1, 8, 4)
    random_v = numbers = np.random.uniform(3, 5.5, 4)
    av = Vehicle(1, random_s[0], random_v[0], AV_path)
    hv1 = Vehicle(2, random_s[1], random_v[1], HV_path1)
    hv2 = Vehicle(3, random_s[2], random_v[2], HV_path2)
    hv3 = Vehicle(4, random_s[3], random_v[3], HV_path3)
    return av, hv1, hv2, hv3


def IDM_get_a_next(v, v_front, s, is_front_car_exists, a_max=2, v_des=20 / 3.6, s0=2.0, T=1.5, b=2.0):
    """
    使用IDM模型计算当前时刻的加速度值，用于车道投影后来控制HV
    :param v: 当前时刻的速度
    :param v_front: 当前时刻的前车速度（如果有前车）；如果没有前车，设置为0
    :param s: 与前车的距离（如果有前车）；如果没有前车，设置为0
    :param is_front_car_exists: True or False，用来表示有无前车
    :param a_max: 最大加速度，默认设置为2m/s^2
    :param v_des: 期望速度，默认设置为20km/h
    :param s0: 最小安全距离，默认设置为2.0m
    :param T:反应时间，默认设置为1.5s
    :param b:舒适减速度，默认设置为2m/s^2
    :return:返回当前时刻的加速度
    """
    if is_front_car_exists:
        s_des = s0 + v * T + v * (v - v_front) / (2 * np.sqrt(a_max * b))
        a_idm = a_max * (1 - (v / v_des) ** 4 - (s_des / s) ** 2)
    else:
        a_idm = a_max * (1 - (v / v_des) ** 4)
    if a_idm < -a_max:
        a_idm = -a_max
    return a_idm

