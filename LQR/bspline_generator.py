import sys
import numpy as np
import math
import matplotlib.pyplot as plt

from scipy.interpolate import make_interp_spline
from typing import List

N_IND_SEARCH = 10

class BSplineGenerator:
    
    def __init__(self, x : List[int], y : List[int]):
        # 定义控制点
        control_points = np.array([[ix, iy] for ix, iy in zip(x, y)])
        # 提取 x 和 y 作为独立和因变量
        x = control_points[:, 0]
        y = control_points[:, 1]
        ds = 0.1
        interp_num = (max(x) - min(x)) / ds

        # 使用 B 样条插值生成通过控制点的曲线
        t_control = np.linspace(0, 1, len(control_points))
        spl_x = make_interp_spline(t_control, x, k=3)  # 三次插值
        spl_y = make_interp_spline(t_control, y, k=3)

        # 生成曲线的 t 参数
        t = np.linspace(0, 1, int(interp_num))

        # 计算样条曲线的坐标
        x_interp = spl_x(t)
        y_interp = spl_y(t)

        # 计算一阶导数（速度向量）
        x_dot = spl_x.derivative(nu=1)(t)
        y_dot = spl_y.derivative(nu=1)(t)

        # 计算二阶导数（用于计算曲率）
        x_ddot = spl_x.derivative(nu=2)(t)
        y_ddot = spl_y.derivative(nu=2)(t)

        # 计算参考速度 v (曲线的一阶导数大小)
        v = np.sqrt(x_dot**2 + y_dot**2)

        # 计算曲率 κ
        numerator = np.abs(x_dot * y_ddot - y_dot * x_ddot)  # 曲率分子
        denominator = (x_dot**2 + y_dot**2)**(3/2)  # 曲率分母
        curvature = numerator / denominator

        # 计算参考角速度 w (根据曲率和参考速度)
        w = v * curvature

        # 计算切线方向（yaw）
        yaw = np.arctan2(y_dot, x_dot)

        # [x, y, yaw, k, v, w], shape: (interp_num, 6)
        self.refer_path = np.column_stack([x_interp, y_interp, yaw, curvature, v, w])

    def normalize_angle(self, angle):
        """
        Normalize an angle to [-pi, pi].

        :param angle: (float)
        :return: (float) Angle in radian in [-pi, pi]
        """
        while angle > np.pi:
            angle -= 2.0 * np.pi

        while angle < -np.pi:
            angle += 2.0 * np.pi

        return angle


    def calc_nearest_index(self, robot_state_xy : List[int], pind : int):

        cx = self.refer_path[:, 0]
        cy = self.refer_path[:, 1]
        cyaw = self.refer_path[:, 2]

        dx = [robot_state_xy[0] - icx for icx in cx[pind:(pind + N_IND_SEARCH)]]
        dy = [robot_state_xy[1] - icy for icy in cy[pind:(pind + N_IND_SEARCH)]]

        d = [idx ** 2 + idy ** 2 for (idx, idy) in zip(dx, dy)]

        min_d = min(d)

        ind = d.index(min_d) + pind

        min_d = math.sqrt(min_d)

        dxl = cx[ind] - robot_state_xy[0]
        dyl = cy[ind] - robot_state_xy[1]

        # angle = self.normalize_angle(cyaw[ind] - math.atan2(dyl, dxl))
        # if angle < 0:       # 如果是航向角<0,则是距离是反向
        #     min_d *= -1

        return ind, min_d





if __name__ == "__main__":

    def get_straight_course1():
        ax = [0.0, -10.0, -20.0, -40.0, -50.0, -60.0, -70.0]
        ay = [0.0, -1.0, 1.0, 0.0, -1.0, 1.0, 0.0]
        return ax, ay
    
    def get_straight_course2():
        ax = [0.0, 5.0, 10.0, 20.0, 30.0, 40.0, 50.0]
        ay = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        return ax, ay
    
    def get_forward_course():
        ax = [0.0, 60.0, 125.0, 50.0, 75.0, 30.0, -10.0]
        ay = [0.0, 0.0, 50.0, 65.0, 30.0, 50.0, -20.0]
        return ax, ay
    
    def get_switch_back_course():
        ax = [0.0, 30.0, 6.0, 20.0, 35.0]
        ay = [0.0, 0.0, 20.0, 35.0, 20.0]
        return ax, ay
    
    def show_course():
        # ax, ay = get_straight_course1()
        # ax, ay = get_straight_course2()
        # ax, ay = get_forward_course()
        ax, ay = get_switch_back_course()

        bspline = BSplineGenerator(ax, ay)
        plt.plot(ax, ay, "^k", label="control point")
        plt.plot(bspline.refer_path[:, 0], bspline.refer_path[:, 1], "-r", label="course")
        plt.axis("equal")
        plt.show()
    
    show_course()