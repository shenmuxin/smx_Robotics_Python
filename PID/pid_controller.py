import numpy as np
import math
import matplotlib.pyplot as plt
from bspline_generator import BSplineGenerator

class KinematicMode:
    """控制量为线速度v,和角速度omega"""

    def __init__(self, x, y, yaw, v, w, dt):
        # state
        self.x = x
        self.y = y
        self.yaw = yaw
        # control
        self.v = v
        self.w = w
        # time tick
        self.dt = dt
    
    def update_state(self, v, w):
        """
        @func:
            update state space according to control input v and w
        @param:
            v, linear velocity
            w, angular velocity
        """
        # update state
        self.x = self.x + self.v * math.cos(self.yaw) * self.dt
        self.y = self.y + self.v * math.sin(self.yaw) * self.dt
        self.yaw = self.yaw + self.w * self.dt
        
        # update input
        self.v = v
        self.w = w
    
    def get_state(self):
        return np.array([self.x, self.y, self.yaw, self.v, self.w])


class PIDControllerPosition:

    def __init__(self, kp, ki, kd, dt, upper_b = 1., lower_b = -1.):
        self.error = 0
        self.pre_error = 0
        self.sum_error = 0

        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.dt = dt            # 采样时间
        self.upper_bound = upper_b
        self.lower_bound = lower_b
        self.u = 0
    
    def set_target(self, target):
        self.target = target
    
    def reset(self):
        self.error = 0
        self.pre_error = 0
        self.sum_error = 0
    
    def set_pid(self, kp, ki, kd):
        self.kp = kp
        self.ki = ki
        self.kd = kd
    
    def cal_to_control(self, e):
        self.error = e
        # 位置式PID
        self.u = self.kp * self.error + self.ki * self.dt * self.sum_error \
            + self.kd / self.dt * (self.error - self.pre_error)

        self.pre_error = self.error
        self.sum_error += self.error
        if self.u > self.upper_bound:
            self.u = self.upper_bound
        elif self.u < self.lower_bound:
            self.u = self.lower_bound

        return self.u



if __name__ == "__main__":
    # test for PID control

    # time tick
    DT = 0.1
    # max time
    MAX_ITER_TIME = 70
    GOAL_DIS = 0.1

    def check_goal(robot_state_xy, goal, target_ind, end_ind):
        dx = robot_state_xy[0] - goal[0]
        dy = robot_state_xy[1] - goal[1]
        d = math.sqrt(dx ** 2 + dy ** 2)

        if (d <= GOAL_DIS):
            isgoal = True
        else:
            isgoal = False

        # 这里有问题
        if abs(target_ind - end_ind) == 0:
            isgoal = True
        
        return isgoal


    def main():
        # initialize pid controller
        # pid_controller_v = PIDControllerPosition(kp=2, ki=0.1, kd=0, dt=DT, upper_b=2, lower_b=-2)
        pid_controller_w = PIDControllerPosition(kp=20, ki=2, kd=1, dt=DT, upper_b=np.pi/3, lower_b=-np.pi/3)
        pid_controller_yaw = PIDControllerPosition(kp=20, ki=3, kd=3, dt=DT, upper_b=np.pi/2, lower_b=-np.pi/2)

        # set reference trajectory
        ax = [0.0, 30.0, 6.0, 20.0, 35.0]
        ay = [0.0, 0.0, 20.0, 35.0, 20.0]
        
        bspline = BSplineGenerator(ax, ay)

        # 初始化x=0,y=0,v=0,w=0,
        uav = KinematicMode(x=-1, y=-2, yaw=0, v=0, w=0, dt=DT)

        fig = plt.figure(1)

        time = 0.0
        x_ = [uav.x]
        y_ = [uav.y]
        yaw_ = [uav.yaw]

        v_ = [uav.v]
        w_ = [uav.w]
        t_ = [0.0]

        target_ind, _ = bspline.calc_nearest_index([uav.x, uav.y], pind=0)
        goal = bspline.refer_path[-1, 0:2]

        while time <= MAX_ITER_TIME:
            target_ind, _ = bspline.calc_nearest_index([uav.x, uav.y], pind=target_ind)

            if check_goal([uav.x, uav.y], goal, target_ind, len(bspline.refer_path)-1):
                break

            # 角度误差
            theta = bspline.refer_path[target_ind, 2]
            error_theta = theta - uav.yaw
            # 位置误差
            error_y = uav.x - bspline.refer_path[target_ind, 1]
            error_x = uav.y - bspline.refer_path[target_ind, 0]

            # l_d = np.linalg.norm(np.array([uav.x, uav.y]) - bspline.refer_path[target_ind, 0:2])
            # error_y = -l_d * math.sin(error_theta)

            # 转换到机器人局部坐标系
            error_local = np.array([
                error_x * math.cos(theta) + error_y * math.sin(theta),
                - error_x * math.sin(theta) + error_y * math.cos(theta)])

            # ov = pid_controller_v.cal_to_control(error_local[0])
            ow = pid_controller_w.cal_to_control(error_local[1])

            ow1 = pid_controller_w.cal_to_control(error_y)
            ow2 = pid_controller_yaw.cal_to_control(error_theta)
            ow = ow1 + ow2
            
            # update state
            uav.update_state(2, ow)     # v是恒速
            # uav.update_state(ov, ow)
            time += DT
            

            x_.append(uav.x)
            y_.append(uav.y)
            yaw_.append(uav.yaw)
            v_.append(uav.v)
            w_.append(uav.w)
            t_.append(time)
        
            # show animation
            plt.cla()
            
            plt.plot(ax, ay, "^k", label="control point")
            plt.plot(bspline.refer_path[:, 0], bspline.refer_path[:, 1], "-r", label="course")
            plt.plot(x_, y_, "ob", label="trajectory")
            plt.plot(bspline.refer_path[target_ind, 0], bspline.refer_path[target_ind, 1], "xg", label="target")
            plt.axis("equal")
            plt.grid(True)
            plt.title("Time[s]: {:.2f}, v[m/s]: {:.2f}, w[rad/s]: {:.4f}".format(time, uav.v, uav.w, 2))
            plt.pause(0.0001)

        plt.subplots()
        plt.plot(bspline.refer_path[:, 0], bspline.refer_path[:, 1], "-r", label="course")
        plt.plot(x_, y_, "ob", label="trajectory")
        plt.axis("equal")
        plt.grid(True)
        plt.xlabel("x [m]")
        plt.ylabel("y [m]")

        plt.subplots()
        plt.plot(t_, v_, "-r", label="linear velocity")
        plt.grid(True)
        plt.xlabel("Time [s]")
        plt.ylabel("Speed [m/s]")
        
        plt.subplots()
        plt.plot(t_, w_, "-r", label="angular velocity")
        plt.grid(True)
        plt.xlabel("Time [s]")
        plt.ylabel("Speed [rad/s]")

        plt.show()
        
    main()