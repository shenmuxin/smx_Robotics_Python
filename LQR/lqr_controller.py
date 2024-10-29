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
    
    def get_state_xyyaw(self):
        return np.array([self.x, self.y, self.yaw])

    def get_linear_model_matrix(self, v_ref, yaw_ref):
        A = np.matrix([
            [1.0, 0.0, - self.dt * v_ref * math.sin(yaw_ref)],
            [0.0, 1.0, self.dt * v_ref * math.cos(yaw_ref)],
            [0.0, 0.0, 1.0]])
        
        B = np.matrix([
            [self.dt * math.cos(yaw_ref), 0.0],
            [self.dt * math.sin(yaw_ref), 0.0],
            [0.0, self.dt]])

        return A, B


class LQRController:

    def __init__(self, Q, R, eps, N):
        """
        @param:
            Q (np.ndarray): Q为半正定的状态加权矩阵, 通常取为对角阵；Q矩阵元素变大意味着希望跟踪偏差能够快速趋近于零；
            R (np.ndarray): R为正定的控制加权矩阵，R矩阵元素变大意味着希望控制输入能够尽可能小。
            eps (_type_) : EPS表示迭代精度
            N (_type_) : N表示迭代范围
        """
        self.Q = Q
        self.R = R
        self.eps = eps
        self.N = N
    
    def cal_Riccati(self, A, B):
        # init P matrix
        Qf = Q
        P = Qf
        # calculate Riccati matrix P
        for t in range(self.N):
            P_n = Q + A.T@P@A - A.T@P@B@ np.linalg.pinv(R + B.T@P@B) @ B.T@P@A
            if (abs(P_n - P).max() < self.eps):
                break
            P = P_n
        return P_n
    
    def cal_to_control(self, robot_state_xyyaw, refer_pos, A, B):
        """
        @param:
            robot_state_xyyaw (np.ndarray) : shape (3,) 表示机器人的当前状态
            refer_pos (np.ndarray) : shape (3,) 表示参考点的位置
        """

        x = robot_state_xyyaw - refer_pos

        P = self.cal_Riccati(A, B)
        # feedback factor
        K = -np.linalg.pinv(R + B.T @ P @ B) @ B.T @ P @ A

        # u_bar = [[v - v_ref, w - w_ref]]
        u_bar = K @ x

        return u_bar

if __name__ == "__main__":
    # test for LQR control
    Q = np.diag([3, 3, 3])
    R = np.diag([2, 2])
    EPS = 1e-4      # 迭代精度
    N = 100         # 迭代次数

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
        # initialize lqr controller
        lqr_controller = LQRController(Q=Q, R=R, eps=EPS, N=N)

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

            robot_state = uav.get_state_xyyaw()
            A, B = uav.get_linear_model_matrix(2, bspline.refer_path[target_ind, 2])

            u_bar = lqr_controller.cal_to_control(robot_state, bspline.refer_path[target_ind, 0:3], A, B)

            # ow = u_bar[0, 1] + bspline.refer_path[target_ind, -1]
            # 参考的角速度就是0
            ow = u_bar[0,1]
            
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