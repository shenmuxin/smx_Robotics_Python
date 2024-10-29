import numpy as np
import math
import matplotlib.pyplot as plt
from bspline_generator import BSplineGenerator
import cvxpy

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


class BSplineGeneratorMPC(BSplineGenerator):

    def __init__(self, T, *wargs, **kwargs):
        super().__init__(*wargs, **kwargs)
        self.T = T
        self.NX = 3
        self.NU = 2

    def calc_ref_control_input(self):
        u_ref = np.zeros((self.NU, self.T))
        u_ref[0, :] = 2     # vr = 2
        u_ref[1, :] = 0     # wr = 0
        return u_ref
    
    def calc_ref_trajectory(self, robot_state_xy, pind):
        # self.NX = 3
        # self.T = 8
        cx = self.refer_path[:, 0]
        cy = self.refer_path[:, 1]
        cyaw = self.refer_path[:, 2]
        ck = self.refer_path[:, 3]

        # shape (NX, T + 1)
        x_ref = np.zeros((self.NX, self.T + 1))

        ncourse = len(cx)

        ind, _ = self.calc_nearest_index(robot_state_xy, pind)

        if pind >= ind:
            ind = pind

        x_ref[0, 0] = cx[ind]
        x_ref[1, 0] = cy[ind]
        x_ref[2, 0] = cyaw[ind]
        
        travel = 0.0
        for i in range(T + 1):
            travel += abs(robot_state_xy[0]) * DT
            dind = int(round(travel))

            if (ind + dind) < ncourse:
                x_ref[0, i] = cx[ind + dind]
                x_ref[1, i] = cy[ind + dind]
                x_ref[2, i] = cyaw[ind + dind]
            else:
                x_ref[0, i] = cx[ncourse - 1]
                x_ref[1, i] = cy[ncourse - 1]
                x_ref[2, i] = cyaw[ncourse - 1]

        return x_ref, ind

class MPCController:

    def __init__(self, Q, R, Qf, T):
        """
        @param:
            Q (np.ndarray): Q为半正定的状态加权矩阵, 通常取为对角阵；Q矩阵元素变大意味着希望跟踪偏差能够快速趋近于零；
            R (np.ndarray): R为正定的控制加权矩阵，R矩阵元素变大意味着希望控制输入能够尽可能小。
            Qf (np.ndarray): Qf为半正定的状态加权矩阵, 用于控制终值误差
            NX (int) : 状态量的维度
            NU (int) : 控制量的维度
            T (int) : 预测区间长度
        """
        # 优化参数
        self.Q = Q
        self.R = R
        self.Qf = Qf
        # 维度参数
        self.T = T                      # horizon length
        self.NX = 3
        self.NU = 2
        # 约束参数
        self.MAX_VEL = 2                # 线速度[m/s]
        self.MAX_DSTEER = np.pi / 4     # 角速度[rad/s]
    
    def _get_ndarray_from_matrix(self, x):

        return  np.array(x).flatten()
        
    def linear_mpc_control(self, x0, x_ref, u_ref, uav : KinematicMode):
        """
        @param:
            x0 (np.ndarray) : shape (NX, ), 初始状态
            x_ref (np.ndarray) : shape (NX, T+1) 状态的参考值
            u_ref (np.ndarray) : shape (NU, T+1) 输入参考值,vr=2, wr=0
            uav (KinematicMode) : 无人机运动模型实例
        """
        x = cvxpy.Variable((self.NX, self.T + 1))
        u = cvxpy.Variable((self.NU, self.T))

        cost = 0.0          # 代价函数
        constraints = []    # 约束条件

        for t in range(self.T):
            cost += cvxpy.quad_form(u[:, t] - u_ref[:, t], self.R)      # 控制量代价

            if t != 0:
                cost += cvxpy.quad_form(x[:, t] - x_ref[:, t], self.Q)  # 状态量代价
            
            A, B = uav.get_linear_model_matrix(u_ref[0, t], x_ref[2, t])
            constraints += [x[:, t+1] - x_ref[:, t+1] == A @ (x[:,  t] - x_ref[:, t]) +
                                B @ (u[:, t] - u_ref[:, t])]
            
        cost += cvxpy.quad_form(x[:, T] - x_ref[:, T], self.Qf)  # 终值状态量代价

        # 约束条件
        constraints += [x[:, 0] == x0]
        constraints += [cvxpy.abs(u[0, :]) <= self.MAX_VEL]
        constraints += [cvxpy.abs(u[1, :]) <= self.MAX_DSTEER]

        prob = cvxpy.Problem(cvxpy.Minimize(cost), constraints)
        prob.solve(solver=cvxpy.ECOS, verbose=False)

        if prob.status == cvxpy.OPTIMAL or prob.status == cvxpy.OPTIMAL_INACCURATE:
            opt_x = self._get_ndarray_from_matrix(x.value[0, :])
            opt_y = self._get_ndarray_from_matrix(x.value[1, :])
            opt_yaw = self._get_ndarray_from_matrix(x.value[2, :])
            opt_v = self._get_ndarray_from_matrix(u.value[0, :])
            opt_w = self._get_ndarray_from_matrix(u.value[1, :])
        else:
            opt_v, opt_w, opt_x, opt_y, opt_yaw = None, None, None, None, None
        
        return opt_v, opt_w, opt_x, opt_y, opt_yaw



if __name__ == "__main__":
    # test for MPC control

    # mpc parameters
    Q = np.diag([1, 1, 1])
    Qf = Q
    R = np.diag([0.1, 0.1])
    T = 8   # horizon length

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
        # initialize MPC controller
        mpc_controller = MPCController(Q=Q, R=R, Qf=Qf, T=T)

        # set reference trajectory
        ax = [0.0, 30.0, 6.0, 20.0, 35.0]
        ay = [0.0, 0.0, 20.0, 35.0, 20.0]
        
        bspline = BSplineGeneratorMPC(T, ax, ay)

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
            x_ref, target_ind = bspline.calc_ref_trajectory([uav.x, uav.y], pind=target_ind)

            if check_goal([uav.x, uav.y], goal, target_ind, len(bspline.refer_path)-1):
                break

            x0 = uav.get_state_xyyaw()
            u_ref = bspline.calc_ref_control_input()

            opt_v, opt_w, opt_x, opt_y, opt_yaw = mpc_controller.linear_mpc_control(x0, x_ref, u_ref, uav)

            if opt_v is None and opt_w is None:
                print("Can't be solved!")
                continue

            ov = opt_v[0]
            ow = opt_w[0]       # 参考的角速度就是0
            
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