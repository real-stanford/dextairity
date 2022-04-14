import threading
import numpy as np

try:
    from . import realur5_utils
except:
    import realur5_utils

from copy import deepcopy
from time import sleep, time

import rtde_control
import rtde_io
import urx


def clamp_angles(angle, up=np.pi, down=-np.pi):
    angle[angle > up] -= up
    angle[angle < down] += down
    return angle


class UR5MoveTimeoutException(Exception):
    def __init__(self):
        super().__init__('UR5 Move Timeout')


class UR5:
    # joint position and tool pose tolerance (epsilon) for blocking calls
    # tool_pose_eps = np.array([0.005, 0.005, 0.005, 0.001, 0.001, 0.001])
    tool_pose_eps = np.array([0.02, 0.02, 0.02, 0.005, 0.005, 0.005])

    GROUPS = {
        'arm': ['shoulder_pan_joint',
                'shoulder_lift_joint',
                'elbow_joint',
                'wrist_1_joint',
                'wrist_2_joint',
                'wrist_3_joint'],
        'gripper': ['finger_joint',  # TODO
                    'left_inner_knuckle_joint',
                    'left_inner_finger_joint',
                    'right_outer_knuckle_joint',
                    'right_inner_knuckle_joint',
                    'right_inner_finger_joint']
    }

    GROUP_INDEX = {
        'arm': [1, 2, 3, 4, 5, 6],
        'gripper': [9, 11, 13, 14, 16, 18]  # TODO
    }

    LINK_COUNT = 10

    LIE = [0, 0, 0, 0, 0, 0]
    UP = [0, -1.5707, 0, -1.5707, 0, 0]
    EE_LINK_NAME = 'ee_link'
    TIP_LINK = "ee_link"
    BASE_LINK = "base_link"
    ARM = "arm"
    GRIPPER = "gripper"
    EE_TIP_LINK = 7
    TOOL_LINK = 6

    # this is read from moveit_configs joint_limits.yaml
    MOVEIT_ARM_MAX_VELOCITY = [3.15, 3.15, 3.15, 3.15, 3.15, 3.15]

    LOWER_LIMIT = np.array([-2, -2, -1, -2, -2, -2]) * np.pi
    UPPER_LIMIT = np.array([2, 2, 1, 2, 2, 2]) * np.pi

    JOINT_EPSILON = 1e-2

    def __init__(self,
                 tcp_ip,
                 velocity=0.6,
                 acceleration=0.4,
                 tcp_port=30002,
                 rtc_port=30003,
                 home_joint_state=None,
                 gripper: realur5_utils.Gripper = None):
        self.RESET = home_joint_state if home_joint_state is not None else [-np.pi, -np.pi / 2, np.pi / 2, -np.pi / 2, -np.pi / 2, 0]
        self.tcp_ip = tcp_ip
        self.velocity = velocity
        self.acceleration = acceleration

        self.create_tcp_sock_fn = lambda: realur5_utils.connect(tcp_ip, tcp_port)
        self.create_rtc_sock_fn = lambda: realur5_utils.connect(tcp_ip, rtc_port)
        self.tcp_sock = self.create_tcp_sock_fn()
        self.rtc_sock = self.create_rtc_sock_fn()

        self.state = realur5_utils.UR5State(
            self.create_tcp_sock_fn,
            self.create_rtc_sock_fn)

        # Start thread to perform robot actions in parallel to main thread
        self.thread_do_grasp, self.thread_is_grasping = False, False
        # grasping primitive parameters
        self.thread_grasp_params = [None, None]
        self.thread_grasp_success = None
        self.thread_do_toss, self.thread_is_tossing = False, False
        # tossing primitive parameters
        self.thread_toss_params = [None, None, None]
        self.gripper = gripper
        if self.gripper is not None:
            tcp_msg = 'set_tcp(p[%f,%f,%f,%f,%f,%f])\n'\
                % tuple(self.gripper.tool_offset)
        else:
            tcp_msg = 'set_tcp(p[0,0,0,0,0,0])\n'
        self.tcp_sock.send(str.encode(tcp_msg))
        sleep(1)
        self.use_pos = False
        self.time_start_command = None
        self.action_timeout = 20

    # Move joints to specified positions or move tool to specified pose

    def movej(self, **kwargs):
        return self.move('j', **kwargs)

    # def movel(self, **kwargs):
    #     return self.move('l', **kwargs)

    
    def movel(self, p, speed=0.3, acceleration=0.2, blocking=True):
        self.movej(use_pos=True, params=p, blocking=blocking, j_acc=acceleration, j_vel=speed)

    def check_pose_reachable(self, pose):
        return np.linalg.norm(pose[:2]) < 0.90\
            and np.linalg.norm(pose[:2]) > 0.30

    def move(self, move_type, params,
             blocking=True,
             j_acc=None, j_vel=None,
             times=0.0, blend=0.0,
             clear_state_history=False, use_pos=False):
        self.use_pos = use_pos
        params = deepcopy(params)
        if not j_acc:
            j_acc = self.acceleration
        if not j_vel:
            j_vel = self.velocity
        multiple_params = any(isinstance(item, list) for item in params)
        if type(params) != np.array:
            params = np.array(params)
        if multiple_params:
            def match_param_len(var):
                if not isinstance(var, list):
                    return [var] * len(params)
                elif len(var) != len(params):
                    raise Exception()
                return var
            j_vel = match_param_len(j_vel)
            j_acc = match_param_len(j_acc)
            move_type = match_param_len(move_type)
            times = match_param_len(times)
            blend = match_param_len(blend)
        else:
            params = [params]
            assert not isinstance(j_vel, list) and \
                not isinstance(j_acc, list)
            j_vel = [j_vel]
            j_acc = [j_acc]
            move_type = [move_type]
            times = [times]
            blend = [blend]
        if use_pos:
            # check all poses are reachable
            if not all([self.check_pose_reachable(pose=param)
                        for param in params]):
                return False
        self.curr_targ = params[-1]
        if self.use_pos:
            self.curr_targ[-3:] = clamp_angles(self.curr_targ[-3:])

        # Move robot
        tcp_msg = 'def process():\n'
        tcp_msg += f' stopj({j_acc[0]})\n'
        for m, p, a, v, t, r in zip(
                move_type, params, j_acc, j_vel, times, blend):
            tcp_msg += ' move%s(%s[%f,%f,%f,%f,%f,%f],a=%f,v=%f,t=%f,r=%f)\n' \
                % (m, 'p' if use_pos else '',
                    p[0], p[1], p[2], p[3], p[4], p[5],
                    a, v, t, r)
        tcp_msg += "set_digital_out(6,True)\n"
        tcp_msg += 'end\n'
        if clear_state_history:
            self.state.clear()
            while not len(self.state):
                sleep(0.001)
        self.tcp_sock.send(str.encode(tcp_msg))

        # If blocking call, pause until robot stops moving
        if blocking:
            self.time_start_command = time()
            while True:
                print('\r ', end='')  # IO so scheduler prioritizes process
                if self.reached_target():
                    self.time_start_command = None
                    return True
                elif self.is_timed_out():
                    self.time_start_command = None
                    raise UR5MoveTimeoutException
        return True

    def is_timed_out(self):
        if self.time_start_command is None:
            return False
        return float(time()-self.time_start_command) > self.action_timeout

    def reached_target(self):
        if not (self.state.get_j_vel() < 1e-1).all():
            return False
        if self.use_pos:
            tool_pose = self.state.get_ee_pose()
            tool_pose_mirror = np.asarray(list(tool_pose))
            tool_pose_mirror[-3:] = clamp_angles(tool_pose_mirror[-3:])
            tool_pose_mirror[3:6] = -tool_pose_mirror[3:6]
            return (
                (np.abs(tool_pose-self.curr_targ)
                    < self.tool_pose_eps).all() or
                (np.abs(tool_pose_mirror-self.curr_targ)
                 < self.tool_pose_eps).all())\
                and np.sum(self.state.get_j_vel()) < 0.01
        else:
            return (np.abs((self.state.get_j_pos() - self.curr_targ))
                    < UR5.JOINT_EPSILON).all()

    # Move joints to home joint configuration
    def homej(self, **kwargs):
        self.movej(params=self.RESET, **kwargs)

    def home(self, **kwargs):
        self.movej(params=self.RESET, **kwargs)

    def reset(self):
        self.homej()

    def get_j_pos(self):
        return self.state.get_j_pos()

class UR5URX:
    def __init__(self, ip, home_joint, gripper=None):
        self.robot = urx.Robot(ip)
        if gripper == 'blower':
            self.robot.set_tcp([-0.14, 0.052, 0.13, 0, 0, 0])
            # self.robot.set_tcp([-0.14-0.023, 0.052, 0.13, 0, 0, 0])
            # self.robot.set_tcp([-0.14-0.063, 0.052, 0.13, 0, 0, 0])

            self.robot.set_payload(1.5, (0, 0, 0.1))
        self.home_joint = home_joint
        
    def home(self, speed=1.5, acceleration=1, blocking=True):
        self.robot.movej(self.home_joint, acceleration, speed, False)

    def movel(self, p, speed=1.5, acceleration=1, blocking=True):
        self.robot.movel(p, acceleration, speed, blocking)

    def open_gripper(self):
        self.robot.set_digital_out(5, True)
        
    def close_gripper(self):
        self.robot.set_digital_out(5, False)
        
        

class UR5RTDE:
    def __init__(self, ip, gripper=None):
        self.rtde_c = rtde_control.RTDEControlInterface(ip)
        if gripper == 'rg2':
            self.rtde_i = rtde_io.RTDEIOInterface(ip)
        self.gripper = gripper
        self.home_joint = [np.pi/2, -np.pi / 2, np.pi / 2, -np.pi / 2, -np.pi / 2, 0]
        if self.gripper is None:
            self.rtde_c.setTcp([0, 0, 0, 0, 0, 0])
        elif gripper == 'rg2':
            self.rtde_c.setTcp([0, 0, 0.195, 0, 0, 0])
            self.rtde_c.setPayload(1.043, [0, 0, 0.08])
        else:
            self.rtde_c.setTcp(self.gripper.tool_offset)
            self.rtde_c.setPayload(self.gripper.mass, [0, 0, 0.08])
    
    def home(self, speed=1.5, acceleration=1, blocking=True):
        self.rtde_c.moveJ(self.home_joint, speed, acceleration, not blocking)

    def movej(self, q, speed=1.5, acceleration=1, blocking=True):
        self.rtde_c.moveJ(q, speed, acceleration, not blocking)
    
    def movel(self, p, speed=1.5, acceleration=1, blocking=True):
        if isinstance(p[0], float):
            self.rtde_c.moveL(p, speed, acceleration, not blocking)
        elif isinstance(p[0], list):
            for x in p:
                x.extend([speed, acceleration, 0])
            self.rtde_c.moveL(p, not blocking)
    
    def movej_ik(self, p, speed=1.5, acceleration=1, blocking=True):
        self.rtde_c.moveJ_IK(p, speed, acceleration, not blocking)

    def open_gripper(self, sleep_time=1):
        if self.gripper == 'rg2':
            self.rtde_i.setToolDigitalOut(0, False)
        else:
            self.gripper.open(blocking=True)
        sleep(sleep_time)
        
    def close_gripper(self, sleep_time=1):
        if self.gripper == 'rg2':
            self.rtde_i.setToolDigitalOut(0, True)
        else:
            self.gripper.close(blocking=False)
        sleep(sleep_time)
    

class UR5PairRTDE:
    def __init__(self, left_ur5, right_ur5):
        self.left_ur5 = left_ur5
        self.right_ur5 = right_ur5

    def home(self, speed=1.5, acceleration=1, blocking=True):
        t1 = threading.Thread(target=self.left_ur5.home, args=(speed, acceleration, blocking))
        t2 = threading.Thread(target=self.right_ur5.home, args=(speed, acceleration, blocking))
        t1.start()
        t2.start()
        t1.join()
        t2.join()

    def movej(self, q_left, q_right, speed=1.5, acceleration=1, blocking=True):
        t1 = threading.Thread(target=self.left_ur5.movej, args=(q_left, speed, acceleration, blocking))
        t2 = threading.Thread(target=self.right_ur5.movej, args=(q_right, speed, acceleration, blocking))
        t1.start()
        t2.start()
        t1.join()
        t2.join()


    def movej_ik(self, p_left, p_right, speed=1.5, acceleration=1, blocking=True):
        t1 = threading.Thread(target=self.left_ur5.movej_ik, args=(p_left, speed, acceleration, blocking))
        t2 = threading.Thread(target=self.right_ur5.movej_ik, args=(p_right, speed, acceleration, blocking))
        t1.start()
        t2.start()
        t1.join()
        t2.join()


    def movel(self, p_left, p_right, speed=1.5, acceleration=1, blocking=True):
        t1 = threading.Thread(target=self.left_ur5.movel, args=(p_left, speed, acceleration, blocking))
        t2 = threading.Thread(target=self.right_ur5.movel, args=(p_right, speed, acceleration, blocking))
        t1.start()
        t2.start()
        t1.join()
        t2.join()

    def open_gripper(self, sleep_time=1):
        self.left_ur5.open_gripper(0)
        self.right_ur5.open_gripper(0)
        sleep(sleep_time)

    def close_gripper(self, sleep_time=1):
        self.left_ur5.close_gripper(0)
        self.right_ur5.close_gripper(0)
        sleep(sleep_time)