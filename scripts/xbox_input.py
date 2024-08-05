#!/usr/bin/python3

import rospy
import rospkg
import actionlib
from geometry_msgs.msg import Twist
from relaxed_ik_ros1.msg import EEVelGoals, EEPoseGoals
import transformations as T
from robot import Robot
from sensor_msgs.msg import Joy
from std_msgs.msg import Int8MultiArray, Float64MultiArray,  Float32MultiArray
from franka_msgs.msg import FrankaState
from scipy.spatial.transform import Rotation as R
from pyquaternion import Quaternion
import hiro_grasp
from klampt.math import so3
from visualization_msgs.msg import Marker
import numpy as np
import fcl
import copy
from sklearn.preprocessing import normalize
from visualization_msgs.msg import MarkerArray
from matplotlib import cm
from geometry_msgs.msg import Vector3, Point
from std_msgs.msg import ColorRGBA
import pandas as pd 
import csv
import time
import os

path_to_src = rospkg.RosPack().get_path('relaxed_ik_ros1') + '/relaxed_ik_core'
class GraspLoop:
    def __init__(self, flag, grasp_pose, grasp_apprach, 
                 home_pose=[0.30871, 0.000905, 0.48742, 0.9994651, -0.00187451, 0.0307489, -0.01097748], 
                 drop_pose = [0.23127, -0.5581, 0.31198, 0.9994651, -0.00187451, 0.0307489, -0.01097748]):
        self.grasp_dict = {
            "x_g" : grasp_pose,
            "x_a" : grasp_apprach,
            "x_h" : home_pose,
            "x_d" : drop_pose,
            "x_c" : None,
            "x_goal": None,
            "grasp": False
        }

        self.home_pose = home_pose
        self.load_file = "/home/caleb/robochem_steps/test2.txt"
        with open(self.load_file, 'r') as file:
            lines = file.readlines()
            float_arrays = [np.array(eval(line)) for line in lines]
            result_array = np.array(float_arrays)

        # print(result_array)
        self.grasp_list = result_array
        self.__release_flag = [0]
        self.__grasp_flag = [1]
        self.__wait_flag = [2]
        self.__end_flag = [3]
        
        self.cur_list_idx = 0
        self.__start_buck_2 = False
        if self.__start_buck_2:
            self.cur_list_idx = 10

        self.hiro_g = hiro_grasp.hiro_grasp()
        self.__pose_order_list = self.set_pose_order_list(flag)
        self.__cur_idx = 0 
        self.already_collided = False
        self.x_history_list = []
        self.y_history_list = []
        self.z_history_list = []
        self.max_history_len = 50

    def set_x_c(self, x_c):
        self.grasp_dict["x_c"] = x_c
    
    def set_x_g(self, x_g):
        self.grasp_dict["x_g"] = x_g
    
    def set_x_a(self, x_a):
        self.grasp_dict["x_a"] = x_a

    def get_curr_error(self, x_ee):
        return self.xyz_diff(x_ee, self.grasp_dict["x_goal"])
    
    def xyz_diff(self, start_pose, end_pose):
        goal = self.grasp_dict["x_goal"]
        difference = [0.0, 0.0, 0.0]
        difference[0] = goal[0] - start_pose[0]
        difference[1] = goal[1] - start_pose[1]
        difference[2] = goal[2] - start_pose[2]
        return difference
    
    def add_to_xyz_history(self, x_ee):
        self.x_history_list.append(x_ee[0])
        self.y_history_list.append(x_ee[1])
        self.z_history_list.append(x_ee[2])
        if len(self.x_history_list) > self.max_history_len:
            self.x_history_list.pop(0)
            self.y_history_list.pop(0)
            self.z_history_list.pop(0)

    def get_franka_xyz_history(self):
        return self.x_history_list, self.y_history_list, self.z_history_list

    def set_pose_order_list(self, flag):
        self.flag = flag
        if self.flag == "linear":
            self.grasp_dict["x_goal"] = self.grasp_dict["x_g"]
            return ["x_g", "grasp", "x_h", "x_d", "x_h"]
            
        elif self.flag == "l-shaped":
            self.grasp_dict["x_goal"] = self.grasp_dict["x_a"]
            return ["x_a", "x_g", "grasp", "x_h", "x_d", "x_h"]
        
        elif self.flag == "list":
            self.grasp_dict["x_goal"] = self.grasp_list[self.cur_list_idx]
            return None
        
    def update_grasp_goal(self):
        self.grasp_dict["x_goal"] = self.grasp_dict[self.__pose_order_list[self.__cur_idx]]

    def set_grasp_width(self, width):
        self.hiro_g.set_grasp_width(width)
    
    def grasp(self):
        self.hiro_g.grasp()


    def get_error_xyz(self, x_ee):
        return self.xyz_diff(x_ee, self.grasp_dict["x_goal"])

    def __inc_pose_list(self):
        self.cur_list_idx += 1 
        if self.cur_list_idx >= len(self.grasp_list):
            exit()
        self.cur_list_idx = self.cur_list_idx % len(self.grasp_list)
        self.grasp_dict["x_goal"] = self.grasp_list[self.cur_list_idx]

    def wait_time_seconds(self, s):
        rospy.sleep(s)

    def check_next_state(self, error):
        error_bound = 0.012
        error_sum = sum([abs(elem) for elem in error])
        ret = False
        
        # print("error sum: ", error_sum)
        # print("error bound", error_bound)
        if error_sum < error_bound:
            if not self.__pose_order_list:
                self.__inc_pose_list()
                if self.grasp_list[self.cur_list_idx][0] == self.__grasp_flag:
                    self.hiro_g.set_grasp_width(0.03)
                    self.hiro_g.grasp()
                    rospy.sleep(2.0)
                    self.__inc_pose_list()
                elif self.grasp_list[self.cur_list_idx][0] == self.__release_flag:
                    self.hiro_g.open()
                    # print("opening gripper")
                    rospy.sleep(2.0)
                    self.__inc_pose_list()
                elif self.grasp_list[self.cur_list_idx][0] == self.__end_flag:
                    # print("QUIT EVERYTHING")
                    exit()

                if len(self.grasp_list[self.cur_list_idx]) == 2 and self.grasp_list[self.cur_list_idx][0] == self.__wait_flag:
                    sleep_time = 300
                    self.wait_time_seconds(self.grasp_list[self.cur_list_idx][1])
                    self.__inc_pose_list()
                return ret
            if self.__pose_order_list[self.__cur_idx] == "x_d":
                self.hiro_g.open()
                rospy.sleep(2.0)
            ret = self.inc_state()
        if not self.__pose_order_list:
            return ret
        elif self.__pose_order_list[self.__cur_idx] == "grasp":
            ret = self.inc_state()
            self.hiro_g.grasp()
            rospy.sleep(2.0)
        return ret
    
    def __wait_for_new_grasp_pose(self):
        while True:
            self.__check_new_grasp()
            rospy.sleep(1)

    def inc_state(self):
        dont_loop = True
        if dont_loop and self.cur_list_idx == len(self.__pose_order_list) -1 :
            # print('Quit!!!')
            exit()
        self.__cur_idx += 1 
        self.__cur_idx = self.__cur_idx % (len(self.__pose_order_list))
            
        self.grasp_dict["x_goal"] = self.grasp_dict[self.__pose_order_list[self.__cur_idx]]
        if self.__cur_idx == 0:
            self.already_collided = False
            return True
        return False

class XboxInput:
    def __init__(self, flag):
        self.flag = flag
        self.save = False
        self.data_array = np.array([[]])
        self.start_time = 0.0
        self.end_time = 0.0
        self.total_time = 0.0

        default_setting_file_path = path_to_src + '/configs/settings.yaml'

        setting_file_path = rospy.get_param('setting_file_path')
        if setting_file_path == '':
            setting_file_path = default_setting_file_path

        self.robot = Robot(setting_file_path)
        self.grasped = False
        self.final_location = False
        self.made_loop = False
        self.grasp_pose = [0,0,0,0,0,0,0]
        self.x_a = [0,0,0,0,0,0,0]
        self.grasp_loop = GraspLoop(self.flag, self.grasp_pose, self.x_a)

        self.ee_pose_goals_pub = rospy.Publisher('relaxed_ik/ee_vel_goals', EEVelGoals, queue_size=1)
        self.hiro_ee_vel_goals_pub = rospy.Publisher('relaxed_ik/hiro_ee_vel_goals', Float64MultiArray, queue_size=1)
        self.ee_vel_goals_pub = rospy.Publisher('relaxed_ik/ee_pose_goals', EEPoseGoals, queue_size=1)
        self.pos_stride = 0.003
        self.rot_stride = 0.0125
        self.p_t = 0.02
        self.p_r = 0.0075
        self.rot_error = [0.0, 0.0, 0.0]

        self.z_offset = 0.0
        self.y_offset = 0.0
        self.x_offset = 0.0
        self.seq = 1
        
        self.linear = [0,0,0]
        self.angular = [0,0,0]
        self.joy_data = None
        self.start_grasp = False
        self.prev_fr_euler = [0, 0, 0]
        self.grasp_midpoint = [0,0,0,0,0,0,0]

        self.grip_cur = 0.08
        self.grip_inc = 0.02
        self.grip_max = 0.08
        self.grip_min = 0.01
        self.prev_pres = 0

        self.fr_position = [0.0, 0.0, 0.0]
        self.fr_rotation_matrix = [[0.0, 0.0, 0.0],
                                   [0.0, 0.0, 0.0],
                                   [0.0, 0.0, 0.0]]
        self.fr_state = False
        self.error_state = [0.0, 0.0, 0.0]
        self.cur_list_idx = 0

        self.fr_is_neg = False
        self.fr_set = False
        self.got_prev_sign = False
        self.og_set = False
        self.in_collision = False
        self.og_x_a = [1, 1, 1]
        self.msg_obj_to_line = 0.0
        self.x_c = [0,0,0,0,0,0,0]
        self.last_grasp_time = time.time()
        self.save_file = "/home/caleb/robochem_steps/test.txt"

        rospy.Subscriber("/mid_grasp_point", Float32MultiArray, self.grasp_midpoint_callback)
        rospy.Subscriber("joy", Joy, self.joy_cb)
        rospy.Subscriber("final_grasp", Float32MultiArray, self.subscriber_callback)
        rospy.Subscriber("/franka_state_controller/franka_states", FrankaState, self.fr_state_cb)
        rospy.Subscriber("/estimated_approach_frame", Float32MultiArray, self.l_shaped_callback)
        rospy.sleep(1.0)
        rospy.Timer(rospy.Duration(0.01), self.timer_callback)

    def joy_cb(self, data):
        self.joy_data = data
        if abs(self.joy_data.axes[1]) > 0.1:
            self.linear[0] -= self.pos_stride * self.joy_data.axes[1]
        if abs(self.joy_data.axes[0]) > 0.1:
            self.linear[1] -= self.pos_stride * self.joy_data.axes[0]
        if abs(self.joy_data.axes[4]) > 0.1:
            self.linear[2] += self.pos_stride * self.joy_data.axes[4]

        if abs(self.joy_data.axes[6]) > 0.1:
            self.angular[0] += self.rot_stride * self.joy_data.axes[6]
        if abs(self.joy_data.axes[7]) > 0.1:
            self.angular[1] += self.rot_stride * self.joy_data.axes[7]
        if abs(self.joy_data.buttons[4]) > 0.1:
            self.angular[2] += self.rot_stride
        if abs(self.joy_data.buttons[5]) > 0.1:
            self.angular[2] -= self.rot_stride

        y_press = data.buttons[3]
        if y_press:
            print("Franka Pose: ", self.franka_pose)

        # Start is button 7
        start = data.buttons[7]
        back = data.buttons[6]

        if start:
            self.save = True
            self.start_time = time.perf_counter()
        if back:
            f = open("/home/caleb/catkin_ws/src/relaxed_ik_ros1/scripts/xyz_data.csv", "w")
            f.truncate()
            f.close()    
            self.save = False
            self.end_time = time.perf_counter()
            self.total_time = self.end_time - self.start_time
            self.data_array = np.append(self.grasp_list, self.total_time)
            print('\n\n SAVING TO CSV \n\n', f'\n\n {self.total_time:0.4f} \n\n')
            np.savetxt("/home/caleb/catkin_ws/src/relaxed_ik_ros1/scripts/xyz_data.csv", self.data_array, delimiter=",")
            exit()

        a = data.buttons[0]
        b = data.buttons[1]

        if a:
            self.grip_cur += self.grip_inc
        elif b:
            self.grip_cur -= self.grip_inc
        if a or b:
            if (time.time() - self.last_grasp_time) >= 2.0:
                self.move_gripper()
                self.last_grasp_time = time.time()
        if self.grip_cur > self.grip_max: self.grip_cur = self.grip_max
        if self.grip_cur < self.grip_min: self.grip_cur = self.grip_min

    def move_gripper(self):
        print('In move gripper grip_cur: ', self.grip_cur)
        self.grasp_loop.set_grasp_width(self.grip_cur)
        self.grasp_loop.grasp()

    def on_release(self):
        self.linear = [0,0,0]
        self.angular = [0,0,0]

    def _xyz_diff(self, start_pose, end_pose):
        difference = [0.0, 0.0, 0.0]
        difference[0] = end_pose[0] - start_pose[0]
        difference[1] = end_pose[1] - start_pose[1]
        difference[2] = end_pose[2] - start_pose[2]
        return difference

    def calc_error(self):
        twist = Twist()
        twist.linear.x = self.error_state[0] * self.p_t
        twist.linear.y = self.error_state[1] * self.p_t
        twist.linear.z = self.error_state[2] * self.p_t
        twist.angular.x = self.rot_error[0] * self.p_r 
        twist.angular.y = self.rot_error[1] * self.p_r
        twist.angular.z = self.rot_error[2] * self.p_r
        return twist
    
    def get_hiro_error_msg(self, grasp_quat):
        ret = []
        for x in self.error_state:
            ret.append(x * self.p_t)
        ret.append(grasp_quat[3])
        ret.append(grasp_quat[0])
        ret.append(grasp_quat[1])
        ret.append(grasp_quat[2])
        return ret
    
    def angle_error(self, gr, fr):
        gr_e = gr.as_euler("xyz", degrees=False)
        fr_e = fr.as_euler("xyz", degrees=False)

        gr_e = np.array(gr_e)
        fr_e = np.array(fr_e)      

        gr_e = gr_e / np.linalg.norm(gr_e)
        fr_e = fr_e / np.linalg.norm(fr_e) 

        ax = np.cross(gr_e, fr_e)
        ang = np.arctan2(np.linalg.norm(ax), np.dot(gr_e, fr_e))
        our_ang = so3.rotation(ax, ang)

        test_r = R.from_matrix(np.reshape(our_ang, newshape=[3,3]))
        output = test_r.as_euler("xyz", degrees=False)
        return output

    def calc_rotation_sign(self, fr_euler, gr_euler):
        fr_rot = R.from_euler("xyz", fr_euler, degrees=False)                   
        fr_quat = fr_rot.as_quat()
        fr_euler = fr_rot.as_euler('xyz', degrees=False)

        return fr_euler, fr_quat
    
    def quaterion_error(self, fr_quat, grasp_quat, fr_euler, grasp_euler):
        q1 = Quaternion(fr_quat[3], fr_quat[0], fr_quat[1], fr_quat[2])
        q2 = Quaternion(grasp_quat[3], grasp_quat[0], grasp_quat[1], grasp_quat[2])

        q_list = []
        for q in Quaternion.intermediates(q1, q2, 1, include_endpoints=True):
            rot = R.from_quat([q.unit.x, q.unit.y, q.unit.z, q.unit.w])
            euler = rot.as_euler('xyz', degrees=False)
            q_list.append(euler)

        if self.got_prev_sign:
            self.got_prev_sign = True
        else:
            for x in range(3):
                if (abs(grasp_euler[x] - fr_euler[x]) > abs(grasp_euler[x] + fr_euler[x])):
                    if (abs(self.prev_fr_euler[x] - fr_euler[x]) > abs(self.prev_fr_euler[x] + fr_euler[x])):
                        fr_euler[x] = -fr_euler[x]
        for x in range(3):
                self.prev_fr_euler[x] = fr_euler[x]
        return q_list[-1], fr_euler
    
    def get_unit_line(self, a, b):
        a_np = np.asarray(a)
        b_np = np.asarray(b)

        line = b_np - a_np
        line = normalize([line], axis=1, norm='l1')
        return line[0]

    def wait_for_new_grasp(self):
        self.prev_grasp = self.grasp_pose[0]
        self.made_loop = False
        while True:
            if self.__check_for_new_grasp():
                return
            rospy.sleep(1)

    def __check_for_new_grasp(self):
        if self.prev_grasp != self.grasp_pose[0]:
            return True
        return False

    def move_through_list(self):
        print(self.grasp_loop.grasp_list)
        if self.fr_state:
            if not self.made_loop:
                self.made_loop = True
                self.grasp_loop = GraspLoop(self.flag, self.grasp_pose, self.og_x_a)
                print("Made GraspLoop in list")

            line = self.get_unit_line(self.grasp_pose[:3], self.og_x_a[:3])
            hiro_msg = Float64MultiArray()
            # pose_msg = EEPoseGoals()
            self.error_state = self.grasp_loop.get_curr_error(self.fr_position)
            
            hiro_msg.data = self.get_hiro_error_msg(self.grasp_loop.grasp_dict["x_goal"][3:])
            self.clamp_linear_position()

            hiro_msg.data[3] = self.grasp_loop.grasp_dict["x_goal"][6]
            hiro_msg.data[4] = self.grasp_loop.grasp_dict["x_goal"][3]
            hiro_msg.data[5] = self.grasp_loop.grasp_dict["x_goal"][4]
            hiro_msg.data[6] = self.grasp_loop.grasp_dict["x_goal"][5]

            for x in line:
                hiro_msg.data.append(x)
                
            hiro_msg.data.append(self.msg_obj_to_line)
            hiro_msg.data.append(self.og_x_a[0])
            hiro_msg.data.append(self.og_x_a[1])
            hiro_msg.data.append(self.og_x_a[2])
            hiro_msg.data.append(self.grasp_pose[0])
            hiro_msg.data.append(self.grasp_pose[1])
            hiro_msg.data.append(self.grasp_pose[2])

            print(hiro_msg.data.copy())

            self.ee_vel_goals_pub.publish(hiro_msg)
            self.on_release()
            if self.grasp_loop.check_next_state(self.error_state):
                self.wait_for_new_grasp()

    def linear_movement(self):
        if self.start_grasp and self.fr_state:
            if not self.made_loop:
                self.made_loop = True
                self.grasp_loop = GraspLoop(self.flag, self.grasp_pose, self.og_x_a)

            line = self.get_unit_line(self.grasp_pose[:3], self.og_x_a[:3])
            hiro_msg = Float64MultiArray()
            self.error_state = self.grasp_loop.get_curr_error(self.fr_position)
            
            hiro_msg.data = self.get_hiro_error_msg(self.grasp_loop.grasp_dict["x_goal"][3:])
            for x in line:
                hiro_msg.data.append(x)
            hiro_msg.data.append(self.msg_obj_to_line)
            hiro_msg.data.append(self.og_x_a[0])
            hiro_msg.data.append(self.og_x_a[1])
            hiro_msg.data.append(self.og_x_a[2])
            hiro_msg.data.append(self.grasp_pose[0])
            hiro_msg.data.append(self.grasp_pose[1])
            hiro_msg.data.append(self.grasp_pose[2])

            self.hiro_ee_vel_goals_pub.publish(hiro_msg)
            self.on_release()
            if self.grasp_loop.check_next_state(self.error_state):
                self.wait_for_new_grasp()

    def l_shaped_movement(self):
        if self.start_grasp and self.fr_state: 
            if not self.made_loop:
                self.made_loop = True
                self.grasp_loop = GraspLoop(self.flag, self.grasp_pose, self.og_x_a)

            line = self.get_unit_line(self.grasp_pose[:3], self.og_x_a[:3])
            hiro_msg = Float64MultiArray()
            self.error_state = self.grasp_loop.get_curr_error(self.fr_position)
            hiro_msg.data = self.get_hiro_error_msg(self.grasp_loop.grasp_dict["x_goal"][3:])
            for x in line:
                hiro_msg.data.append(x)
            hiro_msg.data.append(self.msg_obj_to_line)
            hiro_msg.data.append(self.og_x_a[0])
            hiro_msg.data.append(self.og_x_a[1])
            hiro_msg.data.append(self.og_x_a[2])
            hiro_msg.data.append(self.grasp_pose[0])
            hiro_msg.data.append(self.grasp_pose[1])
            hiro_msg.data.append(self.grasp_pose[2])

            self.hiro_ee_vel_goals_pub.publish(hiro_msg)
            self.on_release()

            if self.grasp_loop.check_next_state(self.error_state):
                self.wait_for_new_grasp()

    def clamp_linear_position(self):
        z_max = 0.7
        z_min = 0.02
        y_max = 0.7
        y_min = -0.7
        x_max = 0.6
        x_min = 0.0

        # Set X bounds
        if self.fr_position[0] > x_max and self.linear[0] > 0: self.linear[0] = 0
        elif self.fr_position[0] < x_min and self.linear[0] < 0: self.linear[0] = 0

        # Set Y bounds
        if self.fr_position[1] > y_max and self.linear[1] > 0: self.linear[1] = 0
        elif self.fr_position[1] < y_min and self.linear[1] < 0: self.linear[1] = 0

        # Set Z bounds
        if self.fr_position[2] > z_max and self.linear[2] > 0: self.linear[2] = 0
        elif self.fr_position[2] < z_min and self.linear[2] < 0: self.linear[2] = 0

    def xbox_input(self):
        # print('xbox loop time: ', time.time())
        msg = EEVelGoals()
        if not self.og_set:
            print('Set OG goal')
            self.og_set = True
            self.og_trans = copy.deepcopy(self.x_a[:3])
            self.og_quat = copy.deepcopy(self.x_a[3:])

        if not self.made_loop:
            # print('Xbox flag:', self.flag)
            print('Loop made')
            self.made_loop = True
            self.grasp_loop = GraspLoop(self.flag, self.grasp_pose, self.og_x_a)
            
        fr_r = R.from_matrix(self.fr_rotation_matrix)
        fr_quat = fr_r.as_quat()
        fr_e = fr_r.as_euler("xyz", degrees=False)
        fr_r = R.from_euler("xyz", fr_e, degrees=False)

        self.franka_pose = []
        self.franka_pose.extend(self.fr_position)
        self.franka_pose.extend(fr_quat)
        print(self.franka_pose)

        self.clamp_linear_position()
        for i in range(self.robot.num_chain):
            twist = Twist()
            tolerance = Twist()
            twist.linear.x = self.linear[0]
            twist.linear.y = self.linear[1]
            twist.linear.z = self.linear[2]
            twist.angular.x = self.angular[0]
            twist.angular.y = self.angular[1]
            twist.angular.z = self.angular[2]
            tolerance.linear.x = 0.0
            tolerance.linear.y = 0.0
            tolerance.linear.z = 0.0
            tolerance.angular.x = 0.0
            tolerance.angular.y = 0.0
            tolerance.angular.z = 0.0

            msg.ee_vels.append(twist)
            msg.tolerances.append(tolerance)
        # print('Msg', msg)
        self.ee_vel_goals_pub.publish(msg)
        self.on_release()

    def timer_callback(self, event):
        if self.flag == "linear":
            self.linear_movement()
        elif self.flag == "xbox":
            self.xbox_input()
        elif self.flag == "l-shaped":
            self.l_shaped_movement()
        elif self.flag == "list":
            self.move_through_list()

    def subscriber_callback(self, data):
        zero = [0.0, 0.0, 0.0]
        if data.data != zero and not self.grasped:
            self.grasp_pose = list(data.data)
            self.grasp_pose[2] = self.grasp_pose[2] -  self.z_offset
            self.grasp_pose[1] = self.grasp_pose[1] -  self.y_offset
            self.grasp_pose[0] = self.grasp_pose[0] -  self.x_offset
    
    def l_shaped_callback(self, data):
        zero = [0.0, 0.0, 0.0]
        if data.data != zero and not self.grasped:
            self.start_grasp = True
            self.x_a = list(data.data)
            self.x_a[2] = self.x_a[2] - self.z_offset
            self.x_a[1] = self.x_a[1] - self.y_offset
            self.x_a[0] = self.x_a[0] - self.x_offset
            self.og_x_a = self.x_a

    def grasp_midpoint_callback(self, data):
        self.grasp_midpoint = list(data.data)
            
    def fr_state_cb(self, data):
        self.fr_position = [data.O_T_EE[12], data.O_T_EE[13], data.O_T_EE[14]]
        self.fr_state = True
        temp_rot_mat = np.array([
            [data.O_T_EE[0], data.O_T_EE[1], data.O_T_EE[2]],
            [data.O_T_EE[4], data.O_T_EE[5], data.O_T_EE[6]],
            [data.O_T_EE[8], data.O_T_EE[9], data.O_T_EE[10]]
        ])
        self.fr_rotation_matrix = temp_rot_mat
        
if __name__ == '__main__':
    flag = rospy.get_param("/xbox_input/flag")
    rospy.init_node('xbox_input')
    xController = XboxInput(flag=flag)
    rospy.spin()

    