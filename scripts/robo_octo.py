#!/usr/bin/python3

from copy import deepcopy 
import rospy
import rospkg
import actionlib
from geometry_msgs.msg import Twist, WrenchStamped, Point, Quaternion, Pose
from relaxed_ik_ros1.msg import EEVelGoals, EEPoseGoals, GUIMsg
import transformations as T
from robot import Robot
from sensor_msgs.msg import Joy
import franka_gripper.msg
from jsk_rviz_plugins.msg import OverlayText  
from std_msgs.msg import Int8MultiArray, Float64MultiArray,  Float32MultiArray, String, Int32, Bool
from franka_msgs.msg import FrankaState
from scipy.spatial.transform import Rotation as R
import hiro_grasp
from klampt.math import so3
import math
from visualization_msgs.msg import Marker
import numpy as np
from franka_example_controllers.msg import JointTorqueComparison
import fcl
import copy
from sklearn.preprocessing import normalize
from visualization_msgs.msg import MarkerArray
from matplotlib import cm
from geometry_msgs.msg import Vector3, Point
from std_msgs.msg import ColorRGBA
import time
from sensor_msgs.msg import JointState

def run_once(f):
    def wrapper(*args, **kwargs):
        if not wrapper.has_run:
            wrapper.has_run = True
            return f(*args, **kwargs)
    wrapper.has_run = False
    return wrapper

path_to_src = rospkg.RosPack().get_path('relaxed_ik_ros1') + '/relaxed_ik_core'
class GraspLoop:
    def __init__(self, flag, gui_flag, grasp_pose, grasp_apprach,
                 home_pose=[0.30871, 0.000905, 0.48742, 0.9994651, -0.00187451, 0.0307489, -0.01097748], 
                 drop_pose = [0.23127, -0.5581, 0.31198, 0.9994651, -0.00187451, 0.0307489, -0.01097748], 
                 cone_radius=0.25, cone_height=0.25):
        # self.grasp_list = []
        self.load_file = "/home/caleb/robochem_steps/transfer_wo_jig.txt"
        # self.octo_file = "/home/caleb/robochem_steps/fixed_octo_action.txt"
        self.octo_file = "/home/caleb/robochem_steps/octo_action.txt"

        self.read_file()
        
        self.grasp_dict = {
            "x_g" : grasp_pose,
            "x_a" : grasp_apprach,
            "x_h" : home_pose,
            "x_d" : drop_pose,
            "x_c" : None,
            "c_r" : cone_radius,
            "c_h" : cone_height,
            "x_goal": None,
            "grasp": False
        }

        self.gui_flag = gui_flag
        self.home_pose = home_pose

        self.release_flag = [0]
        self.grasp_flag = [1]
        self.__wait_flag = [2]
        self.__end_flag = [3]
        self.cur_list_idx = 0
        self.error_bound = 0.001
        self.error_bound_far = 0.003
    
        self.end_flag = False
        self.repeat_flag = False
        self.repeat_place_flag = False



        self.__hiro_g = hiro_grasp.hiro_grasp()
        self.__pose_order_list = self.set_pose_order_list(flag)
        self.__cur_idx = 0 
        self.already_collided = False
        self.x_history_list = []
        self.y_history_list = []
        self.z_history_list = []
        self.max_history_len = 50
        self.ext_force = 0.0
        self.z_force_release_flag = False
        self.curr_transfers = 1

        rospy.Subscriber("/franka_state_controller/F_ext", WrenchStamped, self.fr_ext_cb)

    def set_x_c(self, x_c):
        self.grasp_dict["x_c"] = x_c

    def set_x_g(self, x_g):
        self.grasp_dict["x_g"] = x_g

    def set_x_a(self, x_a):
        self.grasp_dict["x_a"] = x_a

    def get_curr_error(self, x_ee):
        return self.xyz_diff(x_ee, self.grasp_dict["x_goal"])

    def xyz_diff(self, start_pose, end_pose):
        self.p_t = 0.1
        self.grasp_dict["x_goal"] = self.read_octo_file_list()
        goal = self.grasp_dict["x_goal"]
        difference = [0.0, 0.0, 0.0]
        difference[0] = (goal[0] - start_pose[0]) * self.p_t 
        difference[1] = (goal[1] - start_pose[1]) * self.p_t
        difference[2] = (goal[2] - start_pose[2]) * self.p_t
        # print("diff: ", difference)
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

        elif self.flag == "cone":
            return ["x_c", "x_g", "grasp", "x_h", "x_d", "x_h"]

        elif self.flag == "list":
            print(self.grasp_list)
            if len(self.grasp_list) == 0:
                self.grasp_dict["x_goal"] = self.home_pose
            else:
                self.grasp_dict["x_goal"] = self.grasp_list[self.cur_list_idx]

            self.grasp_pose = self.grasp_dict["x_goal"]
            return None
        elif self.flag == "octo":
            self.grasp_dict["x_goal"] = self.grasp_dict["x_g"]
        elif self.flag == "octolist":
            self.grasp_dict["x_goal"] = self.read_octo_file_list()
            return
        elif self.flag == "xbox":
            self.grasp_dict["x_goal"] = self.grasp_dict["x_g"]

    def read_file(self):
        with open(self.load_file, 'r') as file:
            lines = file.readlines()
            float_arrays = [np.array(eval(line)) for line in lines]
            result_array = np.array(float_arrays)
            print("read file", result_array)
            
        self.grasp_list = result_array
        
    def read_octo_file_list(self):
        with open(self.octo_file, 'r') as f:
            line = f.readline().strip()  # e.g., "[ 0.00175201  0.00583863 -0.02097218  0.30789703 -0.0245871  -0.01315031 1.0180444 ]"
        
            # Remove the leading '[' and trailing ']'
            line = line.strip("[]")
            
            # Split by whitespace to get each number as a string
            str_values = line.split()
            if len (str_values) == 0:
                return self.grasp_dict["x_goal"]
            
            # Convert each string value to a float
            float_values = [float(val) for val in line.split(",")]
        #     line = file.readlines()
        #     # line = line.strip("[]")
    
        #     # Split by whitespace to get each number as a string
        #     # str_values = line.split()
        #     float_values = [float(val) for val in line]
            # print(float_values)
            # float_values[3:6] = [val * 0.01 for val in float_values[3:6]]
            # fr_rot = R.from_euler("xyz", float_values[3:6], degrees=False)                   
            # fr_quat = fr_rot.as_quat()
            float_values[3:7] = [0.9994651, -0.00187451, 0.0307489, -0.01097748]
            result_array = np.array(float_values[:7])
            
        return result_array
    
    def read_octo_file(self):
        with open(self.octo_file, 'r') as f:
            line = f.readline()  # e.g., "[ 0.00175201  0.00583863 -0.02097218  0.30789703 -0.0245871  -0.01315031 1.0180444 ]"
            if not line:
                return [0, 0, 0, 0, 0, 0, 0]
            line = line.strip()
            # Remove the leading '[' and trailing ']'
            line = line.strip("[]")
            
            # Split by whitespace to get each number as a string
            str_values = line.split()
            
            # Convert each string value to a float
            float_values = [float(val) for val in line.split(",")]
        #     line = file.readlines()
        #     # line = line.strip("[]")
    
        #     # Split by whitespace to get each number as a string
        #     # str_values = line.split()
        #     float_values = [float(val) for val in line]
            # print(float_values)
        #     # fr_rot = R.from_euler("xyz", float_arrays[3:7], degrees=False)                   
        #     # fr_quat = fr_rot.as_quat()
            result_array = np.array(float_values)
            
        return result_array

    def update_grasp_goal(self):
        self.grasp_dict["x_goal"] = self.grasp_dict[self.__pose_order_list[self.__cur_idx]]

    def set_grasp_width(self, width):
        self.__hiro_g.set_grasp_width(width)

    def grasp(self):
        self.__hiro_g.grasp()

    def get_error_xyz(self, x_ee):
        return self.xyz_diff(x_ee, self.grasp_dict["x_goal"])

    def check_cone_done(self):
        if self.flag == "cone" and self.__cur_idx > 2:
            return True
        return False
    
    def __inc_pose_list(self):
        self.cur_list_idx += 1 
        if self.repeat_flag and self.cur_list_idx == 1:    
            self.cur_list_idx += 1 
        if self.cur_list_idx == len(self.grasp_list):
            self.repeat_flag = True
            self.repeat_place_flag = True
        if len(self.grasp_list) != 0:
            self.cur_list_idx = self.cur_list_idx % (len(self.grasp_list))
        if len(self.grasp_list) != 0:
            self.grasp_dict["x_goal"] = self.grasp_list[self.cur_list_idx]
    
    
    def next_flag_is_grasp_or_release(self, error):
        list_idx = self.cur_list_idx + 1 
        if list_idx >= len(self.grasp_list):
            return False
        ret = self.grasp_list[list_idx]
        if len(ret) == 1 and self.is_in_error_bound(error):
            return True
        else:
            return False

    def is_in_error_bound(self, error):
        error_sum = sum([abs(elem) for elem in error])
        # print("ERROR SUM: ", error_sum)
        if error_sum < self.error_bound:
            return True
        else:
            return False

    def check_next_state(self, error):
        self.z_force_drop_bound = 0.003
        ret = False
        if self.is_in_error_bound(error):
            if not self.__pose_order_list:
                self.__inc_pose_list()
                if self.repeat_flag and self.repeat_place_flag and self.cur_list_idx == 3:
                    self.__hiro_g.open()
                    rospy.sleep(1.0)
                    self.cur_list_idx = 0
                    self.repeat_place_flag = False
                    self.repeat_flag = False
                    if len(self.grasp_list) != 0:
                        self.grasp_dict["x_goal"] = self.grasp_list[self.cur_list_idx]
                if len(self.grasp_list) != 0 and not self.flag == "octolist":
                    if self.grasp_list[self.cur_list_idx][0] == self.grasp_flag and not (self.repeat_flag and self.cur_list_idx == 3):
                        self.curr_transfers += 1
                        self.__hiro_g.set_grasp_width(0.03)
                        self.__hiro_g.grasp()
                        rospy.sleep(1.0)
                        self.__inc_pose_list()
                        ret = True
                    elif self.grasp_list[self.cur_list_idx][0] == self.release_flag or (self.repeat_flag and self.cur_list_idx == 3):
                        self.__hiro_g.open()
                        rospy.sleep(1.0)
                        self.__inc_pose_list()
                        ret = True

        elif self.z_force_release_flag and error[2] < self.z_force_drop_bound:
                if len(self.grasp_list) != 0:
                    next_idx = (self.cur_list_idx + 1 ) % (len(self.grasp_list))
                    if self.grasp_list[next_idx][0] == self.release_flag:
                        self.__hiro_g.open()
                        rospy.sleep(1.0)
                        ret = True
                        self.__inc_pose_list()                    
                        self.__inc_pose_list()  
                        self.grasp_dict["x_goal"] = self.grasp_list[self.cur_list_idx]

        if not self.__pose_order_list:
            return ret
        elif self.__pose_order_list[self.__cur_idx] == "grasp":
            ret = self.inc_state()
            self.__hiro_g.grasp()
            rospy.sleep(1.0)
        return ret
    
    def inc_state(self):
        self.__cur_idx += 1 
        self.__cur_idx = self.__cur_idx % (len(self.__pose_order_list))
        self.grasp_dict["x_goal"] = self.grasp_dict[self.__pose_order_list[self.__cur_idx]]
        if self.__cur_idx == 0:
            self.already_collided = False
            return True
        return False
    
    def fr_ext_cb(self, data):
        self.ext_force = data.wrench.force.z
        if self.ext_force >= 7.0:
            self.z_force_release_flag = True
        else:
            self.z_force_release_flag = False

class XboxInput:
    def __init__(self, flag):
        self.flag = flag
        self.gui_flag = ""
        default_setting_file_path = path_to_src + '/configs/settings.yaml'

        # setting_file_path = rospy.get_param('setting_file_path')
        # if setting_file_path == '':
        #     setting_file_path = default_setting_file_path

        setting_file_path = default_setting_file_path

        self.write_file = "/home/caleb/robochem_steps/transfer_wo_jig.txt"

        self.robot = Robot(setting_file_path)
        self.grasped = False
        self.final_location = False
        self.grasp_loop = None
        self.made_loop = False
        self.octo_pos_stride = 0.02
        self.octo_rot_stride = 0.02

        self.xbox_pos_stride = 0.001
        self.xbox_rot_stride = 0.005

        self.ee_pose_goals_pub = rospy.Publisher('relaxed_ik/ee_pose_goals', EEPoseGoals, queue_size=1)
        self.ee_vel_goals_pub = rospy.Publisher('relaxed_ik/ee_vel_goals', EEVelGoals, queue_size=1)
        self.hiro_ee_vel_goals_pub = rospy.Publisher('relaxed_ik/hiro_ee_vel_goals', Float64MultiArray, queue_size=1)
        self.impedance_change_bool_pub = rospy.Publisher('impedance_change_bool', Bool, queue_size=1)

        # create a ros publisher to the topic "/relaxed_ik/joint_angle_solutions" with type sensor messages joint state
        self.joint_angle_solutions_pub = rospy.Publisher('/relaxed_ik/reset', JointState, queue_size=1)
        
        if self.flag == "xbox":
            self.pos_stride = self.xbox_pos_stride
            self.rot_stride = self.xbox_rot_stride
        if self.flag == "octo" or self.flag == "octolist":
            self.pos_stride = self.octo_pos_stride
            self.rot_stride = self.octo_rot_stride
            
            # 03_25_25 - old steps not really moving
            # self.pos_stride = 0.0001
            # self.rot_stride = 0.0001
            # self.rot_stride = 0.00
 


        self.min_pos_stride = 0.003
        self.min_rot_stride = 0.01875
        self.max_pos_stride = 0.0003
        self.max_rot_stride = 0.001875
        
        # values before octo tests
        # self.p_t = 0.04
        # self.p_r = 0.007
        self.p_t = 0.0004
        self.p_r = 0.007
        if flag == "octolist":
            self.p_t = 0.006
            self.p_r = 0.006
        self.rot_error = [0.0, 0.0, 0.0]

        self.z_offset = 0.0
        self.y_offset = 0.0
        self.x_offset = 0.0
        self.transfer_z_offset = 0.26 #m above the grasp location
        self.transfer_first_quat = [0.0, 0.0, 0.0, 1.0]
        self.transfer_first_z = 0.0
        self.transfer_first_grasp_taken = False

        self.linear = [0,0,0]
        self.angular = [0,0,0]
        self.joy_data = None
        self.grasp_pose = [0,0,0,0,0,0,0]
        self.x_a = [0,0,0,0,0,0,0]
        self.start_grasp = False
        self.prev_fr_euler = [0, 0, 0]
        self.grasp_midpoint = [0,0,0,0,0,0,0]
        self.num_buckets = 0
        self.franka_pose = [0,0,0,0,0,0,0]
        self.series_str = "Series"
        self.parallel_str = "Parallel"
        self.dialysis_type = self.series_str

        self.grip_cur = 0.08
        self.grip_inc = 0.01
        self.grip_max = 0.08
        self.grip_min = 0.01

        self.fr_position = [0.0, 0.0, 0.0]
        self.fr_rotation_matrix = [[0.0, 0.0, 0.0],
                                   [0.0, 0.0, 0.0],
                                   [0.0, 0.0, 0.0]]
        self.fr_state = False
        self.error_state = [0.0, 0.0, 0.0]
        self.cur_list_idx = 0
        self.contact = False

        self.fr_is_neg = False
        self.fr_set = False
        self.got_prev_sign = False
        self.og_set = False
        self.in_collision = False
        self.og_x_a = [1, 1, 1]
        self.cone_radius = 0.25
        self.cone_height = 0.25
        self.msg_obj_to_line = 0.0
        self.msg_cone_start = 0.0
        self.nearest_cone_points = [None, None]
        self.x_c = [0,0,0,0,0,0,0]
        self.last_grasp_time = time.time()
        self.cone_pub = rospy.Publisher('cone_viz', MarkerArray, queue_size=1)
        self.marker_pub = rospy.Publisher('marker_list', MarkerArray, queue_size=1)
        self.text_pub = rospy.Publisher("/text_overlay", OverlayText, queue_size=1)
        self.xyz_quat_goal = []
        self.y_check = 0
        self.r_stick_check = 0
        self.big_x_prev = False

        self.eulers = [0, 0, 0]
        self.marker_list = []
        self.time_until_next = "00:00:00"
        self.robot_mode = "xbox"
        self.franka_joint_states = None
        self.octo_grasp_count = 0
        self.octo_grasp_open_count = 0
        self.octo_grasp_lim = 20
        
        
        # Add subscriber for timer updates
        rospy.Subscriber("/gui_msg", GUIMsg, self.gui_msg_callback)
        rospy.Subscriber("/mid_grasp_point", Float32MultiArray, self.grasp_midpoint_callback)
        rospy.Subscriber("joy", Joy, self.joy_cb)
        rospy.Subscriber("final_grasp", Float32MultiArray, self.subscriber_callback)
        rospy.Subscriber("/franka_state_controller/franka_states", FrankaState, self.fr_state_cb)
        rospy.Subscriber("/estimated_approach_frame", Float32MultiArray, self.l_shaped_callback)
        rospy.Subscriber("/contact", Int32, self.contact_cb)
        # create a subscriber to the joint states topic
        rospy.Subscriber("/joint_states", JointState, self.joint_state_cb)

        rospy.sleep(1.0)
        rospy.Timer(rospy.Duration(0.01), self.timer_callback)

    def joint_state_cb(self, data):
        self.franka_joint_states = data

    def set_to_hand_control(self):
        self.flag = "impedance"
        print(f"IMPEDING: {self.flag}")
        self.made_loop = False
        self.impedance_change_bool_pub.publish(True)

    def joy_cb(self, data):
        if self.flag == "xbox":
            self.pos_stride = self.xbox_pos_stride
            self.rot_stride = self.xbox_rot_stride
        if self.flag == "octo" or self.flag == "octolist":
            # self.pos_stride = 0.03
            # self.rot_stride = 0.01
            self.pos_stride = self.octo_pos_stride
            self.rot_stride = self.octo_rot_stride
        self.joy_data = data
        print("joy cb", flag)

        # Second statement with axes makes it so the left trigger must be depressed to move the robot
        if self.flag == "xbox":
            if abs(self.joy_data.axes[1]) > 0.2:
                self.linear[0] -= self.pos_stride * self.joy_data.axes[1]
            if abs(self.joy_data.axes[0]) > 0.2:
                self.linear[1] -= self.pos_stride * self.joy_data.axes[0]
            if abs(self.joy_data.axes[4]) > 0.2:
                self.linear[2] += self.pos_stride * self.joy_data.axes[4]

            # Caleb - Add back if you want rotation control
            if abs(self.joy_data.axes[6]) > 0.2:
                self.angular[0] += self.rot_stride * self.joy_data.axes[6]
            if abs(self.joy_data.axes[7]) > 0.2:
                self.angular[1] += self.rot_stride * self.joy_data.axes[7]
            if abs(self.joy_data.buttons[4]) > 0.2:
                self.angular[2] += self.rot_stride
            if abs(self.joy_data.buttons[5]) > 0.2:
                self.angular[2] -= self.rot_stride

        if self.flag == "octo" and self.grasp_loop:
            axes_delta = self.grasp_loop.read_octo_file()
            # the first three values are x, y, z, then roll, pitch, and yaw. Disregard the last value also multiply each by a stride
            self.linear[0]  += axes_delta[0] * self.pos_stride
            self.linear[1]  += axes_delta[1] * self.pos_stride
            self.linear[2]  += axes_delta[2] * self.pos_stride
            self.angular[0] += axes_delta[3] * self.rot_stride
            self.angular[1] += axes_delta[4] * self.rot_stride
            self.angular[2] += axes_delta[5] * self.rot_stride
            print(f"LINEAR: {self.linear}")
            print(f"ANGULAR: {self.angular}")
            print("grasp flag", axes_delta[6])
            if axes_delta[6] > 0.5:
                self.octo_grasp_count += 1
                self.octo_grasp_open_count = 0
                if self.octo_grasp_count >= self.octo_grasp_lim and self.grip_cur != 0.1:
                    self.octo_grasp_count = 0
                    self.grip_cur = 0.1
                    # self.move_gripper() 

            else:
                self.octo_grasp_open_count += 1
                self.octo_grasp_count = 0
                if self.octo_grasp_open_count >= self.octo_grasp_lim and self.grip_cur != 0.01:
                    self.octo_grasp_open_count = 0
                    self.grip_cur = 0.01
                    # self.move_gripper() 
            # if (time.time() - self.last_grasp_time) >= 2.0:
            #     self.last_grasp_time = time.time()
            

        a = data.buttons[0]
        b = data.buttons[1]
        print(f"BUTTON A: {a}")
        print(f"BUTTON B: {b}")
        x = data.buttons[2] 
        y = data.buttons[3]
        # rt_trigger = data.axes[5]

        # self.pos_stride = (rt_trigger + 1) / 2 * (self.max_pos_stride - self.min_pos_stride) + self.min_pos_stride
        # self.rot_stride = (rt_trigger + 1) / 2 * (self.max_rot_stride - self.min_rot_stride) + self.min_rot_stride

        back_button = data.buttons[6]
        start_button = data.buttons[7]
        big_x = data.buttons[8]
        r_stick = data.buttons[10]

        if r_stick:
            self.r_stick_check += 1
        if self.r_stick_check == 1 and self.fr_state and self.franka_pose[0]:
            if self.dialysis_type == self.series_str:
                self.dialysis_type = self.parallel_str
            else:
                self.dialysis_type = self.series_str
        self.r_stick_check += 1

        if not r_stick: self.r_stick_check = 0

        if start_button:
            self.flag = "octo"
            self.made_loop = False
            self.og_set = False
            self.grasp_pose = self.franka_pose
            self.impedance_change_bool_pub.publish(False)

        if not big_x == self.big_x_prev:
            if big_x and not self.flag == "xbox":
                print(f"XBOXING: {self.flag}")
                self.flag = "xbox"
                self.made_loop = False
                self.og_set = False
                self.grasp_pose = self.franka_pose
                self.curr_transfers = 1
                self.impedance_change_bool_pub.publish(False)

            elif big_x and not self.flag == "impedance":
                self.set_to_hand_control()
                if self.transfer_first_grasp_taken:
                    self.transfer_first_grasp_taken = False
        self.big_x_prev = big_x

        if back_button:
            with open(self.write_file, 'w') as file:
                file.write("")
            # if I detele the markers, then I also need to clear the grasp list
            self.grasp_loop.grasp_list = []
            self.delete_markers()
            self.repeat_flag = False
            self.repeat_place_flag = False
            self.num_buckets = 0
            self.curr_transfers = 1
            self.transfer_first_grasp_taken = False

        if y:
            self.y_check += 1
        if self.y_check == 1 and self.fr_state and self.franka_pose[0]:
            self.num_buckets += 1
            if not self.transfer_first_grasp_taken:
                print("FIRST GRASP")
                self.transfer_first_grasp_taken = True
                self.transfer_first_z = self.franka_pose[2]
                self.transfer_all_quat = [0.9999914970512331, -0.0023646880938439966, 0.002833256283750675, -0.0018403082033183569]
                self.transfer_first_quat = self.transfer_all_quat
                first_temp = deepcopy(self.franka_pose)
                first_temp[3:] = self.transfer_first_quat
                first_temp[2] = first_temp[2] + self.transfer_z_offset
                self.append_state_to_file(first_temp)
                # self.append_grip_action_to_file(self.grasp_loop.release_flag)
                first_temp[2] = first_temp[2] - self.transfer_z_offset
                self.append_state_to_file(first_temp)
                self.append_grip_action_to_file(self.grasp_loop.grasp_flag)
                first_temp[2] = first_temp[2] + self.transfer_z_offset
                self.append_state_to_file(first_temp)
                print(f"GRASP GOAL:{self.grasp_loop.grasp_dict['x_goal']}")
                self.update_mode_and_grasp_visualization()
            else:
                self.update_mode_and_grasp_visualization()
                print("TRANSFER")
                temp_cur_pose = deepcopy(self.franka_pose)
                temp_cur_pose[2] = temp_cur_pose[2] + self.transfer_z_offset
                temp_cur_pose[3:] = self.transfer_first_quat
                self.append_state_to_file(temp_cur_pose)
                temp_cur_pose[2] = temp_cur_pose[2] - self.transfer_z_offset
                self.append_state_to_file(temp_cur_pose)
                self.append_grip_action_to_file(self.grasp_loop.release_flag)
                temp_cur_pose[2] = temp_cur_pose[2] + self.transfer_z_offset
                self.append_state_to_file(temp_cur_pose)
                temp_cur_pose[2] = temp_cur_pose[2] - self.transfer_z_offset
                self.append_state_to_file(temp_cur_pose)
                self.append_grip_action_to_file(self.grasp_loop.grasp_flag)
                temp_cur_pose[2] = temp_cur_pose[2] + self.transfer_z_offset
                self.append_state_to_file(temp_cur_pose)

            self.grasp_loop.read_file()

        self.y_check += 1

        if not y: self.y_check = 0

        if a:
           self.grip_cur = 0.1
        elif b:
           self.grip_cur = 0.01

        if a or b:
            if (time.time() - self.last_grasp_time) >= 2.0:
                self.move_gripper() 
                self.last_grasp_time = time.time()

    def append_grip_action_to_file(self, action):
        with open(self.write_file, 'a') as file:
            file.write('[' + str(action[0]) + ']' + '\n')
            file.close()

    def append_state_to_file(self, state):
        with open(self.write_file, 'a') as file:
                file.write('[')
                for x, temp in enumerate(start=1, iterable=state):
                    if x != 7:
                        file.write(str(temp) + ', ')
                    else:
                        file.write(str(temp))
                file.write(']\n')
                file.close()

    def move_gripper(self):
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

    def pub_goal_array(self, goal_list):

            # marker_pub = rospy.Publisher('/list_goal_list', Marker, queue_size=1)
            # Make for loop that appends each of the xyz positions in goal to the marker array
            # data type for publishing using the self.marker_pub variable.
            # make it so the start color is red and the end color is green incrementally based on the
            # number of markers in the list
            marker_arr = MarkerArray()
            colors_ = cm.rainbow(np.linspace(0, 1, len(goal_list)))
            for idx, goal in enumerate(goal_list):
                m = Marker()
                m.ns = ""
                m.header.frame_id = "fr3_link0"
                m.id = idx
                m.header.stamp = rospy.Time.now()
                m.type = Marker.SPHERE
                m.action = Marker.ADD
                m.pose.position.x = goal[0]
                m.pose.position.y = goal[1]
                m.pose.position.z = goal[2]
                m.scale = Vector3(0.05, 0.05, 0.05)
                color = colors_[idx]
                m.color = ColorRGBA(color[0], color[1], color[2], 0.5)
                marker_arr.markers.append(m)
            self.marker_pub.publish(marker_arr)



    def pub_cylinder(self, goal_list):
        marker_pub = rospy.Publisher('/cylinder_grasp', Marker, queue_size=1)
        marker = Marker()
        marker.header.frame_id = "fr3_link0"
        marker.ns = ""
        marker.header.stamp = rospy.Time.now()
        marker.id = 0
        marker.type = marker.CYLINDER
        marker.action = marker.ADD
        marker.color.a = 0.5
        marker.color.r = 0.0
        marker.color.g = 1.0
        marker.color.b = 1.0
        marker.scale.x = self.cone_radius
        marker.scale.y = self.cone_radius
        marker.scale.z = self.cone_height
        marker.pose.position.x = self.grasp_midpoint[0]
        marker.pose.position.y = self.grasp_midpoint[1]
        marker.pose.position.z = self.grasp_midpoint[2]
        marker.pose.orientation.x = quat[0]
        marker.pose.orientation.y = quat[1]
        marker.pose.orientation.z = quat[2]
        marker.pose.orientation.w = quat[3]
        marker_pub.publish(marker)

    def pub_closest_point(self, xyz):
        marker_pub = rospy.Publisher('/closest_point', Marker, queue_size=1)
        marker = Marker()
        marker.header.frame_id = "fr3_link0"
        marker.ns = ""
        marker.header.stamp = rospy.Time.now()
        marker.id = 1000
        marker.type = marker.SPHERE
        marker.action = marker.ADD
        marker.color.a = 1.0
        marker.color.r = 0.0
        marker.color.g = 1.0
        marker.color.b = 1.0
        marker.scale.x = 0.05
        marker.scale.y = 0.05
        marker.scale.z = 0.05
        marker.pose.position.x = xyz[0]
        marker.pose.position.y = xyz[1]
        marker.pose.position.z = xyz[2]
        marker.pose.orientation.x = 0.0
        marker.pose.orientation.y = 0.0
        marker.pose.orientation.z = 0.0
        marker.pose.orientation.w = 1.0
        marker_pub.publish(marker)

    def pub_cone_as_cylinders(self, x_a, x_g, quat):
        top_xyz = x_a[:3]
        bottom_xyz = x_g[:3]
        marker_arr = MarkerArray()
        cone_split = 100
        lower_cone_size = 0.01
        colors_ = cm.rainbow(np.linspace(0, 1, cone_split))
        x_inter = np.linspace(bottom_xyz[0], top_xyz[0], cone_split)
        y_inter = np.linspace(bottom_xyz[1], top_xyz[1], cone_split)
        z_inter = np.linspace(bottom_xyz[2], top_xyz[2], cone_split)
        z_step = abs(z_inter[1] - z_inter[2])
        width = np.linspace(lower_cone_size, self.cone_radius*2, cone_split )

        for idx, x in enumerate(width):
            m = Marker()
            m.ns = ""
            m.header.frame_id = "fr3_link0"
            m.id = idx
            m.header.stamp = rospy.Time.now()
            m.type = Marker.CYLINDER
            m.action = Marker.ADD
            m.pose.orientation.x = quat[0]
            m.pose.orientation.y = quat[1]
            m.pose.orientation.z = quat[2]
            m.pose.orientation.w = quat[3]
            m.scale = Vector3(x, x, z_step)
            color = colors_[idx]
            m.color = ColorRGBA(color[0], color[1], color[2], 0.5)
            m.pose.position.x = x_inter[idx]
            m.pose.position.y = y_inter[idx]
            m.pose.position.z = z_inter[idx]   
            marker_arr.markers.append(m)  
        self.cone_pub.publish(marker_arr)

    def make_fcl_cone(self):
        cone = fcl.Cone(self.cone_radius, self.cone_height)
        return cone

    def make_fcl_cylinder(self):
        cylinder = fcl.Cylinder(self.cone_radius, self.cone_height)
        return cylinder

    def make_ee_sphere(self):
        rad = 0.01
        ee = fcl.Sphere(rad)
        return ee


        s = np.dot(a - p, d)
        t = np.dot(p - b, d)

        h = np.maximum.reduce([s, t, 0])

        c = np.cross(p - a, d)

        return abs(np.hypot(h, np.linalg.norm(c)))

    def get_norm(self, a, b):
        a_np = np.array(a)
        b_np = np.array(b)

        return np.linalg.norm(b_np - a_np)

    def get_unit_line(self, a, b):
        a_np = np.asarray(a)
        b_np = np.asarray(b)

        line = b_np - a_np
        line = normalize([line], axis=1, norm='l1')
        return line[0]


    def find_ee_height(self, gripper_loc, x_a, grasp_loc):
        theta = np.rad2deg(np.arctan(self.cone_radius / self.cone_height))
        radius = self.lineseg_dist(gripper_loc, grasp_loc, x_a)

        hypot = self.get_norm(gripper_loc, grasp_loc)
        temp_theta = np.arcsin(radius/hypot)
        temp_theta_deg = np.rad2deg(temp_theta)
        new_height = np.cos(temp_theta) * hypot
        cone_radius = np.tan(theta) * new_height

        self.msg_obj_to_line = radius
        self.msg_cone_start = 1.0

        return new_height


    def check_collide(self, ee, ee_trans, ee_quat, cone, cone_trans, cone_quat):
        tf_ee = fcl.Transform(ee_quat, ee_trans)
        tf_cone = fcl.Transform(cone_quat, cone_trans)
        obj_ee = fcl.CollisionObject(ee, tf_ee)
        obj_cone = fcl.CollisionObject(cone, tf_cone)
        request = fcl.DistanceRequest(enable_nearest_points=True, enable_signed_distance=True)
        result = fcl.DistanceResult()
        ret = fcl.distance(obj_ee, obj_cone, request, result)
        self.nearest_cone_points = result.nearest_points
        if ret > 0 and not self.grasp_loop.already_collided:
            return False
        else:
            self.grasp_loop.already_collided = True
            return True
        
    def clamp_rotation(self):
        # print(self.eulers, "EULERS")
        # print(self.angular, "ANGULAR")

        if abs(self.eulers[0]) < 1.4: 
            if self.eulers[0] < 0 and self.angular[0] < 0: self.angular[0] = 0
            elif self.eulers[0] > 0 and self.angular[0] > 0: self.angular[0] = 0
        
        if self.eulers[1] < -0.3: 
            if self.eulers[1] < 0 and self.angular[1] < 0: self.angular[1] = 0
        if self.eulers[1] > 0.3: 
            if self.eulers[1] > 0 and self.angular[1] > 0: self.angular[1] = 0
        # same thing as above for 2
        if abs(self.eulers[2]) > 1.57:
            if self.eulers[2] < 0 and self.angular[2] < 0: self.angular[2] = 0
            elif self.eulers[2] > 0 and self.angular[2] > 0: self.angular[2] = 0


        # if ro[0] > 0.5: self.angular[0] = 0
        # elif self.eulers[0] < -0.5: self.angular[0] = 0

        # if self.eulers[1] > 0.5: self.angular[1] = 0
        # elif self.eulers[1] < -0.5: self.angular[1] = 0

        # if self.eulers[2] > 0.5: self.angular[2] = 0
        # elif self.eulers[2] < -0.5: self.angular[2] = 0

    def clamp_linear_position(self):
        z_max = 0.8
        z_min = 0.02
        y_max = 0.9
        y_min = -0.9
        x_max = 1.0
        x_min = 0.0
        # print(self.eulers)

        # Set X bounds
        if self.fr_position[0] > x_max and self.linear[0] > 0: self.linear[0] = 0
        elif self.fr_position[0] < x_min and self.linear[0] < 0: self.linear[0] = 0

        # Set Y bounds
        if self.fr_position[1] > y_max and self.linear[1] > 0: self.linear[1] = 0
        elif self.fr_position[1] < y_min and self.linear[1] < 0: self.linear[1] = 0

        # Set Z bounds
        if self.fr_position[2] > z_max and self.linear[2] > 0: self.linear[2] = 0
        elif self.fr_position[2] < z_min and self.linear[2] < 0: self.linear[2] = 0

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
    def move_octo_list(self):
        if self.fr_state:
            if not self.made_loop:
                self.made_loop = True
                self.grasp_loop = GraspLoop(self.flag, self.gui_flag, self.grasp_pose, self.og_x_a)

            self.update_mode_and_grasp_visualization()


            line = self.get_unit_line(self.grasp_pose[:3], self.og_x_a[:3])
            hiro_msg = Float64MultiArray()
            self.error_state = self.grasp_loop.get_curr_error(self.fr_position)
            hiro_msg.data = self.get_hiro_error_msg(self.grasp_loop.grasp_dict["x_goal"][3:])
            # print("goal:", self.grasp_loop.grasp_dict["x_goal"][:3])
            # print("fr:", self.fr_position)
            for x in line:
                hiro_msg.data.append(x)
            hiro_msg.data.append(self.cone_radius)
            hiro_msg.data.append(self.cone_height)
            hiro_msg.data.append(self.msg_obj_to_line)
            hiro_msg.data.append(self.msg_cone_start)
            hiro_msg.data.append(self.og_x_a[0])
            hiro_msg.data.append(self.og_x_a[1])
            hiro_msg.data.append(self.og_x_a[2])
            hiro_msg.data.append(self.grasp_pose[0])
            hiro_msg.data.append(self.grasp_pose[1])
            hiro_msg.data.append(self.grasp_pose[2])

            self.franka_pose = []
            self.franka_pose.extend(self.fr_position)
            self.franka_pose.extend(self.grasp_loop.grasp_dict["x_goal"][3:])

            self.hiro_ee_vel_goals_pub.publish(hiro_msg)
            if self.grasp_loop.next_flag_is_grasp_or_release(self.error_state):
                self.on_release()
                hiro_msg.data[:3] = [0,0,0]

                self.hiro_ee_vel_goals_pub.publish(hiro_msg)
                rospy.sleep(2)
            self.grasp_loop.check_next_state(self.error_state)

    def move_octo(self):
        # print("MOVING OCTO")
        msg = EEVelGoals()
        if not self.og_set:
            self.og_set = True
            self.og_trans = copy.deepcopy(self.x_a[:3])
            self.og_quat = copy.deepcopy(self.x_a[3:])

        if not self.made_loop:
            self.made_loop = True
            self.grasp_loop = GraspLoop(self.flag, self.gui_flag, self.grasp_pose, self.og_x_a)

        self.update_mode_and_grasp_visualization()
        fr_r = R.from_matrix(self.fr_rotation_matrix)
        self.fr_quat = fr_r.as_quat()
        fr_e = fr_r.as_euler("xyz", degrees=False)
        fr_r = R.from_euler("xyz", fr_e, degrees=False)
        quat_2 = fr_r.as_quat()

        fcl_ee = self.make_ee_sphere()
        fcl_cone = self.make_fcl_cone()

        self.pub_cone_as_cylinders(self.og_x_a, self.grasp_pose, self.og_quat)
        self.in_collision = self.check_collide(fcl_ee, self.fr_position, [self.fr_quat[3], self.fr_quat[0], self.fr_quat[1], self.fr_quat[2]]
                        ,fcl_cone, self.grasp_midpoint[:3], [self.og_quat[3], self.og_quat[0], self.og_quat[1], self.og_quat[2]])
        
        self.franka_pose[:3] = self.fr_position
        self.franka_pose[3:] = self.fr_quat

        self.clamp_linear_position()
        self.clamp_rotation()
        self.pub_closest_point(self.nearest_cone_points[1])
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
        self.ee_vel_goals_pub.publish(msg)
        self.on_release()


    def move_through_list(self):
        if self.fr_state:
            if not self.made_loop:
                self.made_loop = True
                self.grasp_loop = GraspLoop(self.flag, self.gui_flag, self.grasp_pose, self.og_x_a)

            self.update_mode_and_grasp_visualization()


            line = self.get_unit_line(self.grasp_pose[:3], self.og_x_a[:3])
            hiro_msg = Float64MultiArray()
            self.error_state = self.grasp_loop.get_curr_error(self.fr_position)
            hiro_msg.data = self.get_hiro_error_msg(self.grasp_loop.grasp_dict["x_goal"][3:])

            for x in line:
                hiro_msg.data.append(x)
            hiro_msg.data.append(self.cone_radius)
            hiro_msg.data.append(self.cone_height)
            hiro_msg.data.append(self.msg_obj_to_line)
            hiro_msg.data.append(self.msg_cone_start)
            hiro_msg.data.append(self.og_x_a[0])
            hiro_msg.data.append(self.og_x_a[1])
            hiro_msg.data.append(self.og_x_a[2])
            hiro_msg.data.append(self.grasp_pose[0])
            hiro_msg.data.append(self.grasp_pose[1])
            hiro_msg.data.append(self.grasp_pose[2])

            self.franka_pose = []
            self.franka_pose.extend(self.fr_position)
            self.franka_pose.extend(self.grasp_loop.grasp_dict["x_goal"][3:])

            self.hiro_ee_vel_goals_pub.publish(hiro_msg)
            if self.grasp_loop.next_flag_is_grasp_or_release(self.error_state):
                self.on_release()
                hiro_msg.data[:3] = [0,0,0]

                self.hiro_ee_vel_goals_pub.publish(hiro_msg)
                rospy.sleep(2)
            self.grasp_loop.check_next_state(self.error_state)


    def linear_movement(self):
        if self.start_grasp and self.fr_state:
            if not self.made_loop:
                self.made_loop = True
                self.grasp_loop = GraspLoop(self.flag, self.gui_flag, self.grasp_pose, self.og_x_a)

            line = self.get_unit_line(self.grasp_pose[:3], self.og_x_a[:3])
            hiro_msg = Float64MultiArray()
            self.error_state = self.grasp_loop.get_curr_error(self.fr_position)

            hiro_msg.data = self.get_hiro_error_msg(self.grasp_loop.grasp_dict["x_goal"][3:])
            for x in line:
                hiro_msg.data.append(x)
            hiro_msg.data.append(self.cone_radius)
            hiro_msg.data.append(self.cone_height)
            hiro_msg.data.append(self.msg_obj_to_line)
            hiro_msg.data.append(self.msg_cone_start)
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
            hiro_msg.data.append(self.cone_radius)
            hiro_msg.data.append(self.cone_height)
            hiro_msg.data.append(self.msg_obj_to_line)
            hiro_msg.data.append(self.msg_cone_start)
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

    def cone(self):
        if self.start_grasp and self.fr_state: 
            if self.grasped:
                self.x_a = self.grasp_pose

            if not self.made_loop:
                self.made_loop = True
                self.grasp_loop = GraspLoop(self.flag, self.grasp_pose, self.og_x_a)

            grasp_r = R.from_quat(self.x_a[3:])
            grasp_euler = grasp_r.as_euler('xyz', degrees=False)

            fr_r = R.from_matrix(self.fr_rotation_matrix)
            fr_quat = fr_r.as_quat()
            fr_euler = fr_r.as_euler('xyz', degrees=False)
            fr_euler, fr_quat = self.calc_rotation_sign(fr_euler, grasp_euler)

            quat_error, fr_euler = self.quaterion_error(fr_quat=fr_quat, grasp_quat=grasp_r.as_quat(), fr_euler=fr_euler, grasp_euler=grasp_euler)

            self.rot_error = self._xyz_diff(fr_euler, quat_error)
            self.rot_error = np.array(self.rot_error) * self.p_r
            fcl_ee = self.make_ee_sphere()
            fcl_cone = self.make_fcl_cone()

            if not self.og_set:
                self.og_set = True
                self.og_trans = self.grasp_midpoint[:3]
                self.og_quat = self.grasp_midpoint[3:]

            if self.start_grasp:
                self.pub_cone_as_cylinders(self.og_x_a, self.grasp_pose, self.grasp_pose[3:])
            self.in_collision = self.check_collide(fcl_ee, self.fr_position, [fr_quat[3], fr_quat[0], fr_quat[1], fr_quat[2]]
                            ,fcl_cone, self.grasp_midpoint[:3], [self.og_quat[3], self.og_quat[0], self.og_quat[1], self.og_quat[2]])

            if self.in_collision:
                self.find_ee_height(self.fr_position, self.og_x_a[:3], self.grasp_pose[:3])
            else:
                self.x_c = self.nearest_cone_points[1]
                self.x_c = np.concatenate((self.x_c, self.og_x_a[3:]))
                self.grasp_loop.set_x_c(self.x_c)
                self.grasp_loop.update_grasp_goal()

            self.error_state = self.grasp_loop.get_curr_error(self.fr_position)
            self.pub_closest_point(self.x_c)

            line = self.get_unit_line(self.grasp_pose[:3], self.og_x_a[:3])

            hiro_msg = Float64MultiArray()
            hiro_msg.data = self.get_hiro_error_msg(self.grasp_loop.grasp_dict["x_goal"][3:])
            self.grasp_loop.add_to_xyz_history(self.fr_position)
            x_hist, y_hist, z_hist = self.grasp_loop.get_franka_xyz_history()
            for x in line:
                hiro_msg.data.append(x)
            hiro_msg.data.append(self.cone_radius)
            hiro_msg.data.append(self.cone_height)
            hiro_msg.data.append(self.msg_obj_to_line)
            if self.grasp_loop.check_cone_done():
                hiro_msg.data.append(0.0)
            else:
                hiro_msg.data.append(self.msg_cone_start)
            hiro_msg.data.append(self.og_x_a[0])
            hiro_msg.data.append(self.og_x_a[1])
            hiro_msg.data.append(self.og_x_a[2])
            hiro_msg.data.append(self.grasp_midpoint[0])
            hiro_msg.data.append(self.grasp_midpoint[1])
            hiro_msg.data.append(self.grasp_midpoint[2])
            for x in x_hist:
                hiro_msg.data.append(x)
            for y in y_hist:
                hiro_msg.data.append(y)
            for z in z_hist:
                hiro_msg.data.append(z)


            self.hiro_ee_vel_goals_pub.publish(hiro_msg)
            self.on_release()
            if self.grasp_loop.check_next_state(self.error_state):
                self.wait_for_new_grasp()

    def xbox_input(self):
        msg = EEVelGoals()
        if not self.og_set:
            self.og_set = True
            self.og_trans = copy.deepcopy(self.x_a[:3])
            self.og_quat = copy.deepcopy(self.x_a[3:])

        if not self.made_loop:
            self.made_loop = True
            self.grasp_loop = GraspLoop(self.flag, self.gui_flag, self.grasp_pose, self.og_x_a)

        self.update_mode_and_grasp_visualization()
        fr_r = R.from_matrix(self.fr_rotation_matrix)
        self.fr_quat = fr_r.as_quat()
        fr_e = fr_r.as_euler("xyz", degrees=False)
        fr_r = R.from_euler("xyz", fr_e, degrees=False)
        quat_2 = fr_r.as_quat()

        fcl_ee = self.make_ee_sphere()
        fcl_cone = self.make_fcl_cone()

        self.pub_cone_as_cylinders(self.og_x_a, self.grasp_pose, self.og_quat)
        self.in_collision = self.check_collide(fcl_ee, self.fr_position, [self.fr_quat[3], self.fr_quat[0], self.fr_quat[1], self.fr_quat[2]]
                        ,fcl_cone, self.grasp_midpoint[:3], [self.og_quat[3], self.og_quat[0], self.og_quat[1], self.og_quat[2]])
        
        self.franka_pose[:3] = self.fr_position
        self.franka_pose[3:] = self.fr_quat

        self.clamp_linear_position()
        self.clamp_rotation()
        self.pub_closest_point(self.nearest_cone_points[1])
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
        self.ee_vel_goals_pub.publish(msg)
        self.on_release()
    # create a new function that deletes all of the current markers from marker array so 
    # they are not visualized in rviz

    def delete_markers(self):
        #deletes all of the current markers from marker array so they are not visualized in rviz
        marker_arr = MarkerArray()
        for idx in range(len(self.marker_list)):
            m = Marker()
            m.ns = ""
            m.header.frame_id = "fr3_link0"
            m.id = idx
            m.header.stamp = rospy.Time.now()
            m.action = Marker.DELETE
            marker_arr.markers.append(m)
            # if idx % 2 == 1:
            m2 = Marker()
            m2.id = idx + 1000
            m2.ns = ""
            m2.header.frame_id = "fr3_link0"
            m2.header.stamp = rospy.Time.now()
            m2.action = Marker.DELETE
            marker_arr.markers.append(m2)
        self.marker_pub.publish(marker_arr)

    def update_grasp_list_visualization(self):

        marker_arr = MarkerArray()
        # prune the grasp list so we only inluce elements that have 7 elements
        self.marker_list = [grasp for grasp in self.grasp_loop.grasp_list if len(grasp) == 7]
        self.temp_list = [grasp for idx, grasp in enumerate(self.marker_list) if idx % 5 == 4]
        if len(self.marker_list) > 1:
            self.marker_list = [self.marker_list[1]] + self.temp_list
        
        # self.marker_list = [grasp for idx, grasp in enumerate(self.marker_list) if idx % 2 != 1]

        colors_ = cm.rainbow(np.linspace(0, 1, len(self.marker_list)))
        col_list = cm.rainbow(np.linspace(0, 1, 20))
        for idx, goal in enumerate(self.marker_list):
            m = Marker()
            m.ns = ""
            m.header.frame_id = "fr3_link0"
            m.id = idx
            m.header.stamp = rospy.Time.now()
            m.type = Marker.SPHERE
            m.action = Marker.ADD
            m.pose.position.x = goal[0]
            m.pose.position.y = goal[1]
            m.pose.position.z = goal[2]
            m.scale = Vector3(0.05, 0.05, 0.05)
            color = col_list[idx]
            m.color = ColorRGBA(color[0], color[1], color[2], 1.0)
            # chage the color map so we only have a gradient for the red color channel
            marker_arr.markers.append(m)
            # if idx % 2 == 1:
                # make a new marker to append to the marker arrray that draws the text number of the grasp
                # above the marker
            m2 = Marker()
            m2.ns = ""
            m2.header.frame_id = "fr3_link0"
            m2.id = idx + 1000
            m2.header.stamp = rospy.Time.now()
            m2.type = Marker.TEXT_VIEW_FACING
            m2.action = Marker.ADD
            m2.pose.position.x = goal[0]
            m2.pose.position.y = goal[1]
            m2.pose.position.z = goal[2] + 0.05
            m2.scale = Vector3(0.05, 0.05, 0.05)
            m2.text = str(idx+1)
            m2.color = ColorRGBA(1.0, 1.0, 1.0, 1.0)
            marker_arr.markers.append(m2)

        self.marker_pub.publish(marker_arr)
    

    def add_text_to_rvis_current_control_state(self):
        """Update the RViz text overlay with current state and timer."""
        text = OverlayText()
        robot_dict = {
            "xbox": "Xbox Control", 
            "list": "Execute saved trajectories", 
            "impedance": "Hand Control",
            "octo": "Octo Control",
            "octolist": "OctoList Control",
        }

        if self.num_buckets == 0:
            text.text = (
                f"Transfers: {self.num_buckets}\n"
                f"Current Transfer:\n"
                f"Robot State: {robot_dict[self.flag]}\n"
            )
        else:
            # Add timer display if in list mode
            if self.flag == "list":
                text.text = (
                    f"Transfers: {self.num_buckets}\n"
                    f"Current Transfer: {self.grasp_loop.curr_transfers}\n"
                    f"Time until next transfer: {self.time_until_next}\n"
                    f"Robot State: {robot_dict[self.flag]}\n"
                )
            else:
                text.text = (
                    f"Transfers: {self.num_buckets}\n"
                    f"Current Transfer: {self.grasp_loop.curr_transfers}\n"
                    f"Robot State: {robot_dict[self.flag]}\n"
                )

        # Set text properties
        text.width = 400
        text.height = 100
        text.left = 10
        text.top = 10
        text.text_size = 12
        text.line_width = 2
        text.bg_color.r = 0.0
        text.bg_color.g = 0.0
        text.bg_color.b = 0.0
        text.bg_color.a = 0.8
        text.fg_color.r = 1.0
        text.fg_color.g = 1.0
        text.fg_color.b = 1.0
        text.fg_color.a = 1.0

        self.text_pub.publish(text)

    def update_mode_and_grasp_visualization(self):
        self.update_grasp_list_visualization()
        self.add_text_to_rvis_current_control_state()

    def gui_msg_callback(self, msg):
        """Handle incoming GUI messages with timer updates."""
        # if hasattr(msg, 'transfer_times'):
        self.time_until_next = msg.transfer_times
        # if hasattr(msg, 'robot_mode'):
        self.robot_mode = msg.robot_mode


    def timer_callback(self, event):
        fr_r = R.from_matrix(self.fr_rotation_matrix)
        self.fr_quat = fr_r.as_quat()
        self.eulers = fr_r.as_euler("xyz", degrees=False)

        self.franka_pose[:3] = self.fr_position
        self.franka_pose[3:] = self.fr_quat

        if self.grasp_loop != None:
            self.update_mode_and_grasp_visualization()

        if self.flag == "linear":
            self.linear_movement()
        elif self.flag == "xbox":
            self.xbox_input()
        elif self.flag == "l-shaped":
            self.l_shaped_movement()
        elif self.flag == "cone":
            self.cone()
        elif self.flag == "list":
            self.move_through_list()
        elif self.flag == "octo":
            self.move_octo()
        elif self.flag == "octolist":
            self.move_octo_list()
        elif self.flag == "impedance":
            # publish the current robot state joint angles through joint_angle_solutions_pub
            # print("update franka joint states")
            self.joint_angle_solutions_pub.publish(self.franka_joint_states)
            # self.on_release()
            self.update_mode_and_grasp_visualization()
            self.grasp_pose = self.franka_pose

    def subscriber_callback(self, data):
        zero = [0.0, 0.0, 0.0]
        if data.data != zero and not self.grasped:
            self.start_grasp = True
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
        temp_rot_mat = np.array([
            [data.O_T_EE[0], data.O_T_EE[1], data.O_T_EE[2]],
            [data.O_T_EE[4], data.O_T_EE[5], data.O_T_EE[6]],
            [data.O_T_EE[8], data.O_T_EE[9], data.O_T_EE[10]]
        ])
        self.fr_rotation_matrix = temp_rot_mat
        self.fr_state = True

    def contact_cb(self, msg):
        if msg.data > 75:
            self.linear[1] -= self.pos_stride * 0.75

if __name__ == '__main__':
    # load_file = "/home/caleb/robochem_steps/v2_temp_grasps.txt"
    load_file = "/home/caleb/robochem_steps/transfer_wo_jig.txt"
    # with open(load_file, 'w') as file:
    #     file.write("")
    flag = rospy.get_param("/xbox_input/flag")
    rospy.init_node('xbox_input')
    xController = XboxInput(flag=flag) 
    rospy.spin()
