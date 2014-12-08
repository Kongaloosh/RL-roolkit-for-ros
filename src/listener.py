#!/usr/bin/env python
import rosdep2 
import rospy
import message_filters
from experiment_wrapper import *
from experiment_state_information import *
from learning_toolkit import *
from bento_controller.msg import JointSelection
from bento_controller.msg import BentoState
from bento_controller.srv import *
from dynamixel_msgs.msg import JointState
from learning.msg import Learner_Information
from std_msgs.msg import *

class LearnerNode():
    
    def __init__(self):
        """By generating all the things we enable people to have access to everything with
            minimal exposure to the actual act of pulling the information from the bento
            -- DO THIS: catkin_make -DCATKIN_BLACKLIST_PACKAGES="bento_java;adc_node"

            ---watch for latency; if we have an issue, remove it.
        """
        self.experiment = experiment_adaptive_joint_switching() # change this per experiment
#         self.experiment = example_experiment_predict_shoulder_movement_true_online_vs_td()
#         self.experiment2 = Foo()
        self.wrist_rotation_state = not_gripper()
        self.wrist_flexion_state = not_gripper()
        self.shoulder_rotation_state = not_gripper()
        self.elbow_flexion_state = not_gripper()
        self.gripper_state = gripper()
        self.joint_activity_state = joint_activity()
        self.currentMessages = list()
        
    def _selected_motor_callback(self, msg):
        self.joint_activity_state.update(msg.joint_group, msg.joint_idx, msg.joint_id)
    
    def _motor_state_callback(self, msg):
        """This is bad code and I feel bad; I am sorry."""
        if msg.name == 'wrist_rotation':
            self.wrist_rotation_state.update(msg.current_pos, msg.load,\
                                             msg.velocity, msg.is_moving)
        if msg.name == 'wrist_flexion':
            self.wrist_flexion_state.update(msg.current_pos, msg.load,\
                                             msg.velocity, msg.is_moving)
        if msg.name == 'shoulder_rotation':
            self.shoulder_rotation_state.update(msg.current_pos, msg.load,\
                                             msg.velocity, msg.is_moving)
        if msg.name == 'gripper':
            self.gripper_state.update(msg.current_pos, msg.load,\
                                             msg.velocity, msg.is_moving)
        if msg.name == 'elbow_flexion':
            self.elbow_flexion_state.update(msg.current_pos, msg.load,\
                                             msg.velocity, msg.is_moving)      
#     def _bento_state_callback(self, msg):
#         self.experiment.update_perception(self.gripper_state, self.wrist_flexion_state,\
#                        self.wrist_rotation_state, self.shoulder_rotation_state,\
#                        self.elbow_flexion_state, self.joint_activity_state)
        
    def listener(self):
        rospy.init_node('listener', anonymous=True)
        rospy.Subscriber('/bento/selected_joint', JointSelection, self._selected_motor_callback)
#         rospy.Subscriber('/bento/state', BentoState, self._bento_state_callback)
        rospy.Subscriber('/bento_controller/dynamixel/wrist_rotation/state', JointState, self._motor_state_callback)
        rospy.Subscriber('/bento_controller/dynamixel/wrist_flexion/state', JointState, self._motor_state_callback)
        rospy.Subscriber('/bento_controller/dynamixel/shoulder_rotation/state', JointState, self._motor_state_callback)
        rospy.Subscriber('/bento_controller/dynamixel/gripper/state', JointState, self._motor_state_callback)
        rospy.Subscriber('/bento_controller/dynamixel/elbow_flexion/state', JointState, self._motor_state_callback)

        
        r = rospy.Rate(20) # 20 Hz
        while not rospy.is_shutdown(): 
            print '=================================='
            print 'Time: ' + str(rospy.get_rostime())
            self.experiment.update_perception(self.gripper_state, self.wrist_flexion_state,\
                       self.wrist_rotation_state, self.shoulder_rotation_state,\
                       self.elbow_flexion_state, self.joint_activity_state)
            r.sleep()
            
#     def talker():
#         pub = rospy.Publisher('chatter',String, queue_size=10)
#         rospy.init_node('talker',anonymous = True)
#         r = rospy.Rate(10)
#         while not rospy.is_shutdown():
#             str = "hi"
#             rospy.loginfo(str)
#             pub.publish(str)
#             r.sleep()
            
if __name__ == '__main__':
    ros = LearnerNode()
    try:
        ros.listener()
    except rospy.ROSInterruptException: pass