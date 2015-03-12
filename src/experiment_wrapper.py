#!/usr/bin/env python
from experiment_state_information import *
from learning_toolkit import *
import rospy
import rosdep2
import time
from std_msgs.msg import *
from bento_controller.srv import *
from bento_controller.msg import *
from PyQt4.QtGui import *
from PyQt4.QtCore import *
import sys
import argparse
from PyQt4 import QtCore
from learning.msg import Float32Timed

from ros_utils.timed import timed

""" Can make more than one experiment class in experiment_wrapper.py for use in listener """
        
class experiment_adaptive_joint_switching(object):
    """
    The job of the experiment handler is to take the information that it has available---the information
    from the bento arm---make sure it is well formatted, and feed it to the learner. The experiment handler
    is the learner's baby-sitter and makes sure that the learners are being given the information needed to
    perform their job.
    """
    def __init__(self):
        self.gripper_states = gripper()
        self.wrist_flexion_states = not_gripper()
        self.wrist_rotation_states = not_gripper()
        self.shoulder_rotation_states = not_gripper()
        self.elbow_flexion_states = not_gripper() 
        self.joint_activity_states = joint_activity()
        self.joint_activity_states.active_joint = 0
        self.joint_activity_states.switch = 0
        self.number_of_steps = 0
        self.start_time = None
        self.switch_count = 0
        self.joint_list = [JointGroupJoint(joint_id=1, min_speed=0, max_speed=1),\
                            JointGroupJoint(joint_id=2, min_speed=0, max_speed=1),\
                             JointGroupJoint(joint_id=3, min_speed=0, max_speed=1),\
                              JointGroupJoint(joint_id=4, min_speed=0, max_speed=1),\
                               JointGroupJoint(joint_id=5, min_speed=0, max_speed=1)]
        self.last_list = [JointGroupJoint(joint_id=1, min_speed=0, max_speed=1),\
                            JointGroupJoint(joint_id=2, min_speed=0, max_speed=1),\
                             JointGroupJoint(joint_id=3, min_speed=0, max_speed=1),\
                              JointGroupJoint(joint_id=4, min_speed=0, max_speed=1),\
                               JointGroupJoint(joint_id=5, min_speed=0, max_speed=1)]
        self.max_gripper_load = -1000
        self.min_gripper_load = 1000
        
        self.learner = SwitchingLearner_bento() # not really used
        self.joints = joint_activity()
        
        """ Instantiate a separate TD Lambda Learner for each joint:
            TDLambdaLearner(numTilings, num_bins (not used), alpha, lambda, gamma, cTableSize (not used)) """
        self.td0 = TDLambdaLearner(4,64,0.01,0.99,0.9,64) # TDLambda Hand
        self.td1 = TDLambdaLearner(4,64,0.01,0.99,0.9,64) # TDLambda Wrist Rotation
        self.td2 = TDLambdaLearner(4,64,0.01,0.99,0.9,64) # TDLambda Wrist Flexion
        self.td3 = TDLambdaLearner(4,64,0.01,0.99,0.9,64) # TDLambda Elbow
        self.td4 = TDLambdaLearner(4,64,0.01,0.99,0.9,64) # TDLambda Shoulder
        self.td_switch = TDLambdaLearner(4,64,0.1,0.99,0.9,64) # 'When' prediction learner
#         self.td = self.learner # not really used
        
        self.active_joint_holder = None
        
        self.moving = 0
        self.switched = 0
        self.trace_gripper = 0
        self.trace_wrist_rotation = 0
        self.trace_wrist_flex = 0
        self.trace_elbow = 0
        self.trace_shoulder = 0
        self.long_trace_gripper = 0
        self.long_trace_wrist_rotation = 0
        self.long_trace_wrist_flex = 0
        self.long_trace_elbow = 0
        self.long_trace_shoulder = 0
        self.max_gripper = 0
        self.max_shoulder = 0
        self.max_elbow = 0
        self.max_wrist_rot = 0
        self.max_wrist_flex = 0
        self.previous_high_prediction = 0
        self.autonomy = None
        self.change_list = None
        self.no_reward = 0
        self.change_order = rospy.ServiceProxy('/bento/configure_group', ConfigureGroup)
        self.auto_switch = rospy.ServiceProxy('/bento/toggle_group', ToggleGroup)
        self.default_joint_list = [JointGroupJoint(joint_id=1, min_speed=0, max_speed=1),\
                            JointGroupJoint(joint_id=2, min_speed=0, max_speed=1),\
                             JointGroupJoint(joint_id=3, min_speed=0, max_speed=1),\
                              JointGroupJoint(joint_id=4, min_speed=0, max_speed=1),\
                               JointGroupJoint(joint_id=5, min_speed=0, max_speed=1)]
        
        """ Names and types of publishers """
        self.publisher_learner_gripper = rospy.Publisher('/agents/prediction_gripper', Float32Timed, queue_size = 10)
        self.publisher_return_gripper = rospy.Publisher('/agents/return_gripper', Float32Timed, queue_size = 10)
        self.publisher_reward_gripper = rospy.Publisher('/agents/reward_gripper', Float32Timed, queue_size = 10)
        self.publisher_learner_wrist_rotation = rospy.Publisher('/agents/prediction_wrist_rotation', Float32Timed, queue_size = 10)
        self.publisher_return_wrist_rotation = rospy.Publisher('/agents/return_wrist_rotation', Float32Timed, queue_size = 10)
        self.publisher_reward_wrist_rotation = rospy.Publisher('/agents/reward_wrist_rotation', Float32Timed, queue_size = 10)
        self.publisher_learner_wrist_flex = rospy.Publisher('/agents/prediction_wrist_flex', Float32Timed, queue_size = 10)
        self.publisher_return_wrist_flex = rospy.Publisher('/agents/return_wrist_flex', Float32Timed, queue_size = 10)
        self.publisher_reward_wrist_flex = rospy.Publisher('/agents/reward_wrist_flex', Float32Timed, queue_size = 10)
        self.publisher_learner_elbow = rospy.Publisher('/agents/prediction_elbow', Float32Timed, queue_size = 10)
        self.publisher_return_elbow = rospy.Publisher('/agents/return_elbow', Float32Timed, queue_size = 10)
        self.publisher_reward_elbow = rospy.Publisher('/agents/reward_elbow', Float32Timed, queue_size = 10)
        self.publisher_learner_shoulder = rospy.Publisher('/agents/prediction_shoulder', Float32Timed, queue_size = 10)
        self.publisher_return_shoulder = rospy.Publisher('/agents/return_shoulder', Float32Timed, queue_size = 10)
        self.publisher_reward_shoulder = rospy.Publisher('/agents/reward_shoulder', Float32Timed, queue_size = 10)
        self.publisher_learner_switch = rospy.Publisher('/agents/prediction_switch', Float32Timed, queue_size = 10)
        self.publisher_return_switch = rospy.Publisher('/agents/return_switch', Float32Timed, queue_size = 10)
        self.publisher_reward_switch = rospy.Publisher('/agents/reward_switch', Float32Timed, queue_size = 10)
#         self.publisher_joint_order = rospy.Publisher('/joint_order', Float32Timed, queue_size = 10)
#         self.publisher_learner2_shoulder = rospy.Publisher('/Agent_Predict_Shoulder', Float32Timed, queue_size = 10)
        self.publisher_switch_count = rospy.Publisher('/switch_count', Float32Timed, queue_size = 10)
        self.publisher_adaptation = rospy.Publisher('/adaptation', Float32Timed, queue_size = 10)
#         self.publisher_joint_ids = rospy.Publisher('/joint_ids', tuple(1), queue_size = 10)
        self.publisher_id1 = rospy.Publisher('/joint_order/id1', Float32Timed, queue_size = 10)
        self.publisher_id2 = rospy.Publisher('/joint_order/id2', Float32Timed, queue_size = 10)
        self.publisher_id3 = rospy.Publisher('/joint_order/id3', Float32Timed, queue_size = 10)
        self.publisher_id4 = rospy.Publisher('/joint_order/id4', Float32Timed, queue_size = 10)
        self.publisher_id5 = rospy.Publisher('/joint_order/id5', Float32Timed, queue_size = 10)
        self.publisher_moving = rospy.Publisher('/joint_moving', Float32Timed, queue_size = 10)
        self.publisher_moved = rospy.Publisher('/joint_moved', Float32Timed, queue_size = 10)
        self.publisher_cycletime = rospy.Publisher('/cycle_time', Float32Timed, queue_size = 10)
        self.publisher_autonomy = rospy.Publisher('/autonomy', Float32Timed, queue_size = 10)
        self.publisher_switch = rospy.Publisher('/switch', Float32Timed, queue_size = 10)
        self.publisher_change_list = rospy.Publisher('/change_list', Float32Timed, queue_size = 10)
        self.publisher_learner2_gripper = rospy.Publisher('/agents/prediction2_gripper', Float32Timed, queue_size = 10)
        self.publisher_learner2_wrist_rotation = rospy.Publisher('/agents/prediction2_wrist_rotation', Float32Timed, queue_size = 10)
        self.publisher_learner2_wrist_flex = rospy.Publisher('/agents/prediction2_wrist_flex', Float32Timed, queue_size = 10)
        self.publisher_learner2_elbow = rospy.Publisher('/agents/prediction2_elbow', Float32Timed, queue_size = 10)
        self.publisher_learner2_shoulder = rospy.Publisher('/agents/prediction2_shoulder', Float32Timed, queue_size = 10)
    
    def update_perception(self, gripper_states, wrist_flexion_states, 
        wrist_rotation_states, shoulder_rotation_states, 
        elbow_flexion_states, joint_activity_states):
        
        """ Update joint information """
        self.gripper_states = gripper_states
        self.wrist_flexion_states = wrist_flexion_states
        self.wrist_rotation_states = wrist_rotation_states
        self.shoulder_rotation_states = shoulder_rotation_states
        self.elbow_flexion_states = elbow_flexion_states
        self.joint_activity_states = joint_activity_states
        
        """ Do experiment-related updates """
        self.step()
        
#     @timed
    def step(self):
        
        self.number_of_steps += 1
        print 'steps = ' + str(self.number_of_steps)
        
        self.active_joint = self.joint_activity_states.joint_id
        
        if self.number_of_steps == 1:
            self.start_time = rospy.get_rostime()
            self.new_clock_time = self.start_time
            self.change_order(group_idx = 0, joints = self.default_joint_list, reset_idx=True) # send default joint list
            
        
        self.old_clock_time = self.new_clock_time
        
        clock_time = rospy.get_rostime()
        
        header = Header()
        header.stamp = clock_time
        
        self.new_clock_time = clock_time
        
        avg_step = (clock_time - self.start_time).to_sec()*1000/(self.number_of_steps)
        cycle_time = float((self.new_clock_time - self.old_clock_time).to_sec()*1000)
        
        print 'average step length = ' + str(avg_step) + ' ms'
        print 'cycle time = ' + str(cycle_time) + ' ms'
        
        """ Setting up the switch signal """
        if (self.active_joint_holder != self.joint_activity_states.joint_id):
            self.switch_flag = 1 
        else:
            self.switch_flag = 0 
        self.active_joint_holder = self.joint_activity_states.joint_id      
        print 'joint id = ' + str(self.joint_activity_states.joint_id)  
                  
            
        """ Is the arm moving? """
        if (numpy.absolute(self.gripper_states.velocity) > 0.2 or\
                numpy.absolute(self.wrist_rotation_states.velocity) > 0.2 or\
                    numpy.absolute(self.wrist_flexion_states.velocity) > 0.2 or\
                        numpy.absolute(self.elbow_flexion_states.velocity) > 0.2 or\
                            numpy.absolute(self.shoulder_rotation_states.velocity) > 0.2):
            self.moving = 1
        else:
            self.moving = 0
            
        """ switched = 1 if the user has switched and not yet moved the arm """
        if (self.switch_flag == 1):
            self.switched = 1
            self.switch_count += 1
        self.switched = self.switched
        if (self.moving == 1):
            self.switched = 0
#         print 'switched? ' + str(self.switched)
            
        """ moved = 1 if the user has moved the arm but not yet switched to a new joint """     
        if (self.switched == 1):
            self.moved = 0
        elif (self.switched == 0):
            self.moved = 1
#         print 'switch count = ' + str(self.switch_count)
        
        """ Short traces of joint movement """    
        if (numpy.absolute(self.gripper_states.velocity) > 0.2):
            self.trace_gripper = 1
        else:
            self.trace_gripper = self.trace_gripper * 0.99
        if (numpy.absolute(self.wrist_rotation_states.velocity) > 0.2):
            self.trace_wrist_rotation = 1
        else:
            self.trace_wrist_rotation = self.trace_wrist_rotation * 0.99
        if (numpy.absolute(self.wrist_flexion_states.velocity) > 0.2):
            self.trace_wrist_flex = 1
        else:
            self.trace_wrist_flex = self.trace_wrist_flex * 0.99
        if (numpy.absolute(self.elbow_flexion_states.velocity) > 0.2):
            self.trace_elbow = 1
        else:
            self.trace_elbow = self.trace_elbow * 0.99
        if (numpy.absolute(self.shoulder_rotation_states.velocity) > 0.2):
            self.trace_shoulder = 1
        else:
            self.trace_shoulder = self.trace_shoulder * 0.99
        """ Long traces of joint movement """    
        if (numpy.absolute(self.gripper_states.velocity) > 0.2):
            self.long_trace_gripper = 1
        else:
            self.long_trace_gripper = self.trace_gripper * 0.999
        if (numpy.absolute(self.wrist_rotation_states.velocity) > 0.2):
            self.long_trace_wrist_rotation = 1
        else:
            self.long_trace_wrist_rotation = self.trace_wrist_rotation * 0.999
        if (numpy.absolute(self.wrist_flexion_states.velocity) > 0.2):
            self.long_trace_wrist_flex = 1
        else:
            self.long_trace_wrist_flex = self.trace_wrist_flex * 0.999
        if (numpy.absolute(self.elbow_flexion_states.velocity) > 0.2):
            self.long_trace_elbow = 1
        else:
            self.long_trace_elbow = self.trace_elbow * 0.999
        if (numpy.absolute(self.shoulder_rotation_states.velocity) > 0.2):
            self.long_trace_shoulder = 1
        else:
            self.long_trace_shoulder = self.trace_shoulder * 0.999
        
        """ Defines the state for joint predictions. Includes current position and 
            velocity for all 5 joints. """
        norm_const = 6 # this is the number of bins
#         state_joints = [(self.shoulder_rotation_states.normalized_position)*16]
#         state_joints = [self.shoulder_rotation_states.normalized_position*16,\
#                    (self.shoulder_rotation_states.velocity+1.5)*16/4,\
#                      self.elbow_flexion_states.normalized_position*16,\
#                        (self.elbow_flexion_states.velocity+1.5)*16/4,\
#                          self.wrist_rotation_states.normalized_position*16,\
#                            (self.wrist_rotation_states.velocity+1.5)*16/4,\
#                              self.wrist_flexion_states.normalized_position*16,\
#                                (self.wrist_flexion_states.velocity+1.5)*16/4,\
#                                  self.gripper_states.normalized_position*16,\
#                                    (self.gripper_states.velocity+1.5)*16/4,\
#                                      self.trace_shoulder*16, self.trace_elbow*16,\
#                                        self.trace_wrist_rotation*16, self.trace_wrist_flex*16,\
#                                          self.trace_gripper*16,\
#                                            self.gripper_states.normalized_load*16]
        state_joints = [(self.shoulder_rotation_states.normalized_position)*norm_const,\
                      (self.elbow_flexion_states.normalized_position-0.4)*2.7*norm_const,\
                              (self.wrist_flexion_states.normalized_position+0.1)*norm_const/1.3,\
                                  ((self.gripper_states.normalized_position*100)-0.141)*17*norm_const,\
                                   (self.shoulder_rotation_states.velocity+1.5)*norm_const/4,\
                                   (self.elbow_flexion_states.velocity+1.5)*norm_const/4,\
                                   (self.wrist_flexion_states.velocity+1.5)*norm_const/4,\
                                   (self.gripper_states.velocity+1.5)*norm_const/4,\
                                                self.gripper_states.normalized_load*norm_const]
        self.numstates_joints = len(state_joints)
#         print self.shoulder_rotation_states.normalized_position
#         print (self.elbow_flexion_states.normalized_position-0.40)*2.7
#         print (self.wrist_flexion_states.normalized_position+0.1)/1.3
#         print ((self.gripper_states.normalized_position*100)-0.141)*17
#         print (self.shoulder_rotation_states.velocity+1.5)/4
#         print (self.elbow_flexion_states.velocity+1.5)/4
#         print (self.wrist_flexion_states.velocity+1.5)/4
#         print (self.gripper_states.velocity+1.5)/4
               
# self.trace_shoulder*norm_const, self.trace_elbow*norm_const,\
#                                         self.trace_wrist_flex*norm_const,\
#                                           self.trace_gripper*norm_const, \

#         (self.shoulder_rotation_states.velocity+1.5)*norm_const/4,\
#         (self.elbow_flexion_states.velocity+1.5)*norm_const/4,\
#         (self.wrist_rotation_states.velocity+1.5)*norm_const/4,\
#         (self.wrist_flexion_states.velocity+1.5)*norm_const/4,\
#         (self.gripper_states.velocity+1.5)*norm_const/4,\
    
#         self.wrist_rotation_states.normalized_position*norm_const,\

#         self.long_trace_shoulder*norm_const,\
#                                             self.long_trace_elbow*norm_const, self.long_trace_wrist_rotation*norm_const,\
#                                               self.long_trace_wrist_flex*norm_const, self.long_trace_gripper*norm_const,\
        """ Defines the state for the switch signal -- includes velocity, position, and traces """
#         state_switch = [self.shoulder_rotation_states.normalized_position]
        
#         state_switch = [self.shoulder_rotation_states.normalized_position*norm_const,\
#                    (self.shoulder_rotation_states.velocity+1.5)*norm_const/4,\
#                      self.elbow_flexion_states.normalized_position*norm_const,\
#                        (self.elbow_flexion_states.velocity+1.5)*norm_const/4,\
#                          self.wrist_rotation_states.normalized_position*norm_const,\
#                            (self.wrist_rotation_states.velocity+1.5)*norm_const/4,\
#                              self.wrist_flexion_states.normalized_position*norm_const,\
#                                (self.wrist_flexion_states.velocity+1.5)*norm_const/4,\
#                                  self.gripper_states.normalized_position*norm_const,\
#                                    (self.gripper_states.velocity+1.5)*norm_const/4]

        state_switch = [(self.shoulder_rotation_states.normalized_position)*norm_const,\
                         (self.elbow_flexion_states.normalized_position-0.4)*2.7*norm_const,\
                            (self.wrist_flexion_states.normalized_position+0.1)*norm_const/1.3,\
                               ((self.gripper_states.normalized_position*100)-0.141)*17*norm_const,\
                                  (self.shoulder_rotation_states.velocity+1.5)*norm_const/4,\
                                     (self.elbow_flexion_states.velocity+1.5)*norm_const/4,\
                                        (self.wrist_flexion_states.velocity+1.5)*norm_const/4,\
                                           (self.gripper_states.velocity+1.5)*norm_const/4,\
                                              self.gripper_states.normalized_load*norm_const]
        
        """ Defines the reward for each joint. Reward equals 1 when current joint velocity > 0.1 """
        if (self.shoulder_rotation_states.normalized_position == None or\
              self.shoulder_rotation_states.velocity == None or\
                self.elbow_flexion_states.normalized_position == None or\
                  self.elbow_flexion_states.velocity == None or\
                    self.wrist_rotation_states.normalized_position == None or\
                      self.wrist_rotation_states.velocity == None or\
                        self.wrist_flexion_states.normalized_position == None or\
                          self.wrist_flexion_states.velocity == None or\
                            self.gripper_states.normalized_position == None or\
                              self.gripper_states.velocity == None): # safety check; make sure what you're putting into the agent is what you're expecting
            print "incomplete state information"   
        else:
            reward_joints = [0]*5
            if numpy.absolute(self.gripper_states.velocity) > 0.2: # if hand is opening/closing
                reward_joints[0] = 1 # make the reward 1 for hand joint
            if numpy.absolute(self.wrist_rotation_states.velocity) > 0.3: # if wrist is rotating
                reward_joints[1] = 1 # make the reward 1 for wrist rotation joint
            if numpy.absolute(self.wrist_flexion_states.velocity) > 0.3: # if wrist is flexing/extending
                reward_joints[2] = 1 # make the reward 1 for wrist flex/extend joint
            if numpy.absolute(self.elbow_flexion_states.velocity) > 0.2: # if elbow is flexing/extending
                reward_joints[3] = 1 # make the reward 1 for elbow joint
            if numpy.absolute(self.shoulder_rotation_states.velocity) > 0.2: # if shoulder is rotating
                reward_joints[4] = 1 # make the reward for shoulder joint
            
            
        """ Defines the reward for switching. Reward = 1 when switch_signal = 1. When no_reward = 1, switch
            signal is computer-generated and no reward is given """
        reward_switch = 0
        if (self.switch_flag == 1 and self.no_reward == 0):
            reward_switch = 1
        self.no_reward = 0
        
        """ Updates to the td learners """
#             self.td.update(state_joints, reward_joints) # not really used
        self.td0.update(state_joints, reward_joints[0])
        self.td1.update(state_joints, reward_joints[1])
#         self.td2.update(state_joints, self.numstates_joints, reward_joints[2])
        self.td2.update(state_joints, reward_joints[2])
        self.td3.update(state_joints, reward_joints[3])
        self.td4.update(state_joints, reward_joints[4])
        self.td_switch.update(state_switch, reward_switch)
        
        """ Publish reward for each joint """
        self.publisher_reward_gripper.publish(header, reward_joints[0])
        self.publisher_reward_wrist_rotation.publish(header, reward_joints[1])
        self.publisher_reward_wrist_flex.publish(header, reward_joints[2])
        self.publisher_reward_elbow.publish(header, reward_joints[3])
        self.publisher_reward_shoulder.publish(header, reward_joints[4])
        self.publisher_reward_switch.publish(header, reward_switch)
        
        """ Publish prediction for each joint """
        self.publisher_learner_gripper.publish(header, self.td0.prediction)
        self.publisher_learner_wrist_rotation.publish(header, self.td1.prediction)
        self.publisher_learner_wrist_flex.publish(header, self.td2.prediction) # change this to self.td2.current_prediction for my learner
        self.publisher_learner_elbow.publish(header, self.td3.prediction)
        self.publisher_learner_shoulder.publish(header, self.td4.prediction)
        self.publisher_learner_switch.publish(header, self.td_switch.prediction)

        """ Publish return for each joint """
        if self.td2.verifier.calculateReturn() != None :
            self.publisher_return_gripper.publish(header, self.td0.verifier.calculateReturn())
            self.publisher_return_wrist_rotation.publish(header, self.td1.verifier.calculateReturn())
            self.publisher_return_wrist_flex.publish(header, self.td2.verifier.calculateReturn())
            self.publisher_return_elbow.publish(header, self.td3.verifier.calculateReturn())
            self.publisher_return_shoulder.publish(header, self.td4.verifier.calculateReturn())
            self.publisher_return_switch.publish(header, self.td_switch.verifier.calculateReturn())
        
        """ Prediction for each of the joints. The currently active joint must not be 
            the highest prediction; set the active joint prediction value to -1 """
        if (self.joint_activity_states.joint_id == 1): # i.e. shoulder
            self.shoulder_prediction = -1.0
        else:
            self.shoulder_prediction = self.td4.prediction
        if (self.joint_activity_states.joint_id == 2): # i.e. elbow
            self.elbow_prediction = -1.0
        else:
            self.elbow_prediction = self.td3.prediction
        if (self.joint_activity_states.joint_id == 3): # i.e. wrist rotation
            self.wrist_rotation_prediction = -1.0
        else:
            self.wrist_rotation_prediction = self.td1.prediction
        if (self.joint_activity_states.joint_id == 4): # i.e. wrist flexion
            self.wrist_flexion_prediction = -1.0
        else:
            self.wrist_flexion_prediction = self.td2.prediction
        if (self.joint_activity_states.joint_id == 5): # i.e. gripper
            self.hand_prediction = -1.0
        else:
            self.hand_prediction = self.td0.prediction
        self.switch_prediction = self.td_switch.prediction
        
#         print 'active joint = ' + str(self.joint_activity_states.joint_id)
        
        if numpy.absolute(self.gripper_states.velocity) > self.max_gripper:
            self.max_gripper = numpy.absolute(self.gripper_states.velocity)
        if numpy.absolute(self.wrist_flexion_states.velocity) > self.max_wrist_flex:
            self.max_wrist_flex = numpy.absolute(self.wrist_flexion_states.velocity)
        if numpy.absolute(self.wrist_rotation_states.velocity) > self.max_wrist_rot:
            self.max_wrist_rot = numpy.absolute(self.wrist_rotation_states.velocity)
        if numpy.absolute(self.elbow_flexion_states.velocity) > self.max_elbow:
            self.max_elbow = numpy.absolute(self.elbow_flexion_states.velocity)
        if numpy.absolute(self.shoulder_rotation_states.velocity) > self.max_shoulder:
            self.max_shoulder = numpy.absolute(self.shoulder_rotation_states.velocity)
            
        print 'elbow = ' + str(self.max_elbow)
        print 'gripper = ' + str(self.max_gripper)
        print 'shoulder = ' + str(self.max_shoulder)
        print 'wrist flex = ' + str(self.max_wrist_flex)
        print 'wrist rotation = ' + str(self.max_wrist_rot)
#         
       
        
        """ List of joints and their respective prediction values """
        self.joint_predictions = [(self.shoulder_prediction, 'shoulder'),(self.elbow_prediction, 'elbow'),\
                                  (self.wrist_flexion_prediction, 'wrist flex'),(self.wrist_rotation_prediction, 'wrist rotation'),\
                                  (self.hand_prediction, 'gripper')]


        """ Sort joints in joint_predictions based on lowest to highest prediction value """
        self.dtype = [('prediction', float),('joint', 'S10')] # data types in prediction array; used for sorting
        self.joint_order = numpy.array(self.joint_predictions, dtype=self.dtype) # list of joints and their respective predictions
        self.sorted_joints = numpy.sort(self.joint_order,order='prediction') # sorted array of joints (low to high)           
#         print 'sorted joints = ' + str(self.sorted_joints)
           
        
        
        """ List of joints can only adapt after the arm has moved again """
        if (self.moved == 1) and (self.switch_count >= 1) and (self.switched == 0):    
            self.adaptation = True
        else:
            self.adaptation = False
        print 'moved? ' + str(self.moved)
        print 'adaptation enabled = ' + str(self.adaptation)
        print 'switch count = ' + str(self.switch_count)
        
      
        """ new_list is the list of joints from the previous time-step """
        self.new_list = self.last_list
        
        """ joint_list is the ordered list formatted to send to the bento arm """
        if (self.sorted_joints[4][1] == 'shoulder'):
            self.last_list[0] = self.joint_list[0]
            self.joint_list[0] = JointGroupJoint(joint_id=1, min_speed=0, max_speed=1)
        elif (self.sorted_joints[3][1] == 'shoulder'):
            self.last_list[1] = self.joint_list[1]
            self.joint_list[1] = JointGroupJoint(joint_id=1, min_speed=0, max_speed=1)
        elif (self.sorted_joints[2][1] == 'shoulder'):
            self.last_list[2] = self.joint_list[2]
            self.joint_list[2] = JointGroupJoint(joint_id=1, min_speed=0, max_speed=1)
        elif (self.sorted_joints[1][1] == 'shoulder'):
            self.last_list[3] = self.joint_list[3]
            self.joint_list[3] = JointGroupJoint(joint_id=1, min_speed=0, max_speed=1)
        else:
            self.last_list[4] = self.joint_list[4]
            self.joint_list[4] = JointGroupJoint(joint_id=1, min_speed=0, max_speed=1)
                 
        if (self.sorted_joints[4][1] == 'elbow'):
            self.last_list[0] = self.joint_list[0]
            self.joint_list[0] = JointGroupJoint(joint_id=2, min_speed=0, max_speed=1)
        elif (self.sorted_joints[3][1] == 'elbow'):
            self.last_list[1] = self.joint_list[1]
            self.joint_list[1] = JointGroupJoint(joint_id=2, min_speed=0, max_speed=1)
        elif (self.sorted_joints[2][1] == 'elbow'):
            self.last_list[2] = self.joint_list[2]
            self.joint_list[2] = JointGroupJoint(joint_id=2, min_speed=0, max_speed=1)
        elif (self.sorted_joints[1][1] == 'elbow'):
            self.last_list[3] = self.joint_list[3]
            self.joint_list[3] = JointGroupJoint(joint_id=2, min_speed=0, max_speed=1)
        else:
            self.last_list[4] = self.joint_list[4]
            self.joint_list[4] = JointGroupJoint(joint_id=2, min_speed=0, max_speed=1)
             
        if (self.sorted_joints[4][1] == 'wrist rota'):
            self.last_list[0] = self.joint_list[0]
            self.joint_list[0] = JointGroupJoint(joint_id=3, min_speed=0, max_speed=1)
        elif (self.sorted_joints[3][1] == 'wrist rota'):
            self.last_list[1] = self.joint_list[1]
            self.joint_list[1] = JointGroupJoint(joint_id=3, min_speed=0, max_speed=1)
        elif (self.sorted_joints[2][1] == 'wrist rota'):
            self.last_list[2] = self.joint_list[2]
            self.joint_list[2] = JointGroupJoint(joint_id=3, min_speed=0, max_speed=1)
        elif (self.sorted_joints[1][1] == 'wrist rota'):
            self.last_list[3] = self.joint_list[3]
            self.joint_list[3] = JointGroupJoint(joint_id=3, min_speed=0, max_speed=1)
        else:
            self.last_list[4] = self.joint_list[4]
            self.joint_list[4] = JointGroupJoint(joint_id=3, min_speed=0, max_speed=1)
             
        if (self.sorted_joints[4][1] == 'wrist flex'):
            self.last_list[0] = self.joint_list[0]
            self.joint_list[0] = JointGroupJoint(joint_id=4, min_speed=0, max_speed=1)
        elif (self.sorted_joints[3][1] == 'wrist flex'):
            self.last_list[1] = self.joint_list[1]
            self.joint_list[1] = JointGroupJoint(joint_id=4, min_speed=0, max_speed=1)
        elif (self.sorted_joints[2][1] == 'wrist flex'):
            self.last_list[2] = self.joint_list[2]
            self.joint_list[2] = JointGroupJoint(joint_id=4, min_speed=0, max_speed=1)
        elif (self.sorted_joints[1][1] == 'wrist flex'):
            self.last_list[3] = self.joint_list[3]
            self.joint_list[3] = JointGroupJoint(joint_id=4, min_speed=0, max_speed=1)
        else:
            self.last_list[4] = self.joint_list[4]
            self.joint_list[4] = JointGroupJoint(joint_id=4, min_speed=0, max_speed=1)
             
        if (self.sorted_joints[4][1] == 'gripper'):
            self.last_list[0] = self.joint_list[0]
            self.joint_list[0] = JointGroupJoint(joint_id=5, min_speed=0, max_speed=1)
        elif (self.sorted_joints[3][1] == 'gripper'):
            self.last_list[1] = self.joint_list[1]
            self.joint_list[1] = JointGroupJoint(joint_id=5, min_speed=0, max_speed=1)
        elif (self.sorted_joints[2][1] == 'gripper'):
            self.last_list[2] = self.joint_list[2]
            self.joint_list[2] = JointGroupJoint(joint_id=5, min_speed=0, max_speed=1)
        elif (self.sorted_joints[1][1] == 'gripper'):
            self.last_list[3] = self.joint_list[3]
            self.joint_list[3] = JointGroupJoint(joint_id=5, min_speed=0, max_speed=1)
        else:
            self.last_list[4] = self.joint_list[4]
            self.joint_list[4] = JointGroupJoint(joint_id=5, min_speed=0, max_speed=1)  
        
        
#         print 'new list ' + str(self.new_list) # by the time it gets here, new_list has somehow changed to be equal to joint_list?!
#         print 'joint list ' + str(self.joint_list)
    
        """ Flag to check if list has changed since last list """   
        if (self.new_list != self.joint_list):
            self.change_list = 1
        
        """ If list has changed and adaptation is enabled, send new joint list to Bento """ 
        if (self.adaptation == True) and (self.change_list == 1):
            self.change_order(group_idx = 0, joints = self.joint_list, reset_idx=True)
            self.change_list = 0
            print 'JOINT LIST CHANGED'
#             time.sleep(0.01)
           
   
#         self.joint_ids = [self.joint_list[0].joint_id,self.joint_list[1].joint_id,self.joint_list[2].joint_id,self.joint_list[3].joint_id,self.joint_list[4].joint_id]
        
        """ Switching autonomy can only be enabled when prediction above specified threshold """
        if (self.switch_prediction > 0.2):
            self.high_prediction = 1
        else:
            self.high_prediction = 0
#         if (self.high_prediction == 1) and (self.previous_high_prediction == 0):
#             self.autonomy = 1
#         else:
#             self.autonomy = 0
#         self.previous_high_prediction = self.high_prediction
        
        """ Autonomous switch when prediction threshold is met, arm is not moving, and a switch has not occurred """
        if (self.high_prediction == 1) and (self.moving == 0) and (self.switched == 0):
            print 'AUTONOMY ENABLED'
            self.auto_switch(group_idx = 0)
            self.autonomy = 1
            self.switched = 1
            self.no_reward = 1
#             time.sleep(0.1)
            # need to also let the system know that a switch has occurred if it happens, so I can count the number of switches made by the system
        else:
            self.autonomy = 0

        print 'switch prediction = ' + str(self.switch_prediction)
        print 'joint order = ' + str(self.joint_order)
        print 'active joint = ' + str(self.active_joint)
#         print 'sorted joints = ' + str(self.sorted_joints)
#         print 'last list = ' + str(self.last_list)
#         
        

        """ Publish other things that might be needed for post-processing """
        self.publisher_switch_count.publish(header, self.switch_count)
        self.publisher_adaptation.publish(header, self.adaptation)
        self.publisher_id1.publish(header, self.joint_list[0].joint_id)
        self.publisher_id2.publish(header, self.joint_list[1].joint_id)
        self.publisher_id3.publish(header, self.joint_list[2].joint_id)
        self.publisher_id4.publish(header, self.joint_list[3].joint_id)
        self.publisher_id5.publish(header, self.joint_list[4].joint_id)
        self.publisher_moving.publish(header, self.moving)
        self.publisher_moved.publish(header, self.moved)
        self.publisher_cycletime.publish(header, cycle_time)
        self.publisher_autonomy.publish(header, self.autonomy)
        self.publisher_switch.publish(header, self.switch_flag)
        self.publisher_change_list.publish(header, self.change_list)
        self.publisher_learner2_wrist_flex.publish(header, self.elbow_prediction)
        self.publisher_learner2_shoulder.publish(header, self.shoulder_prediction)
        self.publisher_learner2_gripper.publish(header, self.hand_prediction)
        self.publisher_learner2_elbow.publish(header, self.elbow_prediction)
        self.publisher_learner2_wrist_rotation.publish(header, self.elbow_prediction)
#         self.publisher_joint_ids.publish(self.joint_ids)
        
    
        
#         print "gripper load = " + str(self.gripper_states.normalized_load)
        if (self.gripper_states.normalized_load > self.max_gripper_load):
            self.max_gripper_load = self.gripper_states.normalized_load  
        if (self.gripper_states.normalized_load < self.min_gripper_load):
            self.min_gripper_load = self.gripper_states.normalized_load
#         print 'min load = ' + str(self.min_gripper_load)   
#         print 'max load = ' + str(self.max_gripper_load)  
 
 
#             if (self.switch_flag == 1):
#                 self.adaptation_enabled = 0
#                 while (self.moved == 0):
#                     self.adaptation_enabled = 0
#                     
#             while (self.moved == 0):
#                 self.adaptation_enabled = 0
#                 if (self.moved == 1):
#                     self.adaptation_enabled = 1
                     
            
            
                        
        