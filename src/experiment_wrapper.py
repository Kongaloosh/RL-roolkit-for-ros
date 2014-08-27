#!/usr/bin/env python
from experiment_state_information import *
from learning_toolkit import *
import rospy
from std_msgs.msg import *

class experiment_wrapper(object):
    """ Abstract Class for Implementing an Experiment """
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
        
    def step(self):
        self.number_of_steps += 1
    
    def update_perception(self, gripper_states, wrist_flexion_states,\
                           wrist_rotation_states, shoulder_rotation_states,\
                           elbow_flexion_states, joint_activity_states):

        self.gripper_states = gripper_states
        self.wrist_flexion_states = wrist_flexion_states
        self.wrist_rotation_states = wrist_rotation_states
        self.shoulder_rotation_states = shoulder_rotation_states
        self.elbow_flexion_states = elbow_flexion_states
        self.joint_activity_states = joint_activity_states
        
class example_experiment(experiment_wrapper):
    """
    The job of the experiment handler is to take the information that it has available---the information
    from the bento arm---make sure it is well formatted, and feed it to the learner. The experiment handler
    is the learner's baby-sitter and makes sure that the learners are being given the information needed to
    perform their job.
    """
    def __init__(self):
        experiment_wrapper.__init__(self) #will still make all the things in the original experiment template
        self.td = TDLambdaLearner(1, 3, 0.5, 0.9, 0.9, 2**(2*1)) # on top of that we build a TD(/) learner
        self.publisher_learner = rospy.Publisher('/Agent_Prediction', Float32, queue_size = 10)
        self.publisher_return = rospy.Publisher('/Agent_Return', Float32, queue_size = 10)
    
    
    def update_perception(self, gripper_states, wrist_flexion_states, 
        wrist_rotation_states, shoulder_rotation_states, 
        elbow_flexion_states, joint_activity_states):
        
        experiment_wrapper.update_perception(self, gripper_states, wrist_flexion_states,\
                                              wrist_rotation_states, shoulder_rotation_states,\
                                               elbow_flexion_states, joint_activity_states)
        self.step()
        self.publisher_learner.publish(self.td.prediction)
        if self.td.verifier.calculateReturn() != None :
            self.publisher_return.publish(self.td.verifier.calculateReturn())
        
    def step(self):
        experiment_wrapper.step(self) #will still update in the same way as the template
        state = [self.shoulder_rotation_states.normalized_load,\
                  self.shoulder_rotation_states.normalized_position,\
                   (self.shoulder_rotation_states.velocity+2)/2] # define an agent's state

        if (self.shoulder_rotation_states.normalized_load == None or\
             self.shoulder_rotation_states.normalized_position == None or\
              self.shoulder_rotation_states.velocity == None): # safety check; make sure what you're putting into the agent is what you're expecting
            print "incomplete state information"
        
        else:
            reward = 0
            if self.joint_activity_states.active_joint == 1: #if the joint is the one you're interested in
                reward = 1 #make the target signal one
            self.td.update(state, reward) # update the agent
            
class example_experiment_predict_Position_shoulder(experiment_wrapper):
    """
    The job of the experiment handler is to take the information that it has available---the information
    from the bento arm---make sure it is well formatted, and feed it to the learner. The experiment handler
    is the learner's baby-sitter and makes sure that the learners are being given the information needed to
    perform their job.
    """
    def __init__(self, moving_average_multiplier = 0.1818):
        experiment_wrapper.__init__(self) #will still make all the things in the original experiment template
        self.td = TDLambdaLearner(1, 2, 0.5, 0.9, 0.9, 2**(2*1))
        self.true_online_td = True_Online_TD2(1, 3, 0.5, 0.9, 0.9, 2**(2*1))
        self.publisher_learner_TOD = rospy.Publisher('/True_Online_Prediction', Float32, queue_size = 10)
        self.publisher_return_TOD = rospy.Publisher('/True_Online_Return', Float32, queue_size = 10)
        self.publisher_learner_TD = rospy.Publisher('/TD_Prediction', Float32, queue_size = 10)
        self.publisher_return_TD = rospy.Publisher('/TD_Return', Float32, queue_size = 10)
        self.publisher_reward = rospy.Publisher('/reward', Float32, queue_size = 10)
        self.shoulder_position_moving_average = 0
        self.moving_average_multiplier = moving_average_multiplier
    
    def update_perception(self, gripper_states, wrist_flexion_states, 
        wrist_rotation_states, shoulder_rotation_states, 
        elbow_flexion_states, joint_activity_states):
        
        experiment_wrapper.update_perception(self, gripper_states, wrist_flexion_states,\
                                              wrist_rotation_states, shoulder_rotation_states,\
                                               elbow_flexion_states, joint_activity_states)
        self.step()
        self.publisher_learner_TD.publish(self.td.prediction)
        if self.td.verifier.calculateReturn() != None :
            self.publisher_return_TD.publish(self.td.verifier.calculateReturn())
        self.publisher_learner_TOD.publish(self.true_online_td.prediction)
        
    def step(self):
        experiment_wrapper.step(self) #will still update in the same way as the template
        
        state = [self.shoulder_rotation_states.normalized_load,\
                  self.shoulder_rotation_states.normalized_position,\
                   (self.shoulder_rotation_states.velocity+2)/2] # define an agent's state
        
        if (self.shoulder_rotation_states.normalized_load == None or\
             self.shoulder_rotation_states.normalized_position == None or\
              self.shoulder_rotation_states.velocity == None): # safety check; make sure what you're putting into the agent is what you're expecting
            print "incomplete state information"
        else:
            self.shoulder_position_moving_average = (self.shoulder_rotation_states.normalized_position - self.shoulder_position_moving_average)\
                                                    * self.moving_average_multiplier + self.shoulder_position_moving_average
            reward = self.shoulder_position_moving_average
            self.td.update(state, reward) # update the agent
            self.true_online_td.update(state, reward)
            self.publisher_reward.publish(reward)

class example_experiment_predict_shoulder_movement_true_online_vs_td2(experiment_wrapper):
    """
    The job of the experiment handler is to take the information that it has available---the information
    from the bento arm---make sure it is well formatted, and feed it to the learner. The experiment handler
    is the learner's baby-sitter and makes sure that the learners are being given the information needed to
    perform their job.
    """
    def __init__(self):
        experiment_wrapper.__init__(self) #will still make all the things in the original experiment template
        #Learners
        self.td = TDLambdaLearner(10, 2, 0.1, 0.99, 0.9,2**5)
        self.totd = True_Online_TD2(10, 2, 0.1, 0.99, 0.9,2**5)
        #publishers for graphing
        self.publisher_learner_TOD = rospy.Publisher('/True_Online_Prediction', Float32, queue_size = 10)
        self.publisher_return_TOD = rospy.Publisher('/True_Online_Return', Float32, queue_size = 10)
        self.publisher_learner_TD = rospy.Publisher('/TD_Prediction', Float32, queue_size = 10)
        self.publisher_return_TD = rospy.Publisher('/TD_Return', Float32, queue_size = 10)
        self.publisher_reward = rospy.Publisher('/reward', Float32, queue_size = 10)
        #variables for moving average
    
    # Kind of like a main, this is what's going to be looped over.
    def update_perception(self, gripper_states, wrist_flexion_states, 
        wrist_rotation_states, shoulder_rotation_states, 
        elbow_flexion_states, joint_activity_states):
        
        # Do the template update
        experiment_wrapper.update_perception(self, gripper_states, wrist_flexion_states,\
                                              wrist_rotation_states, shoulder_rotation_states,\
                                              elbow_flexion_states, joint_activity_states)
        # Do experiment related updates
        self.step()
       
        # Do publisher related things
        
        self.publisher_learner_TD.publish(self.td.verifier.getSyncedPrediction())
        self.publisher_learner_TOD.publish(self.totd.verifier.getSyncedPrediction())
        if self.td.verifier.calculateReturn() != None: # won't have a value until horizon is reached
            self.publisher_return_TD.publish(self.td.verifier.calculateReturn())
            print str(self.td.verifier.calculateReturn()) + " V " 
    def step(self):
        experiment_wrapper.step(self) #will still update in the same way as the template
        
        normalized_velocity =  self.shoulder_rotation_states.velocity
        if normalized_velocity > 0.5:
            normalized_velocity = 1
        elif normalized_velocity < -0.5:
            normalized_velocity = 0
        else:
            normalized_velocity = 0.5
        
        shoulder_rotation_position_normalized = self.shoulder_rotation_states.position /5
        
        if self.joint_activity_states.active_joint == None:
            self.joint_activity_states.update(0, False)
            
        state = [shoulder_rotation_position_normalized,\
                  normalized_velocity] # define an agent's state
        
        if (self.shoulder_rotation_states.normalized_load == None or\
             self.shoulder_rotation_states.normalized_position == None or\
              self.shoulder_rotation_states.velocity == None): # safety check; make sure what you're putting into the agent is what you're expecting
            print "incomplete state information"
        else:
            self.td.update(state, shoulder_rotation_position_normalized) # update the agent
            self.totd.update(state, shoulder_rotation_position_normalized)
            self.publisher_reward.publish(shoulder_rotation_position_normalized)
            
class example_experiment_predict_shoulder_movement_true_online_vs_td(experiment_wrapper):
    """
    The job of the experiment handler is to take the information that it has available---the information
    from the bento arm---make sure it is well formatted, and feed it to the learner. The experiment handler
    is the learner's baby-sitter and makes sure that the learners are being given the information needed to
    perform their job.
    """
    def __init__(self):
        experiment_wrapper.__init__(self) #will still make all the things in the original experiment template
        #Learners
        self.td = TDLambdaLearner(8, 2, 0.1, 0.99, 0.97,64)
        #publishers for graphing
        self.publisher_learner_TOD = rospy.Publisher('/True_Online_Prediction', Float32, queue_size = 10)
        self.publisher_return_TOD = rospy.Publisher('/True_Online_Return', Float32, queue_size = 10)
        self.publisher_learner_TD = rospy.Publisher('/TD_Prediction', Float32, queue_size = 10)
        self.publisher_return_TD = rospy.Publisher('/TD_Return', Float32, queue_size = 10)
        self.publisher_reward = rospy.Publisher('/reward', Float32, queue_size = 10)
        #variables for moving average
    
    # Kind of like a main, this is what's going to be looped over.
    def update_perception(self, gripper_states, wrist_flexion_states, 
        wrist_rotation_states, shoulder_rotation_states, 
        elbow_flexion_states, joint_activity_states):
        
        # Do the template update
        experiment_wrapper.update_perception(self, gripper_states, wrist_flexion_states,\
                                              wrist_rotation_states, shoulder_rotation_states,\
                                              elbow_flexion_states, joint_activity_states)
        # Do experiment related updates
        self.step()
       
        # Do publisher related things
        self.publisher_learner_TD.publish(self.td.prediction)
        if self.td.verifier.calculateReturn() != None: # won't have a value until horizon is reached
            self.publisher_return_TD.publish(self.td.verifier.calculateReturn())
        
        
    def step(self):
        experiment_wrapper.step(self) #will still update in the same way as the template
        
        normalized_velocity =  self.shoulder_rotation_states.velocity
        if normalized_velocity > 0.5:
            normalized_velocity = 1
        elif normalized_velocity < -0.5:
            normalized_velocity = 0.5
        else:
            normalized_velocity = 0
        
        shoulder_rotation_position_normalized = self.shoulder_rotation_states.position /5
        
        if self.joint_activity_states.active_joint == None:
            self.joint_activity_states.update(0, False)
            
        state = [shoulder_rotation_position_normalized,\
                  normalized_velocity] # define an agent's state
        
        if (self.shoulder_rotation_states.normalized_load == None or\
             self.shoulder_rotation_states.normalized_position == None or\
              self.shoulder_rotation_states.velocity == None): # safety check; make sure what you're putting into the agent is what you're expecting
            print "incomplete state information"
        else:
            reward = 0
            if numpy.absolute(self.shoulder_rotation_states.velocity) > 0.01:
                reward = 1
            self.td.update(state, reward) # update the agent
            self.publisher_reward.publish(reward)
        
        