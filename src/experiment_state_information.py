"""
Explanation
"""
class state (object):
    """ interface to hold structure for subscriptions """
    def __init__(self):
        pass
    
    def update (self):
        pass
    
"""
================================================================================

                                Servo Classes

================================================================================
"""
class servo_info (state):
    """ abstract class to hold basic implementation for servo subscriptions"""
    def __init__(self):
        self.position = 0
        self.load = 0
        self.velocity = 0
        self.is_moving = 0
        
        self.normalized_position = 0
        self.normalized_load = 0
        self.normalized_velocity = 0
        
    def update(self, position, load, velocity, is_moving):
        self.position = position
        self.load = load
        self.velocity = velocity
        self.is_moving = is_moving
        
class gripper(servo_info):
    """ specific instance of gripper
        
        {Gripper}
        * Position : 0 - 1023
        * Velocity
        * Load : -1 : 1
        * Is Moving : 0 - 1
    
    """
    def update(self, position, load, velocity, is_moving):
        servo_info.update(self, position, load, velocity, is_moving)
        self.normalized_position = float(self.position)/1023.0
        self.normalized_load = float(self.load+1)/2.0
        self.normalized_velocity = None

class not_gripper(servo_info):
    """ specific instance of all servos other than gripper
        
        {Wrist Flexion}
        * Position : 0 - 4095
        * Velocity
        * Load : -1 : 1
        * Is Moving : 0 - 1
    """
    def update(self, position, load, velocity, is_moving):
        servo_info.update(self, position, load, velocity, is_moving)
#         self.normalized_position = float(self.position)/4095.0
        self.normalized_position = (float(self.position)-1.5)/3.22
        self.normalized_load = float(self.load+1)/2.0
        self.normalized_velocity = None

"""
================================================================================

                            Joint Activity Classes

================================================================================
"""

class joint_activity(state):
    
    def __init__(self):
        self.active_joint = None
        self.switch = None
        self.active_joint_normalized = None
        
    def update(self, active_joint, switch):
        self.active_joint = active_joint
        self.switch = switch
        self.active_joint_normalized = float(self.active_joint)/5.0

    