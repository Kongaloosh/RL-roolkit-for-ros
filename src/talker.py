#!/usr/bin/env python
import rosdep2 
import rospy
from std_msgs.msg import String

def talker():
    #td = tdLambda()
    pub = rospy.Publisher('chatter',String, queue_size=10)
    rospy.init_node('talker',anonymous = True)
    r = rospy.Rate(10)
    while not rospy.is_shutdown():
        str = "hi"
        rospy.loginfo(str)
        pub.publish(str)
        r.sleep()
        
if __name__ == '__main__':
    try:
        talker()
    except rospy.ROSInterruptException: pass
