#!/usr/bin/env python

from __future__ import print_function

import rospy

import sys
import copy
import math
import moveit_commander

import moveit_msgs.msg
from moveit_msgs.msg import Constraints, JointConstraint, PositionConstraint, OrientationConstraint, BoundingVolume
from sensor_msgs.msg import JointState
from moveit_msgs.msg import RobotState
import geometry_msgs.msg
from geometry_msgs.msg import Quaternion, Pose
from std_msgs.msg import String
from moveit_commander.conversions import pose_to_list

from niryo_moveit.srv import MoverService, MoverServiceRequest, MoverServiceResponse

joint_names = ['joint_1', 'joint_2', 'joint_3', 'joint_4', 'joint_5', 'joint_6']

# Between Melodic and Noetic, the return type of plan() changed. moveit_commander has no __version__ variable, so checking the python version as a proxy
if sys.version_info >= (3, 0):
    def planCompat(plan):
        return plan[1]
else:
    def planCompat(plan):
        return plan
        
"""
    Given the start angles of the robot, plan a trajectory that ends at the destination pose.
"""
def plan_trajectory(move_group, destination_pose, start_joint_angles): 
    current_joint_state = JointState()
    current_joint_state.name = joint_names
    current_joint_state.position = start_joint_angles

    moveit_robot_state = RobotState()
    moveit_robot_state.joint_state = current_joint_state
    move_group.set_start_state(moveit_robot_state)

    move_group.set_pose_target(destination_pose)
    plan = move_group.plan()

    if not plan:
        exception_str = """
            Trajectory could not be planned for a destination of {} with starting joint angles {}.
            Please make sure target and destination are reachable by the robot.
        """.format(destination_pose, destination_pose)
        raise Exception(exception_str)

    return planCompat(plan)


"""
    Creates a pick and place plan using the four states below.
    
    1. Pre Grasp - position gripper directly above target object
    2. Grasp - lower gripper so that fingers are on either side of object
    3. Pick Up - raise gripper back to the pre grasp position
    4. Place - move gripper to desired placement position

    Gripper behaviour is handled outside of this trajectory planning.
        - Gripper close occurs after 'grasp' position has been achieved
        - Gripper open occurs after 'place' position has been achieved

    https://github.com/ros-planning/moveit/blob/master/moveit_commander/src/moveit_commander/move_group.py
"""
def plan_pick_and_place(req):
    response = MoverServiceResponse()
    
    ## DH_MODI
    ns = "niryo_one"
    robot_des = "/robot_description"
    group_name = "arm"
    
    if(req.robotNumber != 0):
    	ns = ns+str(req.robotNumber)
    
    
    move_group = moveit_commander.MoveGroupCommander(group_name,(ns+robot_des),ns=ns)

    # TODO: Make message type a list instead
    current_robot_joint_configuration = [
        math.radians(req.joints_input.joint_00),
        math.radians(req.joints_input.joint_01),
        math.radians(req.joints_input.joint_02),
        math.radians(req.joints_input.joint_03),
        math.radians(req.joints_input.joint_04),
        math.radians(req.joints_input.joint_05),
    ]
    print(f"Get current pose: {move_group.get_current_pose()}")
    
    # Pre grasp - position gripper directly above target object
    pre_grasp_pose = plan_trajectory(move_group, req.pick_pose, current_robot_joint_configuration)

    previous_ending_joint_angles = pre_grasp_pose.joint_trajectory.points[-1].positions
    response.trajectories.append(pre_grasp_pose)

    # Grasp - lower gripper so that fingers are on either side of object
    pick_pose = copy.deepcopy(req.pick_pose)
    pick_pose.position.z -= 0.07  # Static value coming from Unity, TODO: pass along with request
    grasp_pose = plan_trajectory(move_group, pick_pose, previous_ending_joint_angles)

    previous_ending_joint_angles = grasp_pose.joint_trajectory.points[-1].positions
    response.trajectories.append(grasp_pose)

    # Pick Up - raise gripper back to the pre grasp position
    pick_up_pose = plan_trajectory(move_group, req.pick_pose, previous_ending_joint_angles)

    previous_ending_joint_angles = pick_up_pose.joint_trajectory.points[-1].positions
    response.trajectories.append(pick_up_pose)
    
    ## DH MODI 
    req.place_pose.position.z += 0.08
    # Place - move gripper to desired placement position
    place_pose = plan_trajectory(move_group, req.place_pose, previous_ending_joint_angles)
    
    previous_ending_joint_angles = place_pose.joint_trajectory.points[-1].positions
    response.trajectories.append(place_pose)

    move_group.clear_pose_targets()
    response.robotNumber = req.robotNumber
    return response


def moveit_server():
    moveit_commander.roscpp_initialize(sys.argv)
    rospy.init_node('niryo_moveit_server')
 

    s = rospy.Service('niryo_moveit', MoverService, plan_pick_and_place)
    print("Ready to plan")
    rospy.spin()


if __name__ == "__main__":
    moveit_server()
