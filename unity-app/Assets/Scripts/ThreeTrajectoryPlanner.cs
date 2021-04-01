using System;
using System.Collections;
using System.Collections.Generic;
using System.Linq;
using RosMessageTypes.Geometry;
using RosMessageTypes.NiryoMoveit;
using UnityEngine;
using System.Threading;

using ROSGeometry;
using Quaternion = UnityEngine.Quaternion;
using RosImage = RosMessageTypes.Sensor.Image;
using Transform = UnityEngine.Transform;
using Vector3 = UnityEngine.Vector3;


public class ThreeTrajectoryPlanner : MonoBehaviour
{
    // ROS Connector
    private ROSConnection ros;
    
    // Hardcoded variables 
    private int numRobotJoints = 6;
    private readonly float jointAssignmentWait = 0.05f;
    private readonly float poseAssignmentWait = 0.05f;
    private readonly Vector3 pickPoseOffset = Vector3.up * 0.1f;
    
    private readonly Quaternion pickOrientation = Quaternion.Euler(90, 0, 0);
    // For Opposite Direction Robot
    private readonly Quaternion pickOrientation2 = Quaternion.Euler(90, 0, 180);

    // Variables required for ROS communication
    public string rosServiceName = "niryo_moveit";

    [SerializeField]
    public GameObject[] niryoOne;
    public MoveObject[] target;
    [SerializeField]
    public GameObject[] targetPlacement;
    public Conveyor conveyor;

    [HideInInspector]
    public int robotTotalNum;
    [HideInInspector]
    public int messageNum = 0;

    // Articulation Bodies
    public ArticulationBody[,] jointArticulationBodies;
    private ArticulationBody[] leftGripper;
    private ArticulationBody[] rightGripper;

    private Transform[] gripperBase;
    private Transform[] leftGripperGameObject;
    private Transform[] rightGripperGameObject;
    private enum Poses
    {
        PreGrasp,
        Grasp,
        PickUp,
        Place
    };
    [HideInInspector]
    public bool[] moving;
    [HideInInspector]
    private float[] offset;
    private float waiting_time;
    public delegate void PlanningResultCallback(int robotid, int grasp_status, float waiting_time, float total_time);
    
    public PlanningResultCallback planning_result_callback;
    
    private float[,] original_joint_config;
    private float[] start_time;
    [HideInInspector]
    public bool reset_robot = false;
    private bool[] robotDirection;
    [HideInInspector]
    public bool stop = false;
    [HideInInspector]
    public int[] planned_obj_idx;
    

    private void CheckDirection(int robotid)
    {
        var rot = niryoOne[robotid].transform.rotation.y;
         // Debug.Log("Robot ID &&&& rotation y : " + robotid + "||||" + rot);
        if(rot == 0)
        {
            robotDirection[robotid] = true;
        }
        else
        {
            robotDirection[robotid] = false;
        }

    }

    /// <summary>
    ///     Close the gripper
    /// </summary>
    private void CloseGripper(int robotid)
    {
        var leftDrive = leftGripper[robotid].xDrive;
        var rightDrive = rightGripper[robotid].xDrive;
       
        leftDrive.target = -0.01f; //  0.018 or 0.014
        rightDrive.target = 0.01f;

        leftGripper[robotid].xDrive = leftDrive;
        rightGripper[robotid].xDrive = rightDrive;
    }

    /// <summary>
    ///     Open the gripper
    /// </summary>
    private void OpenGripper(int robotid)
    {
        var leftDrive = leftGripper[robotid].xDrive;
        var rightDrive = rightGripper[robotid].xDrive;

        leftDrive.target = 0.02f;
        rightDrive.target = -0.02f;

        leftGripper[robotid].xDrive = leftDrive;
        rightGripper[robotid].xDrive = rightDrive;
    }

    /// <summary>
    ///     Get the current values of the robot's joint angles.
    /// </summary>
    /// <returns>NiryoMoveitJoints</returns>
    NiryoMoveitJoints CurrentJointConfig(int robotid)
    {
        NiryoMoveitJoints joints = new NiryoMoveitJoints();
        
        joints.joint_00 = jointArticulationBodies[robotid,0].xDrive.target;
        joints.joint_01 = jointArticulationBodies[robotid,1].xDrive.target;
        joints.joint_02 = jointArticulationBodies[robotid,2].xDrive.target;
        joints.joint_03 = jointArticulationBodies[robotid,3].xDrive.target;
        joints.joint_04 = jointArticulationBodies[robotid,4].xDrive.target;
        joints.joint_05 = jointArticulationBodies[robotid,5].xDrive.target;

        return joints;
    }

    /// <summary>
    ///     Create a new MoverServiceRequest with the current values of the robot's joint angles,
    ///     the target cube's current position and rotation, and the targetPlacement position and rotation.
    ///
    ///     Call the MoverService using the ROSConnection and if a trajectory is successfully planned,
    ///     execute the trajectories in a coroutine.
    /// </summary>
    public async void PublishJoints(float x_, int robotid)
    {
        // Debug.Log("New pick and place planing");
        // Debug.Log("Sending Message to ROS / Robot ID : " + robotid);
        messageNum++;
        
        start_time[robotid] = Time.time;

        moving[robotid] = true;
        offset[robotid] = x_;
        
        MoverServiceRequest request = new MoverServiceRequest();
        request.joints_input = CurrentJointConfig(robotid);
        request.robotNumber = robotid;

        // Pick Pose
        Vector3 hard_code_position = new Vector3(x_, target[robotid].transform.position.y, target[robotid].transform.position.z);
        // Vector3 hard_code_position = new Vector3(0.0f, 0.748799f, 0.15f);
        request.pick_pose = new RosMessageTypes.Geometry.Pose
        {
            position = (hard_code_position + pickPoseOffset).To<FLU>(),
            // The hardcoded x/z angles assure that the gripper is always positioned above the target cube before grasping.
            orientation = Quaternion.Euler(90, target[robotid].transform.eulerAngles.y, 0).To<FLU>()
        };

        // Place Pose
        // Debug.Log("Robot ID & Direction: " +robotid +"|||" + robotDirection[robotid]);
        if(robotDirection[robotid])
        {
            request.place_pose = new RosMessageTypes.Geometry.Pose
            {
                    
                position = (targetPlacement[robotid].transform.position + pickPoseOffset).To<FLU>(),
                orientation = pickOrientation.To<FLU>()
            };
        }
        if(!robotDirection[robotid])
        {
            // Place Pose
            request.place_pose = new RosMessageTypes.Geometry.Pose
            {
                
                position = (targetPlacement[robotid].transform.position + pickPoseOffset).To<FLU>(),
                orientation = pickOrientation2.To<FLU>()
            };
        }

        ros.SendServiceMessage<MoverServiceResponse>(robotid, rosServiceName, request, TrajectoryResponse);
    }

    void TrajectoryResponse(MoverServiceResponse response, int robotid)
    {
        messageNum--;
        if (response.trajectories.Length > 0)
        {
            // Debug.Log("Trajectory returned / Robot ID : " + response.robotNumber);
            // Ien2[robotid] = StartCoroutine(Checking(response));

            StartCoroutine(Checking(response));
            
            // Ien2[robotid] = Checking(response);
            // StartCoroutine(Ien2[robotid]);
        
        }
        else
        {
            float total_time = Time.time - start_time[robotid];
            if (planning_result_callback != null)
            {
                planning_result_callback(robotid, -1, 0.0f , total_time);
            }
            moving[robotid] = false; 
            // Debug.Log("No trajectory returned from MoverService / Robot ID : " + robotid);
            planning_result_callback = null;
        }
        
    }

    public IEnumerator Checking(MoverServiceResponse response)
    {
        // Ien[response.robotNumber] = ExecuteTrajectories(response);
        // yield return StartCoroutine(Ien[response.robotNumber]);

        // yield return Ien[response.robotNumber]=StartCoroutine(ExecuteTrajectories(response));

        yield return StartCoroutine(ExecuteTrajectories(response));

        // Debug.Log("I finished Coroutine");
        
        if(target[response.robotNumber] != null && planning_result_callback != null)
        {
            float total_time = Time.time - start_time[response.robotNumber];
            if (target[response.robotNumber].moving == false)
            {
                planning_result_callback(response.robotNumber, 1, waiting_time, total_time);
            }
            else // late to reach obj
            {
                planning_result_callback(response.robotNumber, 0, waiting_time, total_time);
            }

            target[response.robotNumber].moving = false;
        }
        moving[response.robotNumber] = false;
        target[response.robotNumber] = null;
        planning_result_callback = null;
    }

    /// <summary>
    ///     Execute the returned trajectories from the MoverService.
    ///
    ///     The expectation is that the MoverService will return four trajectory plans,
    ///         PreGrasp, Grasp, PickUp, and Place,
    ///     where each plan is an array of robot poses. A robot pose is the joint angle values
    ///     of the six robot joints.
    ///
    ///     Executing a single trajectory will iterate through every robot pose in the array while updating the
    ///     joint values on the robot.
    /// 
    /// </summary>
    /// <param name="response"> MoverServiceResponse received from niryo_moveit mover service running in ROS</param>
    /// <returns></returns>
    private IEnumerator ExecuteTrajectories(MoverServiceResponse response)
    {
        var robotid = response.robotNumber;
        
        if (response.trajectories != null)
        {
            // For every trajectory plan returned
            for (int poseIndex  = 0 ; poseIndex < response.trajectories.Length; poseIndex++)
            {
                
                // For every robot pose in trajectory plan
                for (int jointConfigIndex  = 0 ; jointConfigIndex < response.trajectories[poseIndex].joint_trajectory.points.Length; jointConfigIndex++)
                {
                    var jointPositions = response.trajectories[poseIndex].joint_trajectory.points[jointConfigIndex].positions;
                    float[] result = jointPositions.Select(r=> (float)r * Mathf.Rad2Deg).ToArray();
                    // Set the joint values for every joint
                    for (int joint = 0; joint < jointArticulationBodies.GetLength(1); joint++)
                    {
                        var joint1XDrive  = jointArticulationBodies[robotid, joint].xDrive;
                        joint1XDrive.target = result[joint];
                        jointArticulationBodies[robotid, joint].xDrive = joint1XDrive;
                        
                    }
                    // Wait for robot to achieve pose for all joint assignments
                    yield return new WaitForSeconds(jointAssignmentWait);
                }
              
                // Close the gripper if completed executing the trajectory for the Grasp pose
                if (poseIndex == (int)Poses.PreGrasp && target[robotid] != null && reset_robot)
                {
                    // // Calculate how much time need to wait
                    // float distance_btw_griper_obj = offset[robotid] - target[robotid].transform.position.x;
                    // // TODO: waiting time..
                    // waiting_time = distance_btw_griper_obj/(target[robotid].speed) - 0.65f; // (original 0.8) Time.... planning time will matter FPS?
                    // // Debug.Log("Waiting Time  : " + waiting_time +"|||" + distance_btw_griper_obj +"|||" +target[robotid].speed);
                    
                    float distance_btw_griper_obj = offset[robotid] - target[robotid].transform.position.x;
                    waiting_time = distance_btw_griper_obj/conveyor.speed - 0.03f / conveyor.speed;
                    if (waiting_time >= 0.0f)
                    {
                        yield return new WaitForSeconds(waiting_time);
                    }
                    else
                    {
                        yield return new WaitForSeconds(0.0f);
                    }
                }
                

                // Close the gripper if completed executing the trajectory for the Grasp pose
                if (poseIndex == (int)Poses.Grasp)
                {
                    CloseGripper(robotid);
                }
                // Wait for the robot to achieve the final pose from joint assignment
                yield return new WaitForSeconds(poseAssignmentWait);
            }
            // All trajectories have been executed, open the gripper to place the target cube
            OpenGripper(robotid);
        }
    }

    /// <summary>
    ///     Find all robot joints in Awake() and add them to the jointArticulationBodies array.
    ///     Find left and right finger joints and assign them to their respective articulation body objects.
    /// </summary>
    void Start()
    {
        // Debug.Log("Start planing");
        
        // Get ROS connection static instance
        ros = ROSConnection.instance;
        robotTotalNum = niryoOne.Length;
        // Debug.Log("Total Robot Number : " + robotTotalNum);

        jointArticulationBodies = new ArticulationBody[robotTotalNum,numRobotJoints];
        
        offset = new float[robotTotalNum];

        gripperBase = new Transform[robotTotalNum];
        leftGripper = new ArticulationBody[robotTotalNum];
        rightGripper = new ArticulationBody[robotTotalNum];

        leftGripperGameObject = new Transform[robotTotalNum];
        rightGripperGameObject = new Transform[robotTotalNum];

        moving = new bool[robotTotalNum];
        start_time = new float[robotTotalNum];
        target = new MoveObject[robotTotalNum];
        robotDirection = new bool[robotTotalNum];
        planned_obj_idx = new int[robotTotalNum];
        
        // TODO : Change original joint config as deepcopy of articulation body
        original_joint_config = new float [robotTotalNum, numRobotJoints];
        for(int i=0; i<robotTotalNum;i++)
        {
            string shoulder_link = "world/base_link/shoulder_link";
            jointArticulationBodies[i,0] = niryoOne[i].transform.Find(shoulder_link).GetComponent<ArticulationBody>();
            var jointXDrive = jointArticulationBodies[i,0].xDrive;
            original_joint_config[i,0] = jointXDrive.target;


            string arm_link = shoulder_link + "/arm_link";
            jointArticulationBodies[i,1] = niryoOne[i].transform.Find(arm_link).GetComponent<ArticulationBody>();
            jointXDrive = jointArticulationBodies[i,1].xDrive;
            original_joint_config[i,1] = jointXDrive.target;

            string elbow_link = arm_link + "/elbow_link";
            jointArticulationBodies[i,2] = niryoOne[i].transform.Find(elbow_link).GetComponent<ArticulationBody>();
            jointXDrive = jointArticulationBodies[i,2].xDrive;
            original_joint_config[i,2] = jointXDrive.target;

            string forearm_link = elbow_link + "/forearm_link";
            jointArticulationBodies[i,3] = niryoOne[i].transform.Find(forearm_link).GetComponent<ArticulationBody>();
            jointXDrive = jointArticulationBodies[i,3].xDrive;
            original_joint_config[i,3] = jointXDrive.target;

            string wrist_link = forearm_link + "/wrist_link";
            jointArticulationBodies[i,4] = niryoOne[i].transform.Find(wrist_link).GetComponent<ArticulationBody>();
            jointXDrive = jointArticulationBodies[i,4].xDrive;
            original_joint_config[i,4] = jointXDrive.target;

            string hand_link = wrist_link + "/hand_link";
            jointArticulationBodies[i,5] = niryoOne[i].transform.Find(hand_link).GetComponent<ArticulationBody>();
            jointXDrive = jointArticulationBodies[i,5].xDrive;
            original_joint_config[i,5] = jointXDrive.target;

            // Find left and right fingers
            string right_gripper = hand_link + "/tool_link/gripper_base/servo_head/control_rod_right/right_gripper";
            string left_gripper = hand_link + "/tool_link/gripper_base/servo_head/control_rod_left/left_gripper";
            string gripper_base = hand_link + "/tool_link/gripper_base/Collisions/unnamed";

            gripperBase[i] = niryoOne[i].transform.Find(gripper_base);
            leftGripperGameObject[i] = niryoOne[i].transform.Find(left_gripper);
            rightGripperGameObject[i] = niryoOne[i].transform.Find(right_gripper);

            rightGripper[i] = rightGripperGameObject[i].GetComponent<ArticulationBody>();
            leftGripper[i] = leftGripperGameObject[i].GetComponent<ArticulationBody>();
            
            
            // TODO: Find way to deep copy original
            // Since NiryoMoveitJoint is message for ros so Modifyed original_joint_config
            // in to ArticulationBody Structure
            // original_joint_config = CurrentJointConfig(i);
            offset[i] = 0.0f;
            moving[i] = false;
            CheckDirection(i);
        }
    }

    private void ResetEachRobot(int robotid)
    {   
        // messageNum++;
        moving[robotid] = true;
        MoverServiceRequest request = new MoverServiceRequest();
        request.joints_input = CurrentJointConfig(robotid);
        request.robotNumber = robotid;
        
        if(robotDirection[robotid])
        {
            request.pick_pose = new RosMessageTypes.Geometry.Pose
            {
                position = (targetPlacement[robotid].transform.position + pickPoseOffset).To<FLU>(),
                orientation = pickOrientation.To<FLU>()
            };
            request.place_pose = new RosMessageTypes.Geometry.Pose
            {
                    
                position = (targetPlacement[robotid].transform.position + pickPoseOffset).To<FLU>(),
                orientation = pickOrientation.To<FLU>()
            };
        }
        else
        {
            request.pick_pose = new RosMessageTypes.Geometry.Pose
            {
                position = (targetPlacement[robotid].transform.position + pickPoseOffset).To<FLU>(),
                orientation = pickOrientation2.To<FLU>()
            };
            // Place Pose
            request.place_pose = new RosMessageTypes.Geometry.Pose
            {
                
                position = (targetPlacement[robotid].transform.position + pickPoseOffset).To<FLU>(),
                orientation = pickOrientation2.To<FLU>()
            };
        }
        ros.SendServiceMessage<MoverServiceResponse>(robotid, rosServiceName, request, TrajectoryResponse);
    }

    public IEnumerator ResetEachRobot()
    {
        for(int i = 0; i< robotTotalNum ; i++)
        {
            yield return new WaitForSeconds(2.0f);
            ResetEachRobot(i);
        }
    }
    private IEnumerator ResetCheck()
    {
        yield return StartCoroutine(ResetEachRobot());
        // Debug.Log("Resetting End");
        reset_robot = true;
    }

    public void ResetRobot()
    {
        stop = false;
        // niryoOne[0].GetComponent<RosSharp.Control.Controller>().damping = 100f;
        StartCoroutine(ResetCheck());

    }
    
}
