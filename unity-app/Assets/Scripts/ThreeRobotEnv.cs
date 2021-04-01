using System;
using System.Collections;
using UnityEngine;
using Unity.MLAgents;
using Unity.MLAgents.Actuators;
using Unity.MLAgents.Sensors;
using Random = UnityEngine.Random;


public class ThreeRobotEnv : Agent
{
    public Conveyor conv;
    public ThreeTrajectoryPlanner planner;
    public MoveObject moveObj;
    public int agent_type = 0;

    private MoveObject[] moveObjects;
    private bool initilized = false;
    private bool conveyorStarted = false;

    private int episode_step = 0;
    private int currentObjIdx = -1;
    private int max_n_objs = 200;
    private int max_episode = 10;
    private int max_n_objs_for_planning = 5;
    private float start_time;
    private float middle_time;
    private float minInterval = 1.0f;
    private float maxInterval = 3.0f;
    private float[] state_info;
    private int n_other_states = 9;
    private int n_features = 4;
    private int n_picked_objs = 0;
    private float default_offet = -0.0f;
    private IEnumerator conveyorSchedule;
    private int ROBOT_ID = 0;

    // Start is called before the first frame update
    void Start()
    {
        // Debug.Log("Start()");
        // Create 100 moving objects
        moveObjects = new MoveObject[max_n_objs];
        for (int i = 0; i < max_n_objs; i++)
        {
            MoveObject obj = Instantiate(moveObj) as MoveObject;
            moveObjects[i] = obj;
        }

        state_info = new float[n_other_states+n_features*max_n_objs_for_planning];
        default_offet = conv.speed*4.5f;
    }

    private IEnumerator WaitRobot()
    {
        yield return new WaitUntil(() => !planner.moving[0] && !planner.moving[1] && !planner.moving[2]);
        planner.ResetRobot();
    }

    public override void OnEpisodeBegin()
    {
        initilized = false;
        
        planner.reset_robot = false;
        StartCoroutine(WaitRobot());
        
        for (int i = 0; i < max_n_objs; i++)
        {
            moveObjects[i].transform.localRotation = Quaternion.Euler(0f, 0f, 0f);
            moveObjects[i].transform.position = new Vector3(-2.1f - i*0.05f, 0.01f, 0.0f);
            moveObjects[i].selected = false;
        }

        if (conveyorSchedule != null)
        {
            StopCoroutine(conveyorSchedule);
        }

        ROBOT_ID = 0;
        n_picked_objs = 0;
        episode_step = 0;
        start_time = Time.time;
        middle_time = start_time;
        currentObjIdx = -1;
        conveyorStarted = false;
        
        
        conv.moveConveyor();
        
        initilized = true;
        // Debug.Log("OnEpisodeBegin | conveyorStarted: " + conveyorStarted);
    }

    public override void CollectObservations(VectorSensor sensor)
    {
        sensor.AddObservation(state_info);
    }

    private int get_random_target_idx()
    {
        int targetIdx = -1;
        float max_rand = -1.0f;
        int n_objs = 0;
        
        for (int obj_idx = 0; obj_idx <= currentObjIdx; obj_idx++)
        {
            if (moveObjects[obj_idx].moving == true && moveObjects[obj_idx].selected == false)
            {   
                var robot_position = planner.niryoOne[ROBOT_ID].transform.position;

                float maxreach_x = (float) Math.Sqrt(Math.Pow(0.4, 2) - 
                Math.Pow(moveObjects[obj_idx].transform.position.z - robot_position.z, 2));

                if (moveObjects[obj_idx].transform.position.x < (robot_position.x + maxreach_x - default_offet))
                {   
                    float rand = Random.Range(0.0f, 1.0f);

                    if (max_rand < rand)
                    {
                        targetIdx = obj_idx;
                        max_rand = rand;
                    }

                    n_objs += 1;
                }

            }

            if (n_objs >= max_n_objs_for_planning)
            {
                break;
            }
        }

        return targetIdx;
    }

    private int get_FSFP_target_idx()
    {
        int targetIdx = -1;
        
        for (int obj_idx = 0; obj_idx <= currentObjIdx; obj_idx++)
        {
            if (moveObjects[obj_idx].moving == true && moveObjects[obj_idx].selected == false)
            {
                var robot_position = planner.niryoOne[ROBOT_ID].transform.position;

                float maxreach_x = (float) Math.Sqrt(Math.Pow(0.4, 2) - 
                Math.Pow(moveObjects[obj_idx].transform.position.z - robot_position.z, 2));

                if (moveObjects[obj_idx].transform.position.x < (robot_position.x + maxreach_x - default_offet))
                {   
                    targetIdx = obj_idx;
                    break;
                }
            }
        }

        return targetIdx;
    }


    private int get_SP_target_idx()
    {
        int targetIdx = -1;
        float shortest_path = 1000.0f;
        int n_objs = 0;
        
        for (int obj_idx = 0; obj_idx <= currentObjIdx; obj_idx++)
        {
            if (moveObjects[obj_idx].moving == true && moveObjects[obj_idx].selected == false)
            {   
                var robot_position = planner.niryoOne[ROBOT_ID].transform.position;
                var placement_position = planner.targetPlacement[ROBOT_ID].transform.position;

                float maxreach_x = (float) Math.Sqrt(Math.Pow(0.4, 2) - 
                Math.Pow(moveObjects[obj_idx].transform.position.z - robot_position.z, 2));

                if (moveObjects[obj_idx].transform.position.x < (robot_position.x + maxreach_x - default_offet))
                {  
                    float new_path = (float) Math.Sqrt(Math.Pow(moveObjects[obj_idx].transform.position.x + default_offet - placement_position.x, 2) + 
                    Math.Pow(moveObjects[obj_idx].transform.position.z - placement_position.z, 2));

                    if (shortest_path > new_path)
                    {
                        targetIdx = obj_idx;
                        shortest_path = new_path;
                    }

                    n_objs += 1;
                }

            }

            if (n_objs >= max_n_objs_for_planning)
            {
                break;
            }
        }

        return targetIdx;
    }

    public override void OnActionReceived(ActionBuffers actionBuffers)
    {
        if (!planner.moving[ROBOT_ID] && initilized && conveyorStarted && currentObjIdx >= 0)
        {
            // Debug.Log("OnActionReceived: " + planner.moving[ROBOT_ID] + " | " + initilized + " | " + conveyorStarted + " | " + currentObjIdx);

            // Find object to pick up
            var raw_actions = actionBuffers.ContinuousActions[0];
            agent_type = actionBuffers.DiscreteActions[0];

            // Debug.Log("agent_type: " + agent_type + " | raw_actions: " + raw_actions);
            int picked_obj_idx;
            float offset;
            bool allow_to_plan = false;
            if (agent_type == 0)
            {
                picked_obj_idx = get_random_target_idx();
                allow_to_plan = true;
                var robot_position = planner.niryoOne[ROBOT_ID].transform.position;
                float maxreach_x = (float) Math.Sqrt(Math.Pow(0.4, 2) - 
                Math.Pow(moveObjects[picked_obj_idx].transform.position.z - robot_position.z, 2));

                offset = moveObjects[picked_obj_idx].transform.position.x + default_offet;
                if (offset < robot_position.x - maxreach_x)
                {
                    offset = robot_position.x - maxreach_x;
                }
            }
            else if (agent_type == 1)
            {
                picked_obj_idx = get_FSFP_target_idx();
                allow_to_plan = true;
                var robot_position = planner.niryoOne[ROBOT_ID].transform.position;
                float maxreach_x = (float) Math.Sqrt(Math.Pow(0.4, 2) - 
                Math.Pow(moveObjects[picked_obj_idx].transform.position.z - robot_position.z, 2));

                offset = moveObjects[picked_obj_idx].transform.position.x + default_offet;
                if (offset < robot_position.x-maxreach_x)
                {
                    offset = robot_position.x-maxreach_x;
                }
            }
            else if (agent_type == 2)
            {
                picked_obj_idx = get_SP_target_idx();
                allow_to_plan = true;
                var robot_position = planner.niryoOne[ROBOT_ID].transform.position;
                float maxreach_x = (float) Math.Sqrt(Math.Pow(0.4, 2) - 
                Math.Pow(moveObjects[picked_obj_idx].transform.position.z - robot_position.z, 2));

                offset = moveObjects[picked_obj_idx].transform.position.x + default_offet;
                if (offset < robot_position.x-maxreach_x)
                {
                    offset = robot_position.x-maxreach_x;
                }
            }
            else if (agent_type == 3)
            {
                // RL Agent
                if (actionBuffers.DiscreteActions[1] == -1)
                {
                    allow_to_plan = false;
                    picked_obj_idx = -1;
                    offset = 0.0f;
                }
                else
                {
                    picked_obj_idx = actionBuffers.DiscreteActions[1];

                    var robot_position = planner.niryoOne[ROBOT_ID].transform.position;
                    float maxreach_x = (float) Math.Sqrt(Math.Pow(0.4, 2) - 
                    Math.Pow(moveObjects[picked_obj_idx].transform.position.z - robot_position.z, 2));
                    

                    offset = moveObjects[picked_obj_idx].transform.position.x + default_offet;
                    if (offset < robot_position.x-maxreach_x)
                    {
                        offset = robot_position.x-maxreach_x;
                        allow_to_plan = true;
                    }
                    else if (offset > robot_position.x + maxreach_x)
                    {
                        allow_to_plan = false;
                    }
                    else
                    {
                        allow_to_plan = true;
                    }
                }
            }
            else
            {
                picked_obj_idx = get_random_target_idx();
                offset = 0.0f;
                allow_to_plan = true;
                var robot_position = planner.niryoOne[ROBOT_ID].transform.position;
                float maxreach_x = (float) Math.Sqrt(Math.Pow(0.4, 2) - 
                Math.Pow(moveObjects[picked_obj_idx].transform.position.z - robot_position.z, 2));

                offset = moveObjects[picked_obj_idx].transform.position.x + default_offet;
                if (offset < robot_position.x-maxreach_x)
                {
                    offset = robot_position.x-maxreach_x;
                }
            }
            
            if (picked_obj_idx >= 0 && allow_to_plan == true)
            {
                // Debug.Log("Current robot: " + ROBOT_ID + " | Current target ID: " + picked_obj_idx);

                moveObjects[picked_obj_idx].selected = true;
                planner.target[ROBOT_ID] = moveObjects[picked_obj_idx];
                planner.planning_result_callback = planning_result_callback;
                planner.PublishJoints(offset, ROBOT_ID);
            }
            else
            {
                // Debug.Log("Current target ID (Don't planning): " + target_idx);
                SetReward(-100.0f);
            }
        }
    }

    private void planning_result_callback(int robotid, int grasp_status, float waiting_time, float total_time)
    {
        // Fail to plan
        // Debug.Log("Reward from robot: " + robotid);
        var reward = 0.0f;
        if (grasp_status == -1) // Fail to plan
        {
            // Debug.Log("Reward: Fail to plan");
            // reward += 0.0f;
            reward += robotid*10000 - 100.0f;
            
        }
        else if (grasp_status == 0) // success to plan but fail to pick
        {
            // Debug.Log("Reward: late to reach obj");
            reward += robotid*10000 + 100.0f + total_time;
        }
        else // success to plan and success to pick
        {
            n_picked_objs += 1;
            // Debug.Log("Reward: success to plan and success to pick");
            reward += robotid*10000 + total_time;
            // reward += Time.time - start_time;
            // reward += n_passed_objs - n_last_passed_objs;
        }

        // Debug.Log("Reward: " + reward);
        SetReward(reward);
    } 

    private void new_object_come(int objIdx)
    {
        var randx = -1.95f;
        var targety = 0.748799f;
        var randz = Random.Range(-0.1f + 0.01f, 0.1f - 0.01f);
        moveObjects[objIdx].transform.localRotation = Quaternion.Euler(0f, 0f, 0f);
        moveObjects[objIdx].transform.position = new Vector3(randx, targety, randz);
        moveObjects[objIdx].moving = true;
        moveObjects[objIdx].selected = false;
    }

    public IEnumerator ConveyorSchedule()
    {
        // conveyorStarted = true;
        for (int i = 0; i < max_n_objs; i++)
        {
            // Add new obj
            currentObjIdx = i;
            // Debug.Log("New object come: " + currentObjIdx);
            new_object_come(currentObjIdx);
            yield return new WaitForSeconds(Random.Range(minInterval, maxInterval));
        }
    }

    private bool CanPlan(int robotid)
    {
        // Debug.Log("CanPlan");
        var robot_position = planner.niryoOne[robotid].transform.position;
    
        // Get robot state
        state_info[0] = planner.jointArticulationBodies[robotid, 0].xDrive.target;
        state_info[1] = planner.jointArticulationBodies[robotid, 1].xDrive.target;
        state_info[2] = planner.jointArticulationBodies[robotid, 2].xDrive.target;
        state_info[3] = planner.jointArticulationBodies[robotid, 3].xDrive.target;
        state_info[4] = planner.jointArticulationBodies[robotid, 4].xDrive.target;
        state_info[5] = planner.jointArticulationBodies[robotid, 5].xDrive.target;

        // var counter = Time.time - start_time;
        // state_info[6] = (float)Math.Sin((counter % 60.0) / 60.0 * 2 * Math.PI);
        // state_info[7] = (float)Math.Cos((counter % 60.0) / 60.0 * 2 * Math.PI);
        // state_info[8] = (float)Math.Sin(((counter / 60.0) % 60.0) / 60.0 * 2 * Math.PI);
        // state_info[9] = (float)Math.Cos(((counter / 60.0) % 60.0) / 60.0 * 2 * Math.PI);

        state_info[6] = (float)planner.targetPlacement[robotid].transform.position.x;
        state_info[7] = (float)planner.targetPlacement[robotid].transform.position.z;

        state_info[8] = robotid;

        int obj_count = 0;

        for (int obj_idx = 0; obj_idx <= currentObjIdx; obj_idx++)
        {
            float maxreach_x = (float) Math.Sqrt(Math.Pow(0.4, 2) - 
                Math.Pow(moveObjects[obj_idx].transform.position.z - robot_position.z, 2));
            if (moveObjects[obj_idx].moving == true && moveObjects[obj_idx].selected == false &&
            moveObjects[obj_idx].transform.position.x <= (robot_position.x + maxreach_x - default_offet))
            {
                state_info[n_other_states+obj_count*n_features] = obj_idx;
                state_info[n_other_states+obj_count*n_features+1] = robot_position.x - moveObjects[obj_idx].transform.position.x;
                state_info[n_other_states+obj_count*n_features+2] = moveObjects[obj_idx].transform.position.z;
                state_info[n_other_states+obj_count*n_features+3] = robot_position.x + maxreach_x - moveObjects[obj_idx].transform.position.x;
                obj_count += 1;

                if (obj_count == max_n_objs_for_planning)
                {
                    break;
                }
            }
        }
        if (obj_count == max_n_objs_for_planning)
        {
            return true;
        }
        else
        {
            return false;
        }
    }

    void FixedUpdate() 
    {
        if (initilized && !conveyorStarted && planner.reset_robot)
        {
            conveyorStarted = true;
            conveyorSchedule = ConveyorSchedule();
            StartCoroutine(conveyorSchedule);
        }

        if (initilized && conveyorStarted && (!planner.moving[0] || !planner.moving[1] || !planner.moving[2]) && planner.messageNum < 1 && planner.reset_robot)
        {
            // End episode condition
            if (currentObjIdx == max_n_objs - 1 || n_picked_objs >= max_episode)
            {
                // Debug.Log("End episode condition: " + currentObjIdx + " | " + moveObjects[currentObjIdx].moving);
                // if (!planner.moving[0] && !planner.moving[1] && !planner.moving[2])
                // {
                conv.stopConveyor();
                float running_time = Time.time - start_time;
                Debug.Log("End episode!!!!!!!!!!!!: " + running_time);
                
                EndEpisode();
                return;
                // }
            }
            else
            {
                if (!planner.moving[0] && CanPlan(0))
                {
                    // Debug.Log("RequestDecision 0");
                    ROBOT_ID = 0;
                    RequestDecision();
                }
                else if(!planner.moving[1] && CanPlan(1))
                {
                    // Debug.Log("RequestDecision 1");
                    ROBOT_ID = 1;
                    RequestDecision();
                }
                else if(!planner.moving[2] && CanPlan(2))
                {
                    // Debug.Log("RequestDecision 2");
                    ROBOT_ID =2;
                    RequestDecision();
                }
            }
        }
    }
}