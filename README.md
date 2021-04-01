# Application Framework for Conveyor Belt-based Pick-and-Place Systems  

---
## Requirement
1. OS: Linux Ubuntu 20.04LTS
2. Unity Hub: v.2020.2.2f1, ML-Agent 1.7.2
3. ROS Noetic
4. Install python3, pytorch

## Run ROS service
1. Move to ROS service folder: `cd ros-app`
2. Source ROS command: `source /opt/ros/noetic/setup.bash`
3. Build ROS service: `catkin_make`
4. Source devel: `source devel/setup.bash`
3. Run ROS service: `roslaunch niryo_moveit bin_picking.launch`

## Working with Unity App
1. Download Unity Hub (v.2020.2.2f1)
2. Open Unity project located at folder `unity-app`
3. Build Unity Simulator (I already build a simulator called `simulator.x86_64` that existed in folder `unity-app`)
4. Execute simulator app (Default algorithm is FSFP (a derived version of FIFO))

## Training Agent
1. Run environment only: `python3 robot_env.py`
2. Training agent with default parameters (given in `trainning.py` file): `python3 training.py`