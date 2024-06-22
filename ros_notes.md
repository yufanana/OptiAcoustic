# Opti-Acoustic Sensor Fusion

## Usage

Start container

```bash
cd DFLS
docker compose up
docker exec -it dfls-dev-1 bash
```

Test installation

```bash
roslaunch dave_demo_launch dave_demo.launch
roslaunch nps_uw_multibeam_sonar sonar_tank_blueview_p900_nps_multibeam.launch
roslaunch gazebo_ros empty_world.launch
```

## Create a ROS package

```bash
cd overlay_ws/src
catkin_create_pkg opti_acoustic std_msgs rospy
cd ..
catkin build
```

## Setting up DAVE based on official docs

1. Docker environment

    - The host computer has ROS2 installed, so I installed the docker environment instead.
    - Set up NVIDIA driver from [this tutorial](https://www.linuxbabe.com/ubuntu/install-nvidia-driver-ubuntu), using the GUI method from "Additional drivers"
    - Install CUDA 12.5 from [this tutorial](https://www.cherryservers.com/blog/install-cuda-ubuntu)
    - Follow the [tutorial from DAVE](https://field-robotics-lab.github.io/dave.doc/contents/installation/Installation/)

2. Get source code

    ```bash
    mkdir -p ~/uuv_ws/src
    cd ~/uuv_ws/src
    git clone https://github.com/Field-Robotics-Lab/dave.git
    vcs import --skip-existing --input dave/extras/repos/dave_sim.repos .
    vcs import --skip-existing --input dave/extras/repos/multibeam_sim.repos .
    ```

3. Build environment inside the docker container

    ```bash
    ./run.bash -c dockwater:noetic
    cd ~/uuv_ws
    catkin build    # or catkin_make
    source ~/uuv_ws/devel/setup.bash
    ```

    - CMake error with PCL. In the container, run:

        ```bash
        sudo apt update
        sudo apt install libpcl-dev ros-noetic-pcl-ros
        ```

4. Testing installation

```bash
cd dockwater
./run.bash -c dockwater:noetic

cd uuv_ws
source devel/setup.bash
roslaunch dave_demo_launch dave_demo.launch
roslaunch nps_uw_multibeam_sonar sonar_tank_blueview_p900_nps_multibeam.launch
roslaunch gazebo_ros empty_world.launch
```

To force shutdown (after trying `Ctrl + C`)

```bash
pkill gzclient && pkill gzserver
```

I think there was an issue with the env variables. bad variable name/segmentaion fault (core dumped)

```bash
export PATH=/usr/local/cuda/bin${PATH:+:${PATH}}
export LD_LIBRARY_PATH=/usr/local/cuda-12.2/lib64\
                         ${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}
```

## Adding a camera

Goal: create a robot with camera sensor

- The following steps recreates the folders found in `dave/urdf/robots/caracara_description`
- `catkin_create_pkg caracara_description`
- Copy `launch`, `urdf`, `meshes` folders from `dave/urdf/robots/caracara_description`
- Add the following code to the end of `caracara_description/udf/caracara.xacro`

    ```xml
    <!-- Includes -->
    <xacro:property name="namespace" value="$(arg namespace)"/>
    <xacro:property name="inertial_reference_frame" value="world"/>
    <xacro:include filename="$(find uuv_sensor_ros_plugins)/urdf/sensor_snippets.xacro"/>
    <xacro:include filename="$(find uuv_gazebo_ros_plugins)/urdf/snippets.xacro"/>
    <xacro:include filename="$(find caracara_description)/urdf/caracara_sensors.xacro"/>
    ```

- Create a new file `caracara_description/urdf/caracara_sensors.xacro`

    ```xml
    <?xml version="1.0"?>

    <robot xmlns:xacro="http://www.ros.org/wiki/xacro">

    <!-- Mount a camera -->
    <xacro:default_camera namespace="${namespace}" parent_link="${namespace}/base_link" suffix="">
        <origin xyz="1.15 0 0.4" rpy="0 0.6 0"/>
    </xacro:default_camera>

    </robot>
    ```

Usage:

```bash
roslaunch opti_acoustic caracara.launch
roslaunch uuv_teleop uuv_keyboard_teleop.launch uuv_name:=caracara 
```

Explanation

- `roslaunch opti_acoustic caracara.launch` loads the world, sonar, and spawns the caracara robot.
- When spawning the robot, it runs `caracara_description/launch/upload_caracara.launch`
- `upload_caracara.launch` finds `caracara.xacro`
  - `caracara.xacro` defines the links, visual, inertial, geometry, sensors of the robot
- `caracara.xacro` uses `caracara_sensors.xacro`

## Questions

Gazebo

- why does the attached sonar not publish any images?
- why is the robot not responding to teleop? what topic is the robot subscribed to? what topic is teleop publishing?
