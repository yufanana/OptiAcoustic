# Notes

## Announcement 1

The sensor has been added to: https://gitlab.gbar.dtu.dk/dtu-asl/courses/34763-autonomous-marine-robotics/-/blob/main/Training_Sessions/TS6_Perception/ts6_ws/src/ts6_bluerov2_perception/urdf/sensors.xacro#L76

Note that the simulated sonar returns a "LaserScan" message (http://docs.ros.org/en/api/sensor_msgs/html/msg/LaserScan.html), which is NOT the actual data returned by the real sonar.

The reason for using a simplified version is that the simulation of a real sensor is computationally quite heavy, i.e., it would require an onboard GPU for a reasonably smooth simulation, which means that for some of you without a GPU, it would render the simulation environment completely unusable.

To get a better understanding of HOW the sonar data actually looks like (spoiler: its an image), you can read the documentation provided in the Project Dave documentation here: https://field-robotics-lab.github.io/dave.doc/contents/dave_sensors/Multibeam-Forward-Looking-Sonar/ .

This means that, if you want to use the sonar data, you will have to find a way to convert the image, into something similar to the LaserScan message (that is: an array of distances for a given field of view, with some fixed angular increment).

## Announcement 2

We are sharing the bag files from our recent experiment at ASTA.

Nicholas and I recorded data at ASTA yesterday, resulting in three bag files:

1. oculus_*.bag: This file contains raw data from the sonar sensors. Please note that the sonar data type differs between the simulator and the real system, necessitating conversion procedures. To stream this data, you'll need to clone and build the ROS package available at: https://gitlab.com/apl-ocean-engineering/apl_msgs. You need to install this package before building it:

sudo apt-get install ros-noetic-genmypy

2. oak_*.bag: This file includes data from the Oak-D camera, including two grayscale cameras (on the left and right) and a single RGB camera in the center.

3. flightdata_*.bag: This file contains various data related to the robot's state, including MAVROS, IMU data, etc.

You can find these files here: Sample Data - 34763 Autonomous Marine Robotics Spring 24 (dtu.dk) We are sharing the bag files from our recent experiment at ASTA.

Nicholas and I recorded data at ASTA yesterday, resulting in three bag files:

1. oculus_*.bag: This file contains raw data from the sonar sensors. Please note that the sonar data type differs between the simulator and the real system, necessitating conversion procedures. To stream this data, you'll need to clone and build the ROS package available at: https://gitlab.com/apl-ocean-engineering/apl_msgs. You need to install this package before building it:

sudo apt-get install ros-noetic-genmypy

2. oak_*.bag: This file includes data from the Oak-D camera, including two grayscale cameras (on the left and right) and a single RGB camera in the center.

3. flightdata_*.bag: This file contains various data related to the robot's state, including MAVROS, IMU data, etc.

You can find these files here: Sample Data - 34763 Autonomous Marine Robotics Spring 24 (dtu.dk) 

## Announcement 3

Here I write the tutorial (suggested by Fletcher) to decode the raw real sonar data to more interpretable data (image or pointcloud)

I assume that you can stream the sonar data now. If not, please check my last post on Teams group: 



From the bag file, you can only see the raw sonar data. To decode the raw data, you need to do these steps:

Update the repo, I have added some ROS packages on ros_ws/src from:   
https://gitlab.com/apl-ocean-engineering/oculus_sonar_driver.git
https://github.com/apl-ocean-engineering/marine_msgs
https://github.com/apl-ocean-engineering/sonar_image_proc/tree/main
Compile the ros_ws workspace by "catkin_make"
Play the bag file (rosbag play oculus*.bag)
Convert sonar raw data to marine_acoustic_msgs/ProjectedSonarImage datatype:
rosrun oculus_sonar_driver reprocess_oculus_raw_data reprocess_oculus_raw_data/raw_data:=oculus/raw_data reprocess_oculus_raw_data/sonar_image:=sonar_image
Convert marine_acoustic_msgs/ProjectedSonarImage to sensor_msgs/Image
rosrun sonar_image_proc draw_sonar_node
Convert marine_acoustic_msgs/ProjectedSonarImage to sensor_msgs/PointCloud2
rosrun sonar_image_proc sonar_pointcloud.py _frame_id:="map"
You can use Rviz to show the sonar image or point cloud:

## Teams

apl_msgs, decoding_image_transport-feature-h265 ROS packages

sudo apt-get install ros-noetic-genmypy
