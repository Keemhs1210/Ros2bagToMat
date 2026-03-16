cmake_minimum_required(VERSION 3.8)
project(ros2_msg)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# find dependencies
find_package(ament_cmake REQUIRED)
find_package(std_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(rosidl_default_generators REQUIRED)

set(msg_files
  "msg/LOGBYTE0.msg"
  "msg/LOGBYTE1.msg"
  "msg/LOGBYTE2.msg"
  "msg/LOGBYTE3.msg"
  "msg/MELaneAddinfo1.msg"
  "msg/MELaneAdditionalData1.msg"
  "msg/MELaneAdditionalData2.msg"
  "msg/MELaneAdditionalData3.msg"
  "msg/MELeftLaneA.msg"
  "msg/MELeftLaneB.msg"
  "msg/MERightLaneA.msg"
  "msg/MERightLaneB.msg"
  "msg/ObstacleData1A.msg"
  "msg/ObstacleData1B.msg"
  "msg/ObstacleData1C.msg"
  "msg/Track1.msg"
  "msg/FusionVisionLanePP.msg"
  "msg/FusionVisionLaneArrayPP.msg"
  "msg/FusionTrackv3.msg"
  "msg/FusionTrackArrayv3.msg"
  "msg/Target.msg"
  "msg/RoadBarrier.msg"
  "msg/Timestamp.msg"
)

rosidl_generate_interfaces(${PROJECT_NAME}
 ${msg_files}
 DEPENDENCIES geometry_msgs std_msgs 
)

ament_package()
