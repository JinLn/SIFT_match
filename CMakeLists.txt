cmake_minimum_required(VERSION 2.8)
project(siftmatch)

IF(NOT CMAKE_BUILD_TYPE)
  SET(CMAKE_BUILD_TYPE Release)
ENDIF()

find_package(OpenCV 3)
if(NOT OpenCV_FOUND)
   find_package(OpenCV 2.4.3 QUIET)
   if(NOT OpenCV_FOUND)
      message(FATAL_ERROR "OpenCV > 2.4.3 not found.")
   endif()
endif()

MESSAGE("OPENCV VERSION:")
MESSAGE(${OpenCV_VERSION})

set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/lib)

include_directories(
    ${OpenCV_INCLUDE_DIRS}
    ${PROJECT_SOURCE_DIR}
    )
message(${OpenCV_INCLUDE_DIRS})


#add_executable(${PROJECT_NAME} "main.cpp")
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/)
#add_executable(sift sift.cpp)
#target_link_libraries(sift ${OpenCV_LIBS} )

add_executable(main main.cpp)
target_link_libraries(main ${OpenCV_LIBS} )
