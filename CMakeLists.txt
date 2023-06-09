cmake_minimum_required(VERSION 2.8)
project(patchmatch)

# SETUP A MAKE CLEAN TO CLEAN EVERYTHING
set_directory_properties (PROPERTIES ADDITIONAL_MAKE_CLEAN_FILES 
      "CMakeCache.txt;CMakeFiles;Makefile;cmake_install.cmake")

# DEFAULT BUILD: RELEASE
if (NOT CMAKE_BUILD_TYPE)
   message(STATUS "No build type selected, default to Release")
   set(CMAKE_BUILD_TYPE "Release")
endif()

set(UTILITIES_DIRS ${PROJECT_SOURCE_DIR}/src/util)
set(CORE_DIRS ${PROJECT_SOURCE_DIR}/src/core)
set(THIRDPARTY_DIRS ${PROJECT_SOURCE_DIR}/thirdparty)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -O3")
# set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++14")
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -std=c99 -O3")
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} ${CMAKE_CURRENT_SOURCE_DIR}/cmake)

FIND_PACKAGE( OpenMP )
if(OPENMP_FOUND)
   message("OPENMP FOUND")
   set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
   set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
   set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
endif()

find_package(Eigen3 REQUIRED)
find_package(OpenCV REQUIRED)
if(OPENCV_FOUND)  
  message(STATUS "Found OPENCV")
  message("OpenCV version: " ${OpenCV_VERSION})
endif()
find_package(Boost COMPONENTS system filesystem date_time thread REQUIRED)
find_package(GTSAM REQUIRED)
if(GTSAM_FOUND)  
  message(STATUS "Found GTSAM ${GTSAM_VERSION}")
else()
  message(STATUS "GTSAM Not Found")
endif()

# Sources files
set(patchmatch_SOURCE_FILES
  ${PROJECT_SOURCE_DIR}/src/util/util.cpp
  ${PROJECT_SOURCE_DIR}/src/core/frame.cpp
  ${PROJECT_SOURCE_DIR}/src/core/FEAmatcher.cpp
  ${PROJECT_SOURCE_DIR}/src/core/optimizer.cpp
  ${PROJECT_SOURCE_DIR}/src/core/LMtriangulatefactor.cpp
  ${PROJECT_SOURCE_DIR}/src/core/SSSpointfactor.cpp
  ${PROJECT_SOURCE_DIR}/src/core/SSSpointfactorSF.cpp
  ${PROJECT_SOURCE_DIR}/thirdparty/ORBextractor.cpp
  ${PROJECT_SOURCE_DIR}/thirdparty/img.cc
)

add_library(patchmatch ${patchmatch_SOURCE_FILES})

target_link_libraries(patchmatch gtsam)

include_directories(
	${UTILITIES_DIRS}
	${CORE_DIRS}
	${THIRDPARTY_DIRS}
	${Boost_INCLUDE_DIRS}
	${EIGEN_INCLUDE_DIRS}
	${OpenCV_INCLUDE_DIRS}
	${GTSAM_INCLUDE_DIR} 
	)

link_directories(
	${Boost_LIBRARY_DIRS}
	${GTSAM_LIBRARY_DIRS}
	)



# Executable file
add_executable (test_demo src/main.cc)
target_link_libraries(test_demo patchmatch ${LIBS} ${OpenCV_LIBRARIES} ${Boost_LIBRARIES})

