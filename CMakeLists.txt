cmake_minimum_required(VERSION 3.1)
project(voting_registration)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fopenmp -Wall")

include_directories(.)

find_package(Eigen3 3.3 REQUIRED)
find_package(PCL 1.10 REQUIRED)
#find_package(teaserpp REQUIRED)

#include_directories(${teaserpp_INCLUDE_DIR})
#set(TEASER_LIBRARIES
#        teaserpp::teaser_resgistration
#        teaserpp::teaser_io
#        teaserpp::teaser_features
#        )

include_directories(${PCL_INCLUDE_DIRS})
link_directories(${PCL_LIBRARY_DIRS})
add_definitions(${PCL_DEFINITIONS})

add_library(tools tools.cpp)

add_executable(${PROJECT_NAME} voting_registration_main.cpp)
target_link_libraries(${PROJECT_NAME} tools ${PCL_LIBRARIES})

#add_executable(reg_SV_random_Pss v_reg_SV_rand_Pss.cpp reg_functions.hpp)
#target_link_libraries(reg_SV_random_Pss ${PCL_LIBRARIES})

#add_executable(reg_SV_single_seed v_reg_SV_single_seed.cpp reg_functions.hpp)
#target_link_libraries(reg_SV_single_seed ${PCL_LIBRARIES})

#add_executable(reg_OCT_all_Pss v_reg_OCT_all_Pss.cpp reg_functions.hpp)
#target_link_libraries(reg_OCT_all_Pss ${PCL_LIBRARIES})

#add_executable(cloud_analysis cloud_comparison.cpp reg_functions.hpp)
#target_link_libraries(cloud_analysis ${PCL_LIBRARIES})

#add_executable(reg_EA_all_Pss v_reg_EA_all_Pss.cpp reg_functions.hpp)
#target_link_libraries(reg_EA_all_Pss ${PCL_LIBRARIES})

#add_executable(subset_analysis subset_size_analysis.cpp reg_functions.hpp)
#target_link_libraries(subset_analysis ${PCL_LIBRARIES})

#add_executable(teaser_tst teaser_pp.cpp reg_functions.hpp)
#target_link_libraries(teaser_tst Eigen3::Eigen ${TEASER_LIBRARIES} ${PCL_LIBRARIES})

#file(COPY /home/luis/Documents/Research/Datasets/currently_used_datasets/PLY/
#        DESTINATION ./ply_data/
#        FILES_MATCHING
#        PATTERN *.ply)

file(COPY /home/luis/Documents/Research/Datasets/currently_used_datasets/PCD/
        DESTINATION ./pcd_data/
        FILES_MATCHING
        PATTERN *.pcd)

#file(COPY /home/luis/Documents/Research/Datasets/3dmatch/test/7-scenes-redkitchen/fragments/pcd/
#        DESTINATION ./pcd_data/
#        FILES_MATCHING
#        PATTERN *.pcd)