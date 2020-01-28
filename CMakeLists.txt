cmake_minimum_required(VERSION 3.13)

project(neat)

add_library(neat "")

add_subdirectory(include)
add_subdirectory(src)

target_compile_features(neat PUBLIC cxx_std_17)

#Set(CMAKE_CXX_FLAGS "-Wall" "-Wextra")

install(TARGETS neat EXPORT neat DESTINATION lib/neat)
install(EXPORT neat DESTINATION lib/)
install(FILES neat-config.cmake DESTINATION lib/neat)


#libs
#find_Package(LIB REQUIRED)
#target_link_libraries(neat PUBLIC LIB)