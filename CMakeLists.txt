cmake_minimum_required(VERSION 3.13)

project(neat)

add_library(neat INTERFACE)
add_executable(neat_test)

add_subdirectory(include)
add_subdirectory(src)
add_subdirectory(test)

target_compile_features(neat INTERFACE cxx_std_17)
target_compile_features(neat_test PUBLIC cxx_std_17)

#set(CMAKE_CXX_FLAGS "-fmax-errors=1")

install(TARGETS neat EXPORT neat DESTINATION lib/neat)
install(EXPORT neat DESTINATION lib/neat)
install(FILES neat-config.cmake DESTINATION lib/neat)

#libs
find_package(random REQUIRED)
target_link_libraries(neat INTERFACE random)