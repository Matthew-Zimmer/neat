target_sources(neat_test PUBLIC 
    genetic.cpp
    neural_network.cpp)

find_package(ordeal REQUIRED)
target_link_libraries(neat_test PUBLIC ordeal neat)