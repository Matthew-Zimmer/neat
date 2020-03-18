target_include_directories(neat INTERFACE 
	$<BUILD_INTERFACE:${CMAKE_CURRENT_LIST_DIR}> 
	$<INSTALL_INTERFACE:include/neat>)

target_include_directories(neat_test PUBLIC 
	$<BUILD_INTERFACE:${CMAKE_CURRENT_LIST_DIR}>)

set(headers ml.hpp)

install(FILES ${headers} DESTINATION include/neat)
