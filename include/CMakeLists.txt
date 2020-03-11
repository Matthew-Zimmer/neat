target_include_directories(neat PUBLIC 
	$<BUILD_INTERFACE:${CMAKE_CURRENT_LIST_DIR}> 
	$<INSTALL_INTERFACE:include/neat>)

target_include_directories(neat_test PUBLIC 
	$<BUILD_INTERFACE:${CMAKE_CURRENT_LIST_DIR}>)

set(headers )

install(FILES ${headers} DESTINATION include/neat)
