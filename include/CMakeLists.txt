target_include_directories(neat PUBLIC 
	$<BUILD_INTERFACE:${CMAKE_CURRENT_LIST_DIR}> 
	$<INSTALL_INTERFACE:include/neat>)

set(headers )

install(FILES ${headers} DESTINATION include/neat)
