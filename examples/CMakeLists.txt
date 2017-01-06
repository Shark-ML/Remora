add_custom_target(remoraexamples)
set(EXAMPLES_CPP_DIR ${CMAKE_CURRENT_BINARY_DIR})
set(TPP_FILES "")

macro( REMORA_ADD_EXAMPLE TUT_SRC NAME)
	# Generate cpp file from tpp file
	add_custom_command(
	OUTPUT ${EXAMPLES_CPP_DIR}/${TUT_SRC}.cpp
	COMMAND ${CMAKE_COMMAND}
	ARGS
		-DTUT_TPP:Filepath=${CMAKE_CURRENT_SOURCE_DIR}/${TUT_SRC}.tpp
		-DTUT_CPP:Filepath=${EXAMPLES_CPP_DIR}/${TUT_SRC}.cpp
		-P ${CMAKE_CURRENT_SOURCE_DIR}/createExampleSource.cmake
	MAIN_DEPENDENCY
		${CMAKE_CURRENT_SOURCE_DIR}/${TUT_SRC}.tpp
	COMMENT "creating ${EXAMPLES_CPP_DIR}/${TUT_SRC}.cpp from ${CMAKE_CURRENT_SOURCE_DIR}/${TUT_SRC}.tpp"
	)

	list(APPEND TPP_FILES ${EXAMPLES_CPP_DIR}/${TUT_SRC}.cpp)

	if( BUILD_EXAMPLES )
		add_executable(${NAME} ${EXAMPLES_CPP_DIR}/${TUT_SRC}.cpp)
		set_property(TARGET ${NAME} PROPERTY CXX_STANDARD 11)
		set_property(TARGET ${NAME} PROPERTY CXX_STANDARD_REQUIRED ON)

		# Add the dependencies to the target examples
		add_dependencies(remoraexamples ${NAME})

		set_target_properties( ${NAME} PROPERTIES FOLDER "examples")

		# Installation
		install(TARGETS ${NAME}
			   RUNTIME DESTINATION ${SHARK_INSTALL_EXAMPLE_DIR}/
			   COMPONENT EXAMPLES)

		install(FILES ${EXAMPLES_CPP_DIR}/${TUT_SRC}.cpp
			   DESTINATION ${SHARK_INSTALL_EXAMPLE_DIR}/)

		source_group(${COMPONENT} FILES ${TUT_SRC})
	endif()
endmacro()

REMORA_ADD_EXAMPLE(linear_regression LinearRegression)