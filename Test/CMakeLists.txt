#####################################################################
#       Remora BLAS Library                                         #
#       Setup for unit testing                                      #
#       Test invocation: CTest                                      #
#       Test implementation: Boost UTF                              #
#####################################################################

#####################################################################
#       Get Boost Unit test                                         #
#####################################################################
find_package(
	Boost 1.65.0 REQUIRED COMPONENTS
	unit_test_framework
)

macro(REMORA_ADD_TEST SRC NAME)

	# Create the test executable
	# Add the test
	add_executable( ${NAME} ${SRC} )
	target_link_libraries( ${NAME} ${Boost_LIBRARIES} ${CBLAS_LIBRARIES})
	if(ENABLE_CLBLAST)
		target_link_libraries( ${NAME} clblast)
	endif()
	add_test( ${NAME} ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/${NAME} )
	set_property(TARGET ${NAME} PROPERTY CXX_STANDARD 14)
	set_property(TARGET ${NAME} PROPERTY CXX_STANDARD_REQUIRED ON)
	set_target_properties(${NAME} PROPERTIES FOLDER "Tests")
endmacro()


macro(REMORA_ADD_HIP_TEST SRC NAME)

	# Create the test executable
	# Add the test
	set_source_files_properties(${SRC} PROPERTIES HIP_SOURCE_PROPERTY_FORMAT 1)
	
	hip_add_executable(${NAME} ${SRC})
	target_link_libraries( ${NAME} ${Boost_LIBRARIES} ${CBLAS_LIBRARIES} ${CUDA_CUBLAS_LIBRARIES})
	add_test( ${NAME} ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/${NAME})
	set_property(TARGET ${NAME} PROPERTY CXX_STANDARD 14)
	set_property(TARGET ${NAME} PROPERTY CXX_STANDARD_REQUIRED ON)
	set_target_properties(${NAME} PROPERTIES FOLDER "Tests")
endmacro()

#CPU Tests
remora_add_test( axis.cpp Axis)
remora_add_test( shape.cpp Shape)
remora_add_test( dense.cpp Dense)
remora_add_test( proxy.cpp Proxy)
remora_add_test( kernels_assign.cpp Kernels_Assign)
remora_add_test( expressions.cpp Expressions)


#remora_add_test( matrix_expression.cpp Matrix_Expression)
#remora_add_test( expression_optimizer.cpp Expression_Optimizer)

