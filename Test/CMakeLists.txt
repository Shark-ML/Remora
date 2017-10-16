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
	Boost 1.48.0 REQUIRED COMPONENTS
	unit_test_framework
)

macro(REMORA_ADD_TEST SRC NAME)

	if( OPT_LOG_TEST_OUTPUT )
		set( XML_LOGGING_COMMAND_LINE_ARGS "--log_level=test_suite --log_format=XML --log_sink=${NAME}_Log.xml --report_level=no" )
	endif( OPT_LOG_TEST_OUTPUT )

	# Create the test executable
	# Add the test
	add_executable( ${NAME} ${SRC} )
	target_link_libraries( ${NAME} ${Boost_LIBRARIES} ${CBLAS_LIBRARIES})
	if(ENABLE_CLBLAST)
		target_link_libraries( ${NAME} clblast)
	endif()
	add_test( ${NAME} ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/${NAME} ${XML_LOGGING_COMMAND_LINE_ARGS} )
	set_property(TARGET ${NAME} PROPERTY CXX_STANDARD 11)
	set_property(TARGET ${NAME} PROPERTY CXX_STANDARD_REQUIRED ON)
	set_target_properties(${NAME} PROPERTIES FOLDER "Tests")
endmacro()

#CPU Tests
remora_add_test( iterators.cpp Iterators)
remora_add_test( vector_assign.cpp Vector_Assign)
remora_add_test( matrix_assign.cpp Matrix_Assign)
#remora_add_test( triangular_matrix.cpp Triangular_Matrix)
#remora_add_test( compressed_vector.cpp Compressed_Vector)
#remora_add_test( compressed_matrix.cpp Compressed_Matrix)
remora_add_test( dense.cpp Dense)
remora_add_test( matrix_proxy.cpp Matrix_Proxy)
remora_add_test( vector_expression.cpp Vector_Expression)
remora_add_test( matrix_expression.cpp Matrix_Expression)
remora_add_test( prod.cpp Prod)
remora_add_test( syrk.cpp Syrk)
remora_add_test( getrf.cpp Getrf)
remora_add_test( conv2d.cpp Conv2D)
#remora_add_test( expression_optimizer.cpp Expression_Optimizer)
remora_add_test( triangular_prod.cpp Triangular_Prod)
remora_add_test( triangular_solve.cpp Triangular_Solve)
remora_add_test( symm_solve.cpp Symm_Solve)
remora_add_test( general_solve.cpp General_Solve)
remora_add_test( cholesky.cpp Cholesky)
remora_add_test( eigensymm.cpp Eigensymm)
remora_add_test( random.cpp Random)

#GPU tests
if(ENABLE_GPU)
remora_add_test( gpu_vector_assign.cpp GPU_Vector_Assign)
remora_add_test( gpu_matrix_assign.cpp GPU_Matrix_Assign)
remora_add_test( gpu_dense.cpp GPU_Dense)
remora_add_test( gpu_copy.cpp GPU_Copy)
remora_add_test( gpu_vector_expression.cpp GPU_Vector_Expression)
remora_add_test( gpu_matrix_expression.cpp GPU_Matrix_Expression)
remora_add_test( gpu_prod.cpp GPU_Prod)
remora_add_test( gpu_syrk.cpp GPU_Syrk)
remora_add_test( gpu_triangular_prod.cpp GPU_Triangular_Prod)
remora_add_test( gpu_triangular_solve.cpp GPU_Triangular_Solve)
remora_add_test( gpu_cholesky.cpp GPU_Cholesky)
remora_add_test( gpu_random.cpp GPU_Random)
endif()


if(ENABLE_CLBLAST)
    remora_add_test( gpu_conv2d.cpp GPU_Conv2D)
endif()