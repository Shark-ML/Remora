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

	# Create the test executable
	# Add the test
	add_executable( ${NAME} ${SRC} )
	target_link_libraries( ${NAME} ${Boost_LIBRARIES} ${CBLAS_LIBRARIES})
	if(ENABLE_CLBLAST)
		target_link_libraries( ${NAME} clblast)
	endif()
	add_test( ${NAME} ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/${NAME} )
	set_property(TARGET ${NAME} PROPERTY CXX_STANDARD 11)
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
	set_property(TARGET ${NAME} PROPERTY CXX_STANDARD 11)
	set_property(TARGET ${NAME} PROPERTY CXX_STANDARD_REQUIRED ON)
	set_target_properties(${NAME} PROPERTIES FOLDER "Tests")
endmacro()

#~ #CPU Tests
#~ remora_add_test( iterators.cpp Iterators)
#~ remora_add_test( vector_assign.cpp Vector_Assign)
#~ remora_add_test( matrix_assign.cpp Matrix_Assign)
#~ #remora_add_test( triangular_matrix.cpp Triangular_Matrix)
#~ remora_add_test( dense.cpp Dense)
#~ remora_add_test( sparse.cpp Sparse)
#~ remora_add_test( matrix_proxy.cpp Matrix_Proxy)
#~ remora_add_test( vector_expression.cpp Vector_Expression)
#~ remora_add_test( vector_set_expression.cpp Vector_Set_Expression)
#~ remora_add_test( matrix_expression.cpp Matrix_Expression)
#~ remora_add_test( prod.cpp Prod)
#~ remora_add_test( syrk.cpp Syrk)
#~ remora_add_test( getrf.cpp Getrf)
#~ remora_add_test( conv2d.cpp Conv2D)
#~ #remora_add_test( expression_optimizer.cpp Expression_Optimizer)
#~ remora_add_test( triangular_prod.cpp Triangular_Prod)
#~ remora_add_test( triangular_solve.cpp Triangular_Solve)
#~ remora_add_test( symm_solve.cpp Symm_Solve)
#~ remora_add_test( general_solve.cpp General_Solve)
#~ remora_add_test( cholesky.cpp Cholesky)
#~ remora_add_test( eigensymm.cpp Eigensymm)
#~ remora_add_test( random.cpp Random)

#OpenCL tests
if(ENABLE_OPENCL)
find_package(CUDA)
remora_add_test( opencl_vector_assign.cpp OpenCL_Vector_Assign)
remora_add_test( opencl_matrix_assign.cpp OpenCL_Matrix_Assign)
remora_add_test( opencl_dense.cpp OpenCL_Dense)
remora_add_test( opencl_copy.cpp OpenCL_Copy)
remora_add_test( opencl_vector_expression.cpp OpenCL_Vector_Expression)
remora_add_test( opencl_vector_set_expression.cpp OpenCL_Vector_Set_Expression)
remora_add_test( opencl_matrix_expression.cpp OpenCL_Matrix_Expression)
remora_add_test( opencl_prod.cpp OpenCL_Prod)
remora_add_test( opencl_syrk.cpp OpenCL_Syrk)
remora_add_test( opencl_triangular_prod.cpp OpenCL_Triangular_Prod)
remora_add_test( opencl_triangular_solve.cpp OpenCL_Triangular_Solve)
remora_add_test( opencl_cholesky.cpp OpenCL_Cholesky)
remora_add_test( opencl_random.cpp OpenCL_Random)
endif()

if(ENABLE_HIP)
remora_add_hip_test( hip_vector_assign.cpp HIP_Vector_Assign)
remora_add_hip_test( hip_matrix_assign.cpp HIP_Matrix_Assign)
remora_add_hip_test( hip_dense.cpp HIP_Dense)
remora_add_hip_test( hip_copy.cpp HIP_Copy)
remora_add_hip_test( hip_vector_expression.cpp HIP_Vector_Expression)
remora_add_hip_test( hip_vector_set_expression.cpp HIP_Vector_Set_Expression)
remora_add_hip_test( hip_matrix_expression.cpp HIP_Matrix_Expression)
remora_add_hip_test( hip_prod.cpp HIP_Prod)
remora_add_hip_test( hip_syrk.cpp HIP_Syrk)
remora_add_hip_test( hip_triangular_prod.cpp HIP_Triangular_Prod)
remora_add_hip_test( hip_triangular_solve.cpp HIP_Triangular_Solve)
remora_add_hip_test( hip_cholesky.cpp HIP_Cholesky)
remora_add_hip_test( hip_random.cpp HIP_Random)
endif()

if(ENABLE_CLBLAST)
    remora_add_test( opencl_conv2d.cpp OpenCL_Conv2D)
endif()
