#define BOOST_TEST_MODULE Remora_GPU_COPY
#include <boost/test/unit_test.hpp>
#include <boost/test/floating_point_comparison.hpp>

#include <remora/vector.hpp>
#include <remora/matrix.hpp>
#include <remora/matrix_expression.hpp>
#include <remora/vector_expression.hpp>

#include <iostream>
using namespace remora;


BOOST_AUTO_TEST_SUITE (Remora_gpu_copy)

BOOST_AUTO_TEST_CASE( Remora_Vector_Copy ){
	std::cout<<"testing vector copy to gpu and back"<<std::endl;
	vector<float> source(100);
	for(std::size_t i = 0; i != 100; ++i){
		source(i) = 2*i+1;
	}
	vector<float, gpu_tag> target_gpu = copy_to_gpu(source);
	vector<float> target_cpu = copy_to_cpu(target_gpu);
	
	BOOST_CHECK_SMALL(norm_inf(source - target_cpu), 1.e-10f);	
}
BOOST_AUTO_TEST_CASE( Remora_Vector_Copy_Plus_Assign ){
	std::cout<<"testing vector assignment to gpu and back"<<std::endl;
	vector<float> source(100);
	for(std::size_t i = 0; i != 100; ++i){
		source(i) = 2*i+1;
	}
	vector<float, gpu_tag> target_gpu(100,1.0);
	noalias(target_gpu) += copy_to_gpu(source);
	vector<float> target_cpu(100,-2.0);
	noalias(target_cpu) += copy_to_cpu(target_gpu);
	
	BOOST_CHECK_SMALL(norm_inf(source - target_cpu-1), 1.e-10f);	
}

BOOST_AUTO_TEST_CASE( Remora_Matrix_Copy ){
	std::cout<<"testing matrix copy to gpu and back"<<std::endl;
	matrix<float,row_major> source(32,16);
	for(std::size_t i = 0; i != 32; ++i){
		for(std::size_t j = 0; j != 16; ++j){
			source(i,j) = i*16+j;
		}
	}
	matrix<float,column_major> source_cm  = source;
	//row-major cpu to row-major gpu to row-major cpu
	{
		matrix<float,row_major, gpu_tag> target_gpu = copy_to_gpu(source);
		matrix<float,row_major> target_cpu = copy_to_cpu(target_gpu);
		BOOST_CHECK_SMALL(norm_inf(source - target_cpu), 1.e-10f);
	}
	//row-major-cpu to column-major gpu to column-major cpu
	{
		matrix<float,column_major, gpu_tag> target_gpu = copy_to_gpu(source);
		matrix<float,column_major> target_cpu = copy_to_cpu(target_gpu);
		BOOST_CHECK_SMALL(norm_inf(source - target_cpu), 1.e-10f);
	}
	//column-major-cpu to column-major gpu to row-major cpu
	{
		matrix<float,column_major, gpu_tag> target_gpu = copy_to_gpu(source_cm);
		matrix<float,row_major> target_cpu = copy_to_cpu(target_gpu);
		BOOST_CHECK_SMALL(norm_inf(source - target_cpu), 1.e-10f);
	}
	//column-major-cpu to row-major gpu to column-major cpu
	{
		matrix<float,row_major, gpu_tag> target_gpu = copy_to_gpu(source_cm);
		matrix<float,column_major> target_cpu = copy_to_cpu(target_gpu);
		BOOST_CHECK_SMALL(norm_inf(source - target_cpu), 1.e-10f);
	}
}

BOOST_AUTO_TEST_CASE( Remora_Matrix_Copy_Plus_Assign ){
	std::cout<<"testing matrix assignment to gpu and back"<<std::endl;
	matrix<float,row_major> source(32,16);
	for(std::size_t i = 0; i != 32; ++i){
		for(std::size_t j = 0; j != 16; ++j){
			source(i,j) = i*16+j;
		}
	}
	matrix<float,column_major> source_cm  = source;
	//row-major cpu to row-major gpu to row-major cpu
	{
		matrix<float,row_major, gpu_tag> target_gpu(32,16,1.0);
		noalias(target_gpu) += copy_to_gpu(source);
		matrix<float,row_major> target_cpu(32,16,-2.0);
		noalias(target_cpu) += copy_to_cpu(target_gpu);
		BOOST_CHECK_SMALL(norm_inf(source - target_cpu-1), 1.e-10f);	
	}
	//row-major-cpu to column-major gpu to column-major cpu
	{
		matrix<float,column_major, gpu_tag> target_gpu(32,16,1.0);
		noalias(target_gpu) += copy_to_gpu(source);
		matrix<float,column_major> target_cpu(32,16,-2.0);
		noalias(target_cpu) += copy_to_cpu(target_gpu);
		BOOST_CHECK_SMALL(norm_inf(source - target_cpu-1), 1.e-10f);	
	}
	//column-major-cpu to column-major gpu to row-major cpu
	{
		matrix<float,column_major, gpu_tag> target_gpu(32,16,1.0);
		noalias(target_gpu) += copy_to_gpu(source_cm);
		matrix<float,row_major> target_cpu(32,16,-2.0);
		noalias(target_cpu) += copy_to_cpu(target_gpu);
		BOOST_CHECK_SMALL(norm_inf(source - target_cpu-1), 1.e-10f);	
	}
	//column-major-cpu to row-major gpu to column-major cpu
	{
		matrix<float,row_major, gpu_tag> target_gpu(32,16,1.0);
		noalias(target_gpu) += copy_to_gpu(source_cm);
		matrix<float,column_major> target_cpu(32,16,-2.0);
		noalias(target_cpu) += copy_to_cpu(target_gpu);
		BOOST_CHECK_SMALL(norm_inf(source - target_cpu-1), 1.e-10f);	
	}
	
}

BOOST_AUTO_TEST_SUITE_END()
