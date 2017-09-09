#define BOOST_TEST_MODULE Remora_GPU_COPY
#include <boost/test/unit_test.hpp>
#include <boost/test/floating_point_comparison.hpp>

#include <iostream>
#include <remora/io.hpp>
#include <remora/device_copy.hpp>
#include <remora/dense.hpp>

#include <iostream>
using namespace remora;

template<class M1, class M2>
void checkMatrixEqual(M1 const& m1, M2 const& m2){
	BOOST_REQUIRE_EQUAL(m1.size1(),m2.size1());
	BOOST_REQUIRE_EQUAL(m1.size2(),m2.size2());
	
	for(std::size_t i = 0; i != m2.size1(); ++i){
		for(std::size_t j = 0; j != m2.size2(); ++j){
			BOOST_CHECK_CLOSE(m1(i,j),m2(i,j), 1.e-4f);
		}
	}
}

template<class V1, class V2>
void checkVectorEqual(V1 const& v1, V2 const& v2){
	BOOST_REQUIRE_EQUAL(v1.size(),v2.size());
	
	for(std::size_t i = 0; i != v2.size(); ++i){
		BOOST_CHECK_CLOSE(v1(i),v2(i), 1.e-4f);
	}
}


BOOST_AUTO_TEST_SUITE (Remora_gpu_copy)

BOOST_AUTO_TEST_CASE( Remora_Vector_Copy ){
	std::cout<<"testing vector copy to gpu and back"<<std::endl;
	vector<float> source(100);
	for(std::size_t i = 0; i != source.size(); ++i){
		source(i) = 2*i+1;
	}
	vector<float, gpu_tag> target_gpu = copy_to_gpu(source);
	vector<float> target_cpu = copy_to_cpu(target_gpu);
	
	checkVectorEqual(source , target_cpu);	
}
BOOST_AUTO_TEST_CASE( Remora_Vector_Copy_Plus_Assign ){
	std::cout<<"testing vector assignment to gpu and back"<<std::endl;
	vector<float> source(100);
	for(std::size_t i = 0; i != source.size(); ++i){
		source(i) = 2*i+1;
	}
	vector<float, gpu_tag> target_gpu(source.size(),1.0);
	noalias(target_gpu) += copy_to_gpu(source);
	vector<float> target_cpu(source.size(),-1.0);
	noalias(target_cpu) += copy_to_cpu(target_gpu);
	checkVectorEqual(source , target_cpu);
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
		checkMatrixEqual(source , target_cpu);
	}
	//row-major-cpu to column-major gpu to column-major cpu
	{
		matrix<float,column_major, gpu_tag> target_gpu = copy_to_gpu(source);
		matrix<float,column_major> target_cpu = copy_to_cpu(target_gpu);
		checkMatrixEqual(source , target_cpu);
	}
	//column-major-cpu to column-major gpu to row-major cpu
	{
		matrix<float,column_major, gpu_tag> target_gpu = copy_to_gpu(source_cm);
		matrix<float,row_major> target_cpu = copy_to_cpu(target_gpu);
		checkMatrixEqual(source , target_cpu);
	}
	//column-major-cpu to row-major gpu to column-major cpu
	{
		matrix<float,row_major, gpu_tag> target_gpu = copy_to_gpu(source_cm);
		matrix<float,column_major> target_cpu = copy_to_cpu(target_gpu);
		checkMatrixEqual(source , target_cpu);
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
		matrix<float,row_major> target_cpu(32,16,-1.0);
		noalias(target_cpu) += copy_to_cpu(target_gpu);
		checkMatrixEqual(source , target_cpu);
	}
	//row-major-cpu to column-major gpu to column-major cpu
	{
		matrix<float,column_major, gpu_tag> target_gpu(32,16,1.0);
		noalias(target_gpu) += copy_to_gpu(source);
		matrix<float,column_major> target_cpu(32,16,-1.0);
		noalias(target_cpu) += copy_to_cpu(target_gpu);
		checkMatrixEqual(source , target_cpu);	
	}
	//column-major-cpu to column-major gpu to row-major cpu
	{
		matrix<float,column_major, gpu_tag> target_gpu(32,16,1.0);
		noalias(target_gpu) += copy_to_gpu(source_cm);
		matrix<float,row_major> target_cpu(32,16,-1.0);
		noalias(target_cpu) += copy_to_cpu(target_gpu);
		checkMatrixEqual(source , target_cpu);
	}
	//column-major-cpu to row-major gpu to column-major cpu
	{
		matrix<float,row_major, gpu_tag> target_gpu(32,16,1.0);
		noalias(target_gpu) += copy_to_gpu(source_cm);
		matrix<float,column_major> target_cpu(32,16,-1.0);
		noalias(target_cpu) += copy_to_cpu(target_gpu);
		checkMatrixEqual(source , target_cpu);
	}
	
}

BOOST_AUTO_TEST_SUITE_END()
