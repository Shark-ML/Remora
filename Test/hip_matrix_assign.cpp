#define BOOST_TEST_MODULE Remora_HIP_MatrixAssign
#include <boost/test/unit_test.hpp>
#include <boost/test/floating_point_comparison.hpp>

#include <remora/kernels/matrix_assign.hpp>
#include <remora/dense.hpp>
#include <remora/device_copy.hpp>

#include <iostream>
using namespace remora;

template<class M1, class M2>
void checkMatrixEqual(M1 const& m1_opencl, M2 const& m2_opencl){
	BOOST_REQUIRE_EQUAL(m1_opencl.size1(),m2_opencl.size1());
	BOOST_REQUIRE_EQUAL(m1_opencl.size2(),m2_opencl.size2());
	
	matrix<float> m1 = copy_to_cpu(m1_opencl);
	matrix<float> m2 = copy_to_cpu(m2_opencl);
	for(std::size_t i = 0; i != m2.size1(); ++i){
		for(std::size_t j = 0; j != m2.size2(); ++j){
			BOOST_CHECK_EQUAL(m1(i,j),m2(i,j));
		}
	}
}

BOOST_AUTO_TEST_SUITE (Remora_opencl_matrix_assign)

BOOST_AUTO_TEST_CASE( Remora_Matrix_Assign_Dense ){
	std::cout<<"testing dense-dense assignment"<<std::endl;
	matrix<float> source_cpu(100,237);
	matrix<float> target_cpu(100,237);
	matrix<float> result_add_cpu(100,237);
	matrix<float> result_add_scalar_cpu(100,237);
	float scalar = 10;
	for(std::size_t i = 0; i != 100; ++i){
		for(std::size_t j = 0; j != 237; ++j){
			source_cpu(i,j) = 2*i+1+0.3*j;
			target_cpu(i,j) = 3*i+2+0.3*j;
			result_add_cpu(i,j) = source_cpu(i,j) + target_cpu(i,j);
			result_add_scalar_cpu(i,j) = target_cpu(i,j) + scalar;
		}
	}
	matrix<float, row_major, hip_tag> source = copy_to_device(source_cpu, hip_tag());
	matrix<float, column_major, hip_tag> source_cm = copy_to_device(source_cpu, hip_tag());
	matrix<float, row_major, hip_tag> result_add = copy_to_device(result_add_cpu, hip_tag());
	matrix<float, row_major, hip_tag> result_add_scalar = copy_to_device(result_add_scalar_cpu, hip_tag());
	{
		std::cout<<"testing direct assignment row-row"<<std::endl;
		matrix<float, row_major, hip_tag> target = copy_to_device(target_cpu, hip_tag());
		kernels::assign(target,source);
		checkMatrixEqual(target,source);
	}
	{
		std::cout<<"testing functor assignment row-row"<<std::endl;
		matrix<float, row_major, hip_tag> target = copy_to_device(target_cpu, hip_tag());
		kernels::assign(target,source, device_traits<hip_tag>::add<float>());
		checkMatrixEqual(target,result_add);
	}
	{
		std::cout<<"testing direct assignment row-column"<<std::endl;
		matrix<float, row_major, hip_tag> target = copy_to_device(target_cpu, hip_tag());
		kernels::assign(target,source_cm);
		checkMatrixEqual(target,source_cm);
	}
	{
		std::cout<<"testing functor assignment row-column"<<std::endl;
		matrix<float, row_major, hip_tag> target = copy_to_device(target_cpu, hip_tag());
		kernels::assign(target,source_cm, device_traits<hip_tag>::add<float>());
		checkMatrixEqual(target,result_add);
	}
	{
		std::cout<<"testing functor scalar assignment"<<std::endl;
		matrix<float, row_major, hip_tag> target = copy_to_device(target_cpu, hip_tag());
		kernels::assign<device_traits<hip_tag>::add<float> >(target,scalar);
		checkMatrixEqual(target,result_add_scalar);
	}
	
}

BOOST_AUTO_TEST_SUITE_END()
