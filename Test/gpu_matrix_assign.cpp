#define BOOST_TEST_MODULE Remora_GPU_MatrixAssign
#define BOOST_COMPUTE_DEBUG_KERNEL_COMPILATION
#include <boost/test/unit_test.hpp>
#include <boost/test/floating_point_comparison.hpp>

#include <remora/kernels/matrix_assign.hpp>
#include <remora/vector.hpp>
#include <remora/matrix.hpp>
#include <remora/matrix_proxy.hpp>//FIXME: should be unneeded
#include <remora/matrix_expression.hpp>// for copy
#include <remora/vector_expression.hpp>

#include <iostream>
using namespace remora;

template<class M1, class M2>
void checkMatrixEqual(M1 const& m1_gpu, M2 const& m2_gpu){
	BOOST_REQUIRE_EQUAL(m1_gpu.size1(),m2_gpu.size1());
	BOOST_REQUIRE_EQUAL(m1_gpu.size2(),m2_gpu.size2());
	
	matrix<float> m1 = copy_to_cpu(m1_gpu);
	matrix<float> m2 = copy_to_cpu(m2_gpu);
	for(std::size_t i = 0; i != m2.size1(); ++i){
		for(std::size_t j = 0; j != m2.size2(); ++j){
			BOOST_CHECK_EQUAL(m1(i,j),m2(i,j));
		}
	}
}

BOOST_AUTO_TEST_SUITE (Remora_gpu_matrix_assign)

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
	matrix<float, row_major, gpu_tag> source = copy_to_gpu(source_cpu);
	matrix<float, column_major, gpu_tag> source_cm = copy_to_gpu(source_cpu);
	matrix<float, row_major, gpu_tag> result_add = copy_to_gpu(result_add_cpu);
	matrix<float, row_major, gpu_tag> result_add_scalar = copy_to_gpu(result_add_scalar_cpu);
	{
		std::cout<<"testing direct assignment row-row"<<std::endl;
		matrix<float, row_major, gpu_tag> target = copy_to_gpu(target_cpu);
		kernels::assign(target,source);
		checkMatrixEqual(target,source);
	}
	{
		std::cout<<"testing functor assignment row-row"<<std::endl;
		matrix<float, row_major, gpu_tag> target = copy_to_gpu(target_cpu);
		kernels::assign(target,source, device_traits<gpu_tag>::add<float>());
		checkMatrixEqual(target,result_add);
	}
	{
		std::cout<<"testing direct assignment row-column"<<std::endl;
		matrix<float, row_major, gpu_tag> target = copy_to_gpu(target_cpu);
		kernels::assign(target,source_cm);
		checkMatrixEqual(target,source_cm);
	}
	{
		std::cout<<"testing functor assignment row-column"<<std::endl;
		matrix<float, row_major, gpu_tag> target = copy_to_gpu(target_cpu);
		kernels::assign(target,source_cm, device_traits<gpu_tag>::add<float>());
		checkMatrixEqual(target,result_add);
	}
	{
		std::cout<<"testing functor scalar assignment"<<std::endl;
		matrix<float, row_major, gpu_tag> target = copy_to_gpu(target_cpu);
		kernels::assign<device_traits<gpu_tag>::add<float> >(target,scalar);
		target.queue().finish();
		checkMatrixEqual(target,result_add_scalar);
	}
	
}

BOOST_AUTO_TEST_SUITE_END()
