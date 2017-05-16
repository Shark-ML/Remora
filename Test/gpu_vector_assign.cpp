#define BOOST_TEST_MODULE Remora_GPU_VectorAssign
#include <boost/test/unit_test.hpp>
#include <boost/test/floating_point_comparison.hpp>

#include <remora/kernels/vector_assign.hpp>
#include <remora/vector.hpp>
#include <remora/vector_expression.hpp>

#include <iostream>
using namespace remora;

template<class V1, class V2>
void checkVectorEqual(V1 const& v1_gpu, V2 const& v2_gpu){
	BOOST_REQUIRE_EQUAL(v1_gpu.size(),v2_gpu.size());
	
	vector<unsigned int> v1 = copy_to_cpu(v1_gpu);
	vector<unsigned int> v2 = copy_to_cpu(v2_gpu);
	for(std::size_t i = 0; i != v2.size(); ++i){
		BOOST_CHECK_EQUAL(v1(i),v2(i));
	}
}

BOOST_AUTO_TEST_SUITE (Remora_gpu_vector_assign)

BOOST_AUTO_TEST_CASE( Remora_Vector_Assign_Dense ){
	std::cout<<"testing dense-dense assignment"<<std::endl;
	vector<unsigned int> source_cpu(1000);
	vector<unsigned int> target_cpu(1000);
	vector<unsigned int> result_add_cpu(1000);
	vector<unsigned int> result_add_scalar_cpu(1000);
	unsigned int scalar = 10;
	for(std::size_t i = 0; i != 1000; ++i){
		source_cpu(i) = 2*i+1;
		target_cpu(i) = 3*i+2;
		result_add_cpu(i) = source_cpu(i) + target_cpu(i);
		result_add_scalar_cpu(i) = target_cpu(i) + scalar;
	}
	vector<unsigned int, gpu_tag> source = copy_to_gpu(source_cpu);
	vector<unsigned int, gpu_tag> result_add = copy_to_gpu(result_add_cpu);
	vector<unsigned int, gpu_tag> result_add_scalar = copy_to_gpu(result_add_scalar_cpu);
	{
		std::cout<<"testing direct assignment"<<std::endl;
		vector<unsigned int, gpu_tag> target = copy_to_gpu(target_cpu);
		kernels::assign(target,source);
		checkVectorEqual(target,source);
	}
	{
		std::cout<<"testing functor assignment"<<std::endl;
		vector<unsigned int, gpu_tag> target = copy_to_gpu(target_cpu);
		kernels::assign<device_traits<gpu_tag>::add<unsigned int> >(target,source);
		checkVectorEqual(target,result_add);
	}
	{
		std::cout<<"testing functor scalar assignment"<<std::endl;
		vector<unsigned int, gpu_tag> target = copy_to_gpu(target_cpu);
		kernels::assign<device_traits<gpu_tag>::add<unsigned int> >(target,scalar);
		checkVectorEqual(target,result_add_scalar);
	}
	
}

BOOST_AUTO_TEST_SUITE_END()
