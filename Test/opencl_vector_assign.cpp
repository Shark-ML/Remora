#define BOOST_TEST_MODULE Remora_OPENCL_VectorAssign
#include <boost/test/unit_test.hpp>
#include <boost/test/floating_point_comparison.hpp>

#include <remora/kernels/vector_assign.hpp>
#include <remora/dense.hpp>
#include <remora/device_copy.hpp>

#include <iostream>
using namespace remora;

template<class V1, class V2>
void checkVectorEqual(V1 const& v1_opencl, V2 const& v2_opencl){
	BOOST_REQUIRE_EQUAL(v1_opencl.size(),v2_opencl.size());
	
	vector<unsigned int> v1 = copy_to_cpu(v1_opencl);
	vector<unsigned int> v2 = copy_to_cpu(v2_opencl);
	for(std::size_t i = 0; i != v2.size(); ++i){
		BOOST_CHECK_EQUAL(v1(i),v2(i));
	}
}

BOOST_AUTO_TEST_SUITE (Remora_opencl_vector_assign)

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
	vector<unsigned int, opencl_tag> source = copy_to_opencl(source_cpu);
	vector<unsigned int, opencl_tag> result_add = copy_to_opencl(result_add_cpu);
	vector<unsigned int, opencl_tag> result_add_scalar = copy_to_opencl(result_add_scalar_cpu);
	{
		std::cout<<"testing direct assignment"<<std::endl;
		vector<unsigned int, opencl_tag> target = copy_to_opencl(target_cpu);
		kernels::assign(target,source);
		checkVectorEqual(target,source);
	}
	{
		std::cout<<"testing functor assignment"<<std::endl;
		vector<unsigned int, opencl_tag> target = copy_to_opencl(target_cpu);
		kernels::assign(target,source, device_traits<opencl_tag>::add<unsigned int>());
		checkVectorEqual(target,result_add);
	}
	{
		std::cout<<"testing functor scalar assignment"<<std::endl;
		vector<unsigned int, opencl_tag> target = copy_to_opencl(target_cpu);
		kernels::assign<device_traits<opencl_tag>::add<unsigned int> >(target,scalar);
		checkVectorEqual(target,result_add_scalar);
	}
	
}

BOOST_AUTO_TEST_SUITE_END()
