#define BOOST_TEST_MODULE Remora_Vector_Proxy
#include <boost/test/unit_test.hpp>
#include <boost/test/floating_point_comparison.hpp>

#include <remora/proxy_expressions.hpp>
#include <remora/device_copy.hpp>
#include <remora/dense.hpp>

#include <iostream>
using namespace remora;

template<class Operation, class Result>
void checkDenseVectorEquality(
	Operation op_gpu, Result const& result
){
	BOOST_REQUIRE_EQUAL(op_gpu.size(), result.size());
	
	//test copy to cpu, this tests the buffer
	vector<float> op = copy_to_cpu(op_gpu);
	for(std::size_t i = 0; i != op.size(); ++i){
		BOOST_CHECK_CLOSE(result(i), op(i),1.e-8);
	}
	
	//test iterators
	BOOST_REQUIRE_EQUAL(op_gpu.end() - op_gpu.begin(), op.size());
	vector<float, gpu_tag> opcopy_gpu(op.size());
	boost::compute::copy(op_gpu.begin(),op_gpu.end(),opcopy_gpu.begin());
	vector<float> opcopy = copy_to_cpu(opcopy_gpu);
	for(std::size_t i = 0; i != result.size(); ++i){
		BOOST_CHECK_CLOSE(result(i), opcopy(i),1.e-8);
	}
}

std::size_t Dimensions = 20;
struct VectorProxyFixture
{
	vector<float, gpu_tag> denseData;
	vector<float> denseData_cpu;
	
	VectorProxyFixture():denseData_cpu(Dimensions){
		for(std::size_t i = 0; i!= Dimensions;++i){
			denseData_cpu(i) = i+5;
		}
		denseData = copy_to_gpu(denseData_cpu);
	}
};

BOOST_FIXTURE_TEST_SUITE (Remora_vector_proxy, VectorProxyFixture);

BOOST_AUTO_TEST_CASE( Remora_Dense_Subrange ){
	//all possible combinations of ranges on the data vector
	for(std::size_t rangeEnd=0;rangeEnd!= Dimensions;++rangeEnd){
		for(std::size_t rangeBegin =0;rangeBegin <=rangeEnd;++rangeBegin){//<= for 0 range
			
			//first check that the subrange has the right values
			std::size_t size=rangeEnd-rangeBegin;
			vector<float> vTest(size);
			for(std::size_t i = 0; i != size; ++i){
				vTest(i) = denseData_cpu(i+rangeBegin);
			}
			checkDenseVectorEquality(subrange(denseData,rangeBegin,rangeEnd),vTest);
			
			//now test whether we can assign to a range like this.
			vector<float, gpu_tag> newData(Dimensions,1.0);
			auto rangeTest = subrange(newData,rangeBegin,rangeEnd);
			noalias(rangeTest) = subrange(denseData,rangeBegin,rangeEnd);
			//check that the assignment has been carried out correctly
			checkDenseVectorEquality(rangeTest,vTest);

			//check that after assignment all elements outside the range are still intact
			vector<float> data = copy_to_cpu(newData);
			for(std::size_t i = 0; i != rangeBegin; ++i){
				BOOST_CHECK_EQUAL(data(i),1.0);
			}
			for(std::size_t i = rangeBegin; i != rangeEnd; ++i){
				BOOST_CHECK_EQUAL(data(i),vTest(i-rangeBegin));
			}
			for(std::size_t i = rangeEnd; i != Dimensions; ++i){
				BOOST_CHECK_EQUAL(data(i),1.0);
			}
		}
	}
}

BOOST_AUTO_TEST_SUITE_END();