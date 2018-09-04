#define BOOST_TEST_MODULE Remora_HIP_Syrk
#include <boost/test/unit_test.hpp>
#include <boost/test/floating_point_comparison.hpp>
#include <boost/mpl/list.hpp>


#define BOOST_COMPUTE_DEBUG_KERNEL_COMPILATION
#include <remora/kernels/syrk.hpp>
#include <remora/dense.hpp>
#include <remora/device_copy.hpp>
#include <remora/matrix_expression.hpp>


#include <iostream>
using namespace remora;

template<class M, class Result>
void checkSyrk(M const& arg_opencl, Result const& result_opencl,double init, double alpha, bool upper){
	BOOST_REQUIRE_EQUAL(arg_opencl.size1(), result_opencl.size1());
	BOOST_REQUIRE_EQUAL(result_opencl.size1(), result_opencl.size2());
	
	matrix<float> arg = copy_to_cpu(arg_opencl);
	matrix<float> result = copy_to_cpu(result_opencl);
	
	if(upper){
		for(std::size_t i = 0; i != result.size1(); ++i) {
			for(std::size_t j = 0; j != result.size2(); ++j) {
				if(j < i){
					BOOST_CHECK_CLOSE(result(i,j),init, 1.e-4);
				}else{
					double test_result = alpha*inner_prod(row(arg,i),row(arg,j))+init;
					BOOST_CHECK_CLOSE(result(i,j), test_result, 1.e-4);
				}
			}
		}
	}else{
		for(std::size_t i = 0; i != result.size1(); ++i) {
			for(std::size_t j = 0; j != result.size2(); ++j) {
				if(j > i){
					BOOST_CHECK_CLOSE(result(i,j),init, 1.e-4);
				}else{
					double test_result = alpha*inner_prod(row(arg,i),row(arg,j))+init;
					BOOST_CHECK_CLOSE(result(i,j), test_result, 1.e-4);
				}
			}
		}
	}
}

BOOST_AUTO_TEST_SUITE (Remora_HIP_Syrk)

template<class Orientation>
void syrk_test(Orientation) {
	std::size_t dims = 936;//chosen as not to be a multiple of the block size
	std::size_t K = 1039;

	//rhs
	matrix<float, row_major> arg_cpu(dims, K, 1.0);
	for(std::size_t i = 0; i != dims; ++i) {
		for(std::size_t j = 0; j != K; ++j) {
			arg_cpu(i, j) = (1.0/ dims) * i + 0.2/K * j + 1;
		}
	}
	
	matrix<float,row_major, hip_tag> argrm = copy_to_device(arg_cpu, hip_tag());
	matrix<float,column_major, hip_tag> argcm = copy_to_device(arg_cpu, hip_tag());

	std::cout << "\nchecking syrk V+=AA^T" << std::endl;
	{
		std::cout<<"row major A, lower V"<<std::endl;
		matrix<float, Orientation, hip_tag> result(dims,dims,3.0);
		kernels::syrk<false>(argrm,result, 2.0);
		checkSyrk(argrm,result, 3.0, 2.0,false);
	}
	{
		std::cout<<"row major A, upper V"<<std::endl;
		matrix<float, Orientation, hip_tag> result(dims,dims,3.0);
		kernels::syrk<true>(argrm,result, 2.0);
		checkSyrk(argrm,result, 3.0, 2.0,true);
	}
	{
		std::cout<<"column major A, lower V"<<std::endl;
		matrix<float, Orientation, hip_tag> result(dims,dims,3.0);
		kernels::syrk<false>(argcm,result, 2.0);
		checkSyrk(argrm,result, 3.0, 2.0,false);
	}
	{
		std::cout<<"column major A, upper V"<<std::endl;
		matrix<float, Orientation, hip_tag> result(dims,dims,3.0);
		kernels::syrk<true>(argcm,result, 2.0);
		checkSyrk(argrm,result, 3.0, 2.0,true);
	}
}

BOOST_AUTO_TEST_CASE(HIP_syrk){
	syrk_test(row_major());
	syrk_test(column_major());
}

BOOST_AUTO_TEST_SUITE_END()
