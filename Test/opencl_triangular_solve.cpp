#define BOOST_TEST_MODULE Remora_OPENCL_Solve_Triangular
#include <boost/test/unit_test.hpp>
#include <boost/test/floating_point_comparison.hpp>

#define BOOST_COMPUTE_DEBUG_KERNEL_COMPILATION
#include <remora/solve.hpp>
#include <remora/dense.hpp>
#include <remora/device_copy.hpp>
#include <remora/matrix_expression.hpp>
#include <remora/vector_expression.hpp>

#include <iostream>
#include <boost/mpl/list.hpp>

using namespace remora;

//simple test which checks for all argument combinations whether they are correctly translated
BOOST_AUTO_TEST_SUITE (OPENCL_Solve_Triangular)


BOOST_AUTO_TEST_CASE( Solve_Vector ){
	std::size_t size = 326;
	
	matrix<float,row_major> A_cpu(size,size,1.0);
	for(std::size_t i = 0; i != size; ++i){
		for(std::size_t j = 0; j <=i; ++j){
			A_cpu(i,j) = 0.1/size*i-0.05/(i+1.0)*j;
			if(i ==j)
				A_cpu(i,j) += 10;
		}
	}
	matrix<float,row_major, opencl_tag> A = copy_to_opencl(A_cpu);
	matrix<float,row_major, opencl_tag> Aupper = trans(A);
	
	vector<float> b_cpu(size);
	for(std::size_t i = 0; i != size; ++i){
		b_cpu(i) = (0.1/size)*i;
	}
	
	vector<float, opencl_tag> b = copy_to_opencl(b_cpu);

	std::cout<<"triangular vector"<<std::endl;
	std::cout<<"left - lower"<<std::endl;
	{
		vector<float, opencl_tag> testResult = solve(A,b, lower(), left());
		vector<float, opencl_tag> result = triangular_prod<lower>(A,testResult);
		float error = norm_inf(result-b);
		BOOST_CHECK_SMALL(error, 1.e-5f);
	}
	std::cout<<"right - lower"<<std::endl;
	{
		vector<float, opencl_tag> testResult = solve(A,b, lower(), right());
		vector<float, opencl_tag> result = triangular_prod<upper>(Aupper,testResult);
		float error = norm_inf(result-b);
		BOOST_CHECK_SMALL(error, 1.e-5f);
	}
	
	std::cout<<"left - unit_lower"<<std::endl;
	{
		vector<float, opencl_tag> testResult = solve(A,b, unit_lower(), left());
		vector<float, opencl_tag> result = triangular_prod<unit_lower>(A,testResult);
		float error = norm_inf(result-b);
		BOOST_CHECK_SMALL(error, 1.e-5f);
	}
	std::cout<<"right - unit_lower"<<std::endl;
	{
		vector<float, opencl_tag> testResult = solve(A,b, unit_lower(), right());
		vector<float, opencl_tag> result = triangular_prod<unit_upper>(Aupper,testResult);
		float error = norm_inf(result-b);
		BOOST_CHECK_SMALL(error, 1.e-5f);
	}
	
	std::cout<<"left - upper"<<std::endl;
	{
		vector<float, opencl_tag> testResult = solve(Aupper,b, upper(), left());
		vector<float, opencl_tag> result = triangular_prod<upper>(Aupper,testResult);
		float error = norm_inf(result-b);
		BOOST_CHECK_SMALL(error, 1.e-5f);
	}
	std::cout<<"right - upper"<<std::endl;
	{
		vector<float, opencl_tag> testResult = solve(Aupper,b, upper(), right());
		vector<float, opencl_tag> result = triangular_prod<lower>(A,testResult);
		float error = norm_inf(result-b);
		BOOST_CHECK_SMALL(error, 1.e-5f);
	}
	
	std::cout<<"left - unit_upper"<<std::endl;
	{
		vector<float, opencl_tag> testResult = solve(Aupper,b, unit_upper(), left());
		vector<float, opencl_tag> result = triangular_prod<unit_upper>(Aupper,testResult);
		float error = norm_inf(result-b);
		BOOST_CHECK_SMALL(error, 1.e-5f);
	}
	std::cout<<"right - unit_upper"<<std::endl;
	{
		vector<float, opencl_tag> testResult = solve(Aupper,b, unit_upper(), right());
		vector<float, opencl_tag> result = triangular_prod<unit_lower>(A,testResult);
		float error = norm_inf(result-b);
		BOOST_CHECK_SMALL(error, 1.e-5f);
	}
}

typedef boost::mpl::list<row_major,column_major> result_orientations;
BOOST_AUTO_TEST_CASE_TEMPLATE(OPENCL_Solve_Matrix, Orientation,result_orientations) {
	std::size_t size = 139;
	std::size_t k = 238;
	
	matrix<float,row_major> A_cpu(size,size,1.0);
	for(std::size_t i = 0; i != size; ++i){
		for(std::size_t j = 0; j <=i; ++j){
			A_cpu(i,j) = 0.3*(0.1/size*i-0.05/(i+1.0)*j);
			if(i ==j)
				A_cpu(i,j) += 10;
		}
	}
	matrix<float,row_major, opencl_tag> A = copy_to_opencl(A_cpu);
	matrix<float,row_major, opencl_tag> Aupper = trans(A);
	
	
	matrix<float,row_major> B_cpu(size,k);
	for(std::size_t i = 0; i != size; ++i){
		for(std::size_t j = 0; j != k; ++j){
			B_cpu(i,j) = (0.1/size)*i+0.1/k*j;
		}
	}
	matrix<float,row_major, opencl_tag> B = copy_to_opencl(B_cpu);
	matrix<float,row_major, opencl_tag> Bright = trans(B);

	std::cout<<"triangular opencl matrix"<<std::endl;
	std::cout<<"left - lower"<<std::endl;
	{
		matrix<float,Orientation, opencl_tag> testResult = solve(A,B, lower(), left());
		matrix<float,row_major, opencl_tag> result = triangular_prod<lower>(A,testResult);
		float error = norm_inf(result-B);
		BOOST_CHECK_SMALL(error, 1.e-5f);
	}
	std::cout<<"right - lower"<<std::endl;
	{
		matrix<float,Orientation, opencl_tag> testResult = solve(A,Bright, lower(), right());
		matrix<float, row_major, opencl_tag> result = triangular_prod<upper>(Aupper,trans(testResult));
		float error = norm_inf(trans(result)-Bright);
		BOOST_CHECK_SMALL(error, 1.e-5f);
	}
	std::cout<<"left - unit_lower"<<std::endl;
	{
		matrix<float,Orientation, opencl_tag> testResult = solve(A,B, unit_lower(), left());
		matrix<float,row_major, opencl_tag> result = triangular_prod<unit_lower>(A,testResult);
		float error = norm_inf(result-B);
		BOOST_CHECK_SMALL(error, 1.e-5f);
	}
	std::cout<<"right - unit_lower"<<std::endl;
	{
		matrix<float,Orientation, opencl_tag> testResult = solve(A,Bright, unit_lower(), right());
		matrix<float,row_major, opencl_tag> result = triangular_prod<unit_upper>(Aupper,trans(testResult));
		float error = norm_inf(trans(result) - Bright);
		BOOST_CHECK_SMALL(error, 1.e-5f);
	}
	
	std::cout<<"left - upper"<<std::endl;
	{
		matrix<float,Orientation, opencl_tag> testResult = solve(Aupper,B, upper(), left());
		matrix<float,row_major, opencl_tag> result = triangular_prod<upper>(Aupper,testResult);
		float error = norm_inf(result-B);
		BOOST_CHECK_SMALL(error, 1.e-5f);
	}
	std::cout<<"right - upper"<<std::endl;
	{
		matrix<float,Orientation, opencl_tag> testResult = solve(Aupper,Bright, upper(), right());
		matrix<float, row_major, opencl_tag> result = triangular_prod<lower>(A,trans(testResult));
		float error = norm_inf(trans(result) - Bright);
		BOOST_CHECK_SMALL(error, 1.e-5f);
	}
	std::cout<<"left - unit_upper"<<std::endl;
	{
		matrix<float,Orientation, opencl_tag> testResult = solve(Aupper,B, unit_upper(), left());
		matrix<float,row_major, opencl_tag> result = triangular_prod<unit_upper>(Aupper,testResult);
		float error = norm_inf(result-B);
		BOOST_CHECK_SMALL(error, 1.e-5f);
	}
	std::cout<<"right - unit_upper"<<std::endl;
	{
		matrix<float,Orientation, opencl_tag> testResult = solve(Aupper,Bright, unit_upper(), right());
		matrix<float,row_major, opencl_tag> result = triangular_prod<unit_lower>(A,trans(testResult));
		float error = norm_inf(trans(result) - Bright);
		BOOST_CHECK_SMALL(error, 1.e-5f);
	}
	
}

BOOST_AUTO_TEST_SUITE_END()
