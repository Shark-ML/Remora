#define BOOST_TEST_MODULE Remora_GPU_Solve_Triangular
#include <boost/test/unit_test.hpp>
#include <boost/test/floating_point_comparison.hpp>

#define BOOST_COMPUTE_DEBUG_KERNEL_COMPILATION
#include <remora/gpu/vector.hpp>
#include <remora/solve.hpp>
#include <remora/gpu/matrix.hpp>

#include <remora/gpu/copy.hpp>
#include <remora/matrix.hpp>
#include <remora/matrix_expression.hpp>
#include <remora/vector_expression.hpp>

#include <iostream>
#include <boost/mpl/list.hpp>

using namespace remora;

//simple test which checks for all argument combinations whether they are correctly translated
BOOST_AUTO_TEST_SUITE (GPU_Solve_Triangular)


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
	gpu::matrix<float,row_major> A = gpu::copy_to_gpu(A_cpu);
	gpu::matrix<float,row_major> Aupper = trans(A);
	
	vector<float> b_cpu(size);
	for(std::size_t i = 0; i != size; ++i){
		b_cpu(i) = (0.1/size)*i;
	}
	
	gpu::vector<float> b = gpu::copy_to_gpu(b_cpu);

	std::cout<<"triangular vector"<<std::endl;
	std::cout<<"left - lower"<<std::endl;
	{
		gpu::vector<float> testResult = solve(A,b, lower(), left());
		vector<float> result = copy_to_cpu(triangular_prod<lower>(A,testResult));
		float error = norm_inf(result-b_cpu);
		BOOST_CHECK_SMALL(error, 1.e-5f);
	}
	std::cout<<"right - lower"<<std::endl;
	{
		gpu::vector<float> testResult = solve(A,b, lower(), right());
		vector<float> result = copy_to_cpu(triangular_prod<upper>(Aupper,testResult));
		float error = norm_inf(result-b_cpu);
		BOOST_CHECK_SMALL(error, 1.e-5f);
	}
	
	std::cout<<"left - unit_lower"<<std::endl;
	{
		gpu::vector<float> testResult = solve(A,b, unit_lower(), left());
		vector<float> result = copy_to_cpu(triangular_prod<unit_lower>(A,testResult));
		float error = norm_inf(result-b_cpu);
		BOOST_CHECK_SMALL(error, 1.e-5f);
	}
	std::cout<<"right - unit_lower"<<std::endl;
	{
		gpu::vector<float> testResult = solve(A,b, unit_lower(), right());
		vector<float> result = copy_to_cpu(triangular_prod<unit_upper>(Aupper,testResult));
		float error = norm_inf(result-b_cpu);
		BOOST_CHECK_SMALL(error, 1.e-5f);
	}
	
	std::cout<<"left - upper"<<std::endl;
	{
		gpu::vector<float> testResult = solve(Aupper,b, upper(), left());
		vector<float> result = copy_to_cpu(triangular_prod<upper>(Aupper,testResult));
		float error = norm_inf(result-b_cpu);
		BOOST_CHECK_SMALL(error, 1.e-5f);
	}
	std::cout<<"right - upper"<<std::endl;
	{
		gpu::vector<float> testResult = solve(Aupper,b, upper(), right());
		vector<float> result = copy_to_cpu(triangular_prod<lower>(A,testResult));
		float error = norm_inf(result-b_cpu);
		BOOST_CHECK_SMALL(error, 1.e-5f);
	}
	
	std::cout<<"left - unit_upper"<<std::endl;
	{
		gpu::vector<float> testResult = solve(Aupper,b, unit_upper(), left());
		vector<float> result = copy_to_cpu(triangular_prod<unit_upper>(Aupper,testResult));
		float error = norm_inf(result-b_cpu);
		BOOST_CHECK_SMALL(error, 1.e-5f);
	}
	std::cout<<"right - unit_upper"<<std::endl;
	{
		gpu::vector<float> testResult = solve(Aupper,b, unit_upper(), right());
		vector<float> result = copy_to_cpu(triangular_prod<unit_lower>(A,testResult));
		float error = norm_inf(result-b_cpu);
		BOOST_CHECK_SMALL(error, 1.e-5f);
	}
}

typedef boost::mpl::list<row_major,column_major> result_orientations;
BOOST_AUTO_TEST_CASE_TEMPLATE(GPU_Solve_Matrix, Orientation,result_orientations) {
	std::size_t size = 139;
	std::size_t k = 238;
	
	matrix<float,row_major> A_cpu(size,size,1.0);
	for(std::size_t i = 0; i != size; ++i){
		for(std::size_t j = 0; j <=i; ++j){
			A_cpu(i,j) = 0.1/size*i-0.05/(i+1.0)*j;
			if(i ==j)
				A_cpu(i,j) += 10;
		}
	}
	gpu::matrix<float,row_major> A = gpu::copy_to_gpu(A_cpu);
	gpu::matrix<float,row_major> Aupper = trans(A);
	
	
	matrix<float,row_major> B_cpu(size,k);
	for(std::size_t i = 0; i != size; ++i){
		for(std::size_t j = 0; j != k; ++j){
			B_cpu(i,j) = (0.1/size)*i+0.1/k*j;
		}
	}
	gpu::matrix<float,row_major> B = gpu::copy_to_gpu(B_cpu);
	gpu::matrix<float,row_major> Bright = trans(B);

	std::cout<<"triangular gpu::matrix"<<std::endl;
	std::cout<<"left - lower"<<std::endl;
	{
		gpu::matrix<float,Orientation> testResult = solve(A,B, lower(), left());
		gpu::matrix<float,row_major> result = triangular_prod<lower>(A,testResult);
		float error = norm_inf(result-B);
		BOOST_CHECK_SMALL(error, 1.e-5f);
	}
	std::cout<<"right - lower"<<std::endl;
	{
		gpu::matrix<float,Orientation> testResult = solve(A,Bright, lower(), right());
		gpu::matrix<float> result = trans(gpu::matrix<float>(triangular_prod<upper>(Aupper,trans(testResult))));
		float error = norm_inf(result-Bright);
		BOOST_CHECK_SMALL(error, 1.e-5f);
	}
	std::cout<<"left - unit_lower"<<std::endl;
	{
		gpu::matrix<float,Orientation> testResult = solve(A,B, unit_lower(), left());
		gpu::matrix<float,row_major> result = triangular_prod<unit_lower>(A,testResult);
		float error = norm_inf(result-B);
		BOOST_CHECK_SMALL(error, 1.e-5f);
	}
	std::cout<<"right - unit_lower"<<std::endl;
	{
		gpu::matrix<float,Orientation> testResult = solve(A,Bright, unit_lower(), right());
		gpu::matrix<float,row_major> result = trans(gpu::matrix<float>(triangular_prod<unit_upper>(Aupper,trans(testResult))));
		float error = norm_inf(result-Bright);
		BOOST_CHECK_SMALL(error, 1.e-5f);
	}
	
	std::cout<<"left - upper"<<std::endl;
	{
		gpu::matrix<float,Orientation> testResult = solve(Aupper,B, upper(), left());
		gpu::matrix<float,row_major> result = triangular_prod<upper>(Aupper,testResult);
		float error = norm_inf(result-B);
		BOOST_CHECK_SMALL(error, 1.e-5f);
	}
	std::cout<<"right - upper"<<std::endl;
	{
		gpu::matrix<float,Orientation> testResult = solve(Aupper,Bright, upper(), right());
		gpu::matrix<float> result = trans(gpu::matrix<float>(triangular_prod<lower>(A,trans(testResult))));
		float error = norm_inf(result-Bright);
		BOOST_CHECK_SMALL(error, 1.e-5f);
	}
	std::cout<<"left - unit_upper"<<std::endl;
	{
		gpu::matrix<float,Orientation> testResult = solve(Aupper,B, unit_upper(), left());
		gpu::matrix<float,row_major> result = triangular_prod<unit_upper>(Aupper,testResult);
		float error = norm_inf(result-B);
		BOOST_CHECK_SMALL(error, 1.e-5f);
	}
	std::cout<<"right - unit_upper"<<std::endl;
	{
		gpu::matrix<float,Orientation> testResult = solve(Aupper,Bright, unit_upper(), right());
		gpu::matrix<float,row_major> result = trans(gpu::matrix<float>(triangular_prod<unit_lower>(A,trans(testResult))));
		float error = norm_inf(result-Bright);
		BOOST_CHECK_SMALL(error, 1.e-5f);
	}
	
}

BOOST_AUTO_TEST_SUITE_END()
