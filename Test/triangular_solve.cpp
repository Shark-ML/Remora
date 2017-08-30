#define BOOST_TEST_MODULE Remora_Solve_Triangular
#include <boost/test/unit_test.hpp>
#include <boost/test/floating_point_comparison.hpp>

#include <remora/solve.hpp>
#include <remora/dense.hpp>
#include <remora/matrix_expression.hpp>
#include <remora/proxy_expressions.hpp>
#include <remora/vector_expression.hpp>

#include <iostream>
#include <boost/mpl/list.hpp>

using namespace remora;

//simple test which checks for all argument combinations whether they are correctly translated
BOOST_AUTO_TEST_SUITE (Solve_Triangular)

BOOST_AUTO_TEST_CASE( Solve_Vector ){
	std::size_t size = 158;
	
	matrix<double,row_major> A(size,size,1.0);
	for(std::size_t i = 0; i != size; ++i){
		for(std::size_t j = 0; j <=i; ++j){
			A(i,j) = 0.1/size*i-0.05/(i+1.0)*j;
			if(i ==j)
				A(i,j) += 0.5;
		}
	}
	matrix<double,row_major> Aupper = trans(A);
	
	vector<double> b(size);
	for(std::size_t i = 0; i != size; ++i){
		b(i) = (0.1/size)*i;
	}

	std::cout<<"triangular vector"<<std::endl;
	std::cout<<"left - lower"<<std::endl;
	{
		vector<double> testResult = solve(A,b, lower(), left());
		vector<double> resultProd = prod(inv(A,lower()),b);
		BOOST_CHECK_SMALL(norm_inf(testResult - resultProd),1.e-15);//check that both expressions are the same
		vector<double> result = triangular_prod<lower>(A,testResult);
		double error = norm_inf(result-b);
		BOOST_CHECK_SMALL(error, 1.e-11);
	}
	std::cout<<"right - lower"<<std::endl;
	{
		vector<double> testResult = solve(A,b, lower(), right());
		vector<double> resultProd = prod(b,inv(A,lower()));
		BOOST_CHECK_SMALL(norm_inf(testResult - resultProd),1.e-15);
		vector<double> result = triangular_prod<upper>(Aupper,testResult);
		double error = norm_inf(result-b);
		BOOST_CHECK_SMALL(error, 1.e-11);
	}
	
	std::cout<<"left - unit_lower"<<std::endl;
	{
		vector<double> testResult = solve(A,b,unit_lower(), left());
		vector<double> resultProd = prod(inv(A,unit_lower()),b);
		BOOST_CHECK_SMALL(norm_inf(testResult - resultProd),1.e-15);
		vector<double> result = triangular_prod<unit_lower>(A,testResult);
		double error = norm_inf(result-b);
		BOOST_CHECK_SMALL(error, 1.e-11);
	}
	std::cout<<"right - unit_lower"<<std::endl;
	{
		vector<double> testResult = solve(A,b, unit_lower(), right());
		vector<double> resultProd = prod(b,inv(A,unit_lower()));
		BOOST_CHECK_SMALL(norm_inf(testResult - resultProd),1.e-15);
		vector<double> result = triangular_prod<unit_upper>(Aupper,testResult);
		double error = norm_inf(result-b);
		BOOST_CHECK_SMALL(error, 1.e-11);
	}
	
	std::cout<<"left - upper"<<std::endl;
	{
		vector<double> testResult = solve(Aupper,b, upper(), left());
		vector<double> resultProd = prod(inv(Aupper,upper()),b);
		BOOST_CHECK_SMALL(norm_inf(testResult - resultProd),1.e-15);
		vector<double> result = triangular_prod<upper>(Aupper,testResult);
		double error = norm_inf(result-b);
		BOOST_CHECK_SMALL(error, 1.e-11);
	}
	std::cout<<"right - upper"<<std::endl;
	{
		vector<double> testResult = solve(Aupper,b,upper(), right());
		vector<double> resultProd = prod(b,inv(Aupper,upper()));
		BOOST_CHECK_SMALL(norm_inf(testResult - resultProd),1.e-15);
		vector<double> result = triangular_prod<lower>(A,testResult);
		double error = norm_inf(result-b);
		BOOST_CHECK_SMALL(error, 1.e-11);
	}
	
	std::cout<<"left - unit_upper"<<std::endl;
	{
		vector<double> testResult = solve(Aupper,b, unit_upper(), left());
		vector<double> resultProd = prod(inv(Aupper,unit_upper()),b);
		BOOST_CHECK_SMALL(norm_inf(testResult - resultProd),1.e-15);
		vector<double> result = triangular_prod<unit_upper>(Aupper,testResult);
		double error = norm_inf(result-b);
		BOOST_CHECK_SMALL(error, 1.e-11);
	}
	std::cout<<"right - unit_upper"<<std::endl;
	{
		vector<double> testResult = solve(Aupper,b, unit_upper(), right());
		vector<double> resultProd = prod(b,inv(Aupper,unit_upper()));
		BOOST_CHECK_SMALL(norm_inf(testResult - resultProd),1.e-15);
		vector<double> result = triangular_prod<unit_lower>(A,testResult);
		double error = norm_inf(result-b);
		BOOST_CHECK_SMALL(error, 1.e-11);
	}
}

typedef boost::mpl::list<row_major,column_major> result_orientations;
BOOST_AUTO_TEST_CASE_TEMPLATE(Solve_Matrix, Orientation,result_orientations) {
	std::size_t size = 158;
	std::size_t k = 138;
	
	matrix<double,row_major> A(size,size,1.0);
	for(std::size_t i = 0; i != size; ++i){
		for(std::size_t j = 0; j <=i; ++j){
			A(i,j) = 0.2/size*i-0.05/(i+1.0)*j;
			if(i ==j)
				A(i,j) += 10;
		}
	}
	matrix<double,row_major> Aupper = trans(A);
	
	matrix<double,Orientation> B(size,k);
	for(std::size_t i = 0; i != size; ++i){
		for(std::size_t j = 0; j != k; ++j){
			B(i,j) = (0.1/size)*i+0.1*k;
		}
	}
	matrix<double,Orientation> Bright = trans(B);

	std::cout<<"triangular matrix"<<std::endl;
	std::cout<<"left - lower"<<std::endl;
	{
		matrix<double,Orientation> testResult = solve(A,B, lower(), left());
		matrix<double,Orientation> prodResult = prod(inv(A,lower()),B);
		BOOST_CHECK_SMALL(max(abs(testResult - prodResult)),1.e-15);
		matrix<double,row_major> result = triangular_prod<lower>(A,testResult);
		double error = norm_inf(result-B);
		BOOST_CHECK_SMALL(error, 1.e-11);
	}
	std::cout<<"right - lower"<<std::endl;
	{
		matrix<double,Orientation> testResult = solve(A,Bright, lower(), right());
		matrix<double,Orientation> prodResult = prod(Bright,inv(A,lower()));
		BOOST_CHECK_SMALL(max(abs(testResult - prodResult)),1.e-15);
		matrix<double> result = triangular_prod<upper>(Aupper,trans(testResult));
		double error = norm_inf(trans(result)-Bright);
		BOOST_CHECK_SMALL(error, 1.e-11);
	}
	std::cout<<"left - unit_lower"<<std::endl;
	{
		matrix<double,Orientation> testResult = solve(A,B, unit_lower(), left());
		matrix<double,Orientation> prodResult = prod(inv(A,unit_lower()),B);
		BOOST_CHECK_SMALL(max(abs(testResult - prodResult)),1.e-15);
		matrix<double,row_major> result = triangular_prod<unit_lower>(A,testResult);
		double error = norm_inf(result-B);
		BOOST_CHECK_SMALL(error, 1.e-11);
	}
	std::cout<<"right - unit_lower"<<std::endl;
	{
		matrix<double,Orientation> testResult = solve(A,Bright, unit_lower(), right());
		matrix<double,Orientation> prodResult = prod(Bright,inv(A,unit_lower()));
		BOOST_CHECK_SMALL(max(abs(testResult - prodResult)),1.e-15);
		matrix<double,row_major> result = triangular_prod<unit_upper>(Aupper,trans(testResult));
		double error = norm_inf(trans(result)-Bright);
		BOOST_CHECK_SMALL(error, 1.e-11);
	}
	
	std::cout<<"left - upper"<<std::endl;
	{
		matrix<double,Orientation> testResult = solve(Aupper,B, upper(), left());
		matrix<double,Orientation> prodResult = prod(inv(Aupper,upper()),B);
		BOOST_CHECK_SMALL(max(abs(testResult - prodResult)),1.e-15);
		matrix<double,row_major> result = triangular_prod<upper>(Aupper,testResult);
		double error = norm_inf(result-B);
		BOOST_CHECK_SMALL(error, 1.e-11);
	}
	std::cout<<"right - upper"<<std::endl;
	{
		matrix<double,Orientation> testResult = solve(Aupper,Bright, upper(), right());
		matrix<double,Orientation> prodResult = prod(Bright,inv(Aupper,upper()));
		BOOST_CHECK_SMALL(max(abs(testResult - prodResult)),1.e-15);
		matrix<double> result = triangular_prod<lower>(A,trans(testResult));
		double error = norm_inf(trans(result)-Bright);
		BOOST_CHECK_SMALL(error, 1.e-11);
	}
	std::cout<<"left - unit_upper"<<std::endl;
	{
		matrix<double,Orientation> testResult = solve(Aupper,B, unit_upper(), left());
		matrix<double,Orientation> prodResult = prod(inv(Aupper,unit_upper()),B);
		BOOST_CHECK_SMALL(max(abs(testResult - prodResult)),1.e-15);
		matrix<double,row_major> result = triangular_prod<unit_upper>(Aupper,testResult);
		double error = norm_inf(result-B);
		BOOST_CHECK_SMALL(error, 1.e-11);
	}
	std::cout<<"right - unit_upper"<<std::endl;
	{
		matrix<double,Orientation> testResult = solve(Aupper,Bright, unit_upper(), right());
		matrix<double,Orientation> prodResult = prod(Bright,inv(Aupper,unit_upper()));
		BOOST_CHECK_SMALL(max(abs(testResult - prodResult)),1.e-15);
		matrix<double,row_major> result = triangular_prod<unit_lower>(A,trans(testResult));
		double error = norm_inf(trans(result)-Bright);
		BOOST_CHECK_SMALL(error, 1.e-11);
	}
}

BOOST_AUTO_TEST_SUITE_END()
