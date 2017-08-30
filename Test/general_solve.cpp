#define BOOST_TEST_MODULE Remora_General_Solve
#include <boost/test/unit_test.hpp>
#include <boost/test/floating_point_comparison.hpp>

#include <remora/solve.hpp>
#include <remora/dense.hpp>
#include <remora/matrix_expression.hpp>
#include <remora/vector_expression.hpp>

#include <iostream>
#include <boost/mpl/list.hpp>

using namespace remora;

//the matrix is designed such that a lot of permutations will be performed
matrix<double> createMatrix(std::size_t dimensions){
	matrix<double> L(dimensions,dimensions,0.0);
	matrix<double> U(dimensions,dimensions,0.0);
	
	for(std::size_t i = 0; i != dimensions; ++i){
		for(std::size_t j = 0; j <i; ++j){
			U(j,i) = 0.0;//1 - 0.1/dimensions * std::abs((int)i -(int)j);
			L(i,j)  = 3 - 3.0/dimensions*std::abs((int)i -(int)j);
		}
		U(i,i) = 0.5/dimensions*i+1;
		L(i,i) = 1;
	}
	matrix<double> A = prod(L,U);
	diag(A) += 1;
	return A;
}
typedef boost::mpl::list<row_major,column_major> result_orientations;

//simple test which checks for all argument combinations whether they are correctly translated
BOOST_AUTO_TEST_SUITE (Solve_General)

BOOST_AUTO_TEST_CASE( Solve_Indefinite_Full_Rank_Vector ){
	std::size_t Dimensions = 123;
	matrix<double> A = createMatrix(Dimensions);
	vector<double> b(Dimensions);
	for(std::size_t i = 0; i != Dimensions; ++i){
		b(i) = (1.0/Dimensions) * i;
	}
	
	//Ax=b
	{
		vector<double> x = solve(A,b, indefinite_full_rank(),left());
		vector<double> xprod = prod(inv(A,indefinite_full_rank()),b);
		BOOST_CHECK_SMALL(norm_inf(x-xprod),1.e-15);
		vector<double> test = prod(A,x);
		double error = norm_inf(test-b);
		BOOST_CHECK_SMALL(error,1.e-12);
	}
	//A^Tx=b
	{
		vector<double> x = solve(trans(A),b, indefinite_full_rank(),left());
		vector<double> xprod = prod(inv(trans(A),indefinite_full_rank()),b);
		BOOST_CHECK_SMALL(norm_inf(x-xprod),1.e-15);
		vector<double> test = prod(trans(A),x);
		double error = norm_inf(test-b);
		BOOST_CHECK_SMALL(error,1.e-12);
	}
	//xA=b
	{
		matrix<double,column_major> t=A;
		vector<double> x = solve(A,b, indefinite_full_rank(),right());
		vector<double> xprod = prod(b,inv(A,indefinite_full_rank()));
		BOOST_CHECK_SMALL(norm_inf(x-xprod),1.e-15);
		vector<double> test = prod(x,A);
		double error = norm_inf(test-b);
		BOOST_CHECK_SMALL(error,1.e-12);
	}
	
	//xA^T=b
	{
		vector<double> x = solve(trans(A),b, indefinite_full_rank(),right());
		vector<double> xprod = prod(b,inv(trans(A),indefinite_full_rank()));
		BOOST_CHECK_SMALL(norm_inf(x-xprod),1.e-14);
		vector<double> test = prod(x,trans(A));
		double error = norm_inf(test-b);
		BOOST_CHECK_SMALL(error,1.e-12);
	}
}

BOOST_AUTO_TEST_CASE_TEMPLATE(Solve_Matrix, Orientation,result_orientations) {
	std::size_t Dimensions = 128;
	std::size_t k = 151;
	
	std::cout<<"solve Symmetric matrix"<<std::endl;
	matrix<double> A = createMatrix(Dimensions);
	matrix<double,Orientation> B(Dimensions,k);
	for(std::size_t i = 0; i != Dimensions; ++i){
		for(std::size_t j = 0; j < k; ++j){
			B(i,j) = 0.1*j+(1.0/Dimensions) * i;
		}
	}
	matrix<double,Orientation> Bright = trans(B);
	//Ax=b
	{
		matrix<double,Orientation> X= solve(A,B, indefinite_full_rank(),left());
		matrix<double,Orientation> Xprod = prod(inv(A,indefinite_full_rank()),B);
		BOOST_CHECK_SMALL(max(abs(X-Xprod)),1.e-15);
		matrix<double> test = prod(A,X);
		double error = max(abs(test-B));
		BOOST_CHECK_SMALL(error,1.e-12);
	}
	//A^Tx=b
	{
		matrix<double,Orientation> X = solve(trans(A),B, indefinite_full_rank(),left());
		matrix<double,Orientation> Xprod = prod(inv(trans(A),indefinite_full_rank()),B);
		BOOST_CHECK_SMALL(max(abs(X-Xprod)),1.e-15);
		matrix<double> test = prod(trans(A),X);
		double error = max(abs(test-B));
		BOOST_CHECK_SMALL(error,1.e-12);
	}
	//xA=b
	{
		matrix<double,Orientation> X = solve(A,Bright, indefinite_full_rank(),right());
		matrix<double,Orientation> Xprod = prod(Bright,inv(A,indefinite_full_rank()));
		BOOST_CHECK_SMALL(max(abs(X-Xprod)),1.e-15);
		matrix<double> test = prod(X,A);
		double error = max(abs(test-Bright));
		BOOST_CHECK_SMALL(error,1.e-12);
	}
	
	//xA^T=b
	{
		matrix<double,Orientation> X = solve(trans(A),Bright, indefinite_full_rank(),right());
		matrix<double,Orientation> Xprod = prod(Bright,inv(trans(A),indefinite_full_rank()));
		BOOST_CHECK_SMALL(max(abs(X-Xprod)),1.e-15);
		matrix<double> test = prod(X,trans(A));
		double error = max(abs(test-Bright));
		BOOST_CHECK_SMALL(error,1.e-12);
	}
}

BOOST_AUTO_TEST_SUITE_END()
