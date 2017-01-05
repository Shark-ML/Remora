#define BOOST_TEST_MODULE Remora_Symm_Solve
#include <boost/test/unit_test.hpp>
#include <boost/test/floating_point_comparison.hpp>

#include <remora/solve.hpp>
#include <remora/matrix.hpp>
#include <remora/matrix_expression.hpp>
#include <remora/matrix_proxy.hpp>
#include <remora/vector_expression.hpp>

#include <iostream>
#include <boost/mpl/list.hpp>

using namespace remora;

//the matrix is designed such that a lot of permutations will be performed
matrix<double> createSymm(std::size_t dimensions, std::size_t rank = 0){
	if(rank == 0) rank = dimensions;
	matrix<double> R(dimensions,dimensions,0.0);
	
	for(std::size_t i = 0; i != dimensions; ++i){
		for(std::size_t j = 0; j <std::min(i,rank); ++j){
			R(i,j) = 0.2/std::abs((int)i -(int)j);
		}
		if(i < rank)
			R(i,i) = 0.5/dimensions*i+1;
	}
	matrix<double> A = prod(R,trans(R));
	if(rank != dimensions){
		for(std::size_t i = 0; i != rank/2; ++i){
			A.swap_rows(2*i,dimensions-i-1);
			A.swap_columns(2*i,dimensions-i-1);
		}
	}
	return A;
}
typedef boost::mpl::list<row_major,column_major> result_orientations;

//simple test which checks for all argument combinations whether they are correctly translated
BOOST_AUTO_TEST_SUITE (Solve_Symm)

BOOST_AUTO_TEST_CASE( Solve_Symm_Vector ){
	std::size_t Dimensions = 128;
	matrix<double> A = createSymm(Dimensions);
	vector<double> b(Dimensions);
	for(std::size_t i = 0; i != Dimensions; ++i){
		b(i) = (1.0/Dimensions) * i;
	}
	
	//Ax=b
	{
		vector<double> x = solve(A,b, symm_pos_def(),left());
		vector<double> xprod = prod(inv(A,symm_pos_def()),b);
		BOOST_CHECK_SMALL(norm_inf(x-xprod),1.e-15);//check that both expressions are the same
		vector<double> test = prod(A,x);
		double error = norm_inf(test-b);
		BOOST_CHECK_SMALL(error,1.e-12);
	}
	//A^Tx=b
	{
		vector<double> x = solve(trans(A),b, symm_pos_def(),left());
		vector<double> xprod = prod(inv(trans(A),symm_pos_def()),b);
		BOOST_CHECK_SMALL(norm_inf(x-xprod),1.e-15);//check that both expressions are the same
		vector<double> test = prod(trans(A),x);
		double error = norm_inf(test-b);
		BOOST_CHECK_SMALL(error,1.e-12);
	}
	//xA=b
	{
		vector<double> x = solve(A,b, symm_pos_def(),right());
		vector<double> xprod = prod(b,inv(A,symm_pos_def()));
		BOOST_CHECK_SMALL(norm_inf(x-xprod),1.e-15);//check that both expressions are the same
		vector<double> test = prod(x,A);
		double error = norm_inf(test-b);
		BOOST_CHECK_SMALL(error,1.e-12);
	}
	
	//xA^T=b
	{
		vector<double> x = solve(trans(A),b, symm_pos_def(),right());
		vector<double> xprod = prod(b,inv(trans(A),symm_pos_def()));
		BOOST_CHECK_SMALL(norm_inf(x-xprod),1.e-15);//check that both expressions are the same
		vector<double> test = prod(x,trans(A));
		double error = norm_inf(test-b);
		BOOST_CHECK_SMALL(error,1.e-12);
	}
}

BOOST_AUTO_TEST_CASE_TEMPLATE(Solve_Matrix, Orientation,result_orientations) {
	std::size_t Dimensions = 128;
	std::size_t k = 151;
	
	std::cout<<"solve Symmetric matrix"<<std::endl;
	matrix<double> A = createSymm(Dimensions);
	matrix<double,Orientation> B(Dimensions,k);
	for(std::size_t i = 0; i != Dimensions; ++i){
		for(std::size_t j = 0; j < k; ++j){
			B(i,j) = 0.1*j+(1.0/Dimensions) * i;
		}
	}
	matrix<double,Orientation> Bright = trans(B);
	//Ax=b
	{
		matrix<double,Orientation> X= solve(A,B, symm_pos_def(),left());
		matrix<double,Orientation> Xprod = prod(inv(A,symm_pos_def()),B);
		BOOST_CHECK_SMALL(max(abs(X-Xprod)),1.e-15);//check that both expressions are the same
		matrix<double> test = prod(A,X);
		double error = max(abs(test-B));
		BOOST_CHECK_SMALL(error,1.e-12);
	}
	//A^Tx=b
	{
		matrix<double,Orientation> X = solve(trans(A),B, symm_pos_def(),left());
		matrix<double,Orientation> Xprod = prod(trans(inv(A,symm_pos_def())),B);
		BOOST_CHECK_SMALL(max(abs(X-Xprod)),1.e-15);//check that both expressions are the same
		matrix<double> test = prod(trans(A),X);
		double error = max(abs(test-B));
		BOOST_CHECK_SMALL(error,1.e-12);
	}
	//xA=b
	{
		matrix<double,Orientation> X = solve(A,Bright, symm_pos_def(),right());
		matrix<double,Orientation> Xprod = prod(Bright,inv(A,symm_pos_def()));
		BOOST_CHECK_SMALL(max(abs(X-Xprod)),1.e-15);//check that both expressions are the same
		matrix<double> test = prod(X,A);
		double error = max(abs(test-Bright));
		BOOST_CHECK_SMALL(error,1.e-12);
	}
	
	//xA^T=b
	{
		matrix<double,Orientation> X = solve(trans(A),Bright, symm_pos_def(),right());
		matrix<double,Orientation> Xprod = prod(Bright, trans(inv(A,symm_pos_def())));
		BOOST_CHECK_SMALL(max(abs(X-Xprod)),1.e-15);//check that both expressions are the same
		matrix<double> test = prod(X,trans(A));
		double error = max(abs(test-Bright));
		BOOST_CHECK_SMALL(error,1.e-12);
	}
}

BOOST_AUTO_TEST_CASE( Solve_Symm_Semi_Pos_Def_Vector_Full_Rank ){
	std::size_t Dimensions = 128;
	
	std::cout<<"solve Symmetric semi pos def vector, full rank"<<std::endl;
	matrix<double> A = createSymm(Dimensions);
	vector<double> b(Dimensions);
	for(std::size_t i = 0; i != Dimensions; ++i){
		b(i) = (1.0/Dimensions) * i;
	}
	
	//Ax=b
	{
		vector<double> x = solve(A,b, symm_semi_pos_def(),left());
		vector<double> xprod = prod(inv(A,symm_semi_pos_def()),b);
		BOOST_CHECK_SMALL(norm_inf(x-xprod),1.e-15);//check that both expressions are the same
		vector<double> test = prod(A,x);
		double error = norm_inf(test-b);
		BOOST_CHECK_SMALL(error,1.e-12);
	}
	//A^Tx=b
	{
		vector<double> x = solve(trans(A),b, symm_semi_pos_def(),left());
		vector<double> xprod = prod(inv(trans(A),symm_semi_pos_def()),b);
		BOOST_CHECK_SMALL(norm_inf(x-xprod),1.e-15);//check that both expressions are the same
		vector<double> test = prod(trans(A),x);
		double error = norm_inf(test-b);
		BOOST_CHECK_SMALL(error,1.e-12);
	}
	//xA=b
	{
		vector<double> x = solve(A,b, symm_semi_pos_def(),right());
		vector<double> xprod = prod(b,inv(A,symm_semi_pos_def()));
		BOOST_CHECK_SMALL(norm_inf(x-xprod),1.e-15);//check that both expressions are the same
		vector<double> test = prod(x,A);
		double error = norm_inf(test-b);
		BOOST_CHECK_SMALL(error,1.e-12);
	}
	
	//xA^T=b
	{
		vector<double> x = solve(trans(A),b, symm_semi_pos_def(),right());
		vector<double> xprod = prod(b,inv(trans(A),symm_semi_pos_def()));
		BOOST_CHECK_SMALL(norm_inf(x-xprod),1.e-15);//check that both expressions are the same
		vector<double> test = prod(x,trans(A));
		double error = norm_inf(test-b);
		BOOST_CHECK_SMALL(error,1.e-12);
	}
}

typedef boost::mpl::list<row_major,column_major> result_orientations;
BOOST_AUTO_TEST_CASE_TEMPLATE(Solve_Symm_Semi_Pos_Def_Matrix_Full_Rank, Orientation,result_orientations) {
	std::size_t Dimensions = 128;
	std::size_t k = 151;
	
	std::cout<<"solve Symmetric semi pos def matrix, full rank"<<std::endl;
	matrix<double> A = createSymm(Dimensions);
	matrix<double,Orientation> B(Dimensions,k);
	for(std::size_t i = 0; i != Dimensions; ++i){
		for(std::size_t j = 0; j < k; ++j){
			B(i,j) = 0.1*j+(1.0/Dimensions) * i;
		}
	}
	matrix<double,Orientation> Bright = trans(B);
	//Ax=b
	{
		matrix<double,Orientation> X= solve(A,B, symm_semi_pos_def(),left());
		matrix<double,Orientation> Xprod = prod(inv(A,symm_semi_pos_def()),B);
		BOOST_CHECK_SMALL(max(abs(X-Xprod)),1.e-15);//check that both expressions are the same
		matrix<double> test = prod(A,X);
		double error = max(abs(test-B));
		BOOST_CHECK_SMALL(error,1.e-12);
	}
	//A^Tx=b
	{
		matrix<double,Orientation> X = solve(trans(A),B, symm_semi_pos_def(),left());
		matrix<double,Orientation> Xprod = prod(inv(trans(A),symm_semi_pos_def()),B);
		BOOST_CHECK_SMALL(max(abs(X-Xprod)),1.e-15);//check that both expressions are the same
		matrix<double> test = prod(trans(A),X);
		double error = max(abs(test-B));
		BOOST_CHECK_SMALL(error,1.e-12);
	}
	//xA=b
	{
		matrix<double,Orientation> X = solve(A,Bright, symm_semi_pos_def(),right());
		matrix<double,Orientation> Xprod = prod(Bright, inv(A,symm_semi_pos_def()));
		BOOST_CHECK_SMALL(max(abs(X-Xprod)),1.e-15);//check that both expressions are the same
		matrix<double> test = prod(X,A);
		double error = max(abs(test-Bright));
		BOOST_CHECK_SMALL(error,1.e-12);
	}
	
	//xA^T=b
	{
		matrix<double,Orientation> X = solve(trans(A),Bright, symm_semi_pos_def(),right());
		matrix<double,Orientation> Xprod = prod(Bright, inv(trans(A),symm_semi_pos_def()));
		BOOST_CHECK_SMALL(max(abs(X-Xprod)),1.e-15);//check that both expressions are the same
		matrix<double> test = prod(X,trans(A));
		double error = max(abs(test-Bright));
		BOOST_CHECK_SMALL(error,1.e-12);
	}
}

BOOST_AUTO_TEST_CASE( Solve_Symm_Semi_Pos_Def_Vector_Rank_Deficient ){
	std::size_t Dimensions = 128;
	std::size_t Rank = 50;
	
	std::cout<<"solve Symmetric semi pos def vector, rank deficient"<<std::endl;
	matrix<double> A = createSymm(Dimensions,Rank);
	vector<double> b(Dimensions);
	for(std::size_t i = 0; i != Dimensions; ++i){
		b(i) = (1.0/Dimensions) * i;
	}
	
	//if A is not full rank and b i not orthogonal to the nullspace,
	//Ax -b does not hold for the right solution.
	//instead A(Ax-b) must be small (the residual of Ax-b must be orthogonal to A)
	
	//Ax=b
	{
		vector<double> x = solve(A,b, symm_semi_pos_def(),left());
		vector<double> diff = prod(A,x)-b;
		double error = norm_inf(prod(A,diff));
		BOOST_CHECK_SMALL(error,1.e-12);
	}
	//A^Tx=b
	{
		vector<double> x = solve(trans(A),b, symm_semi_pos_def(),left());
		vector<double> diff = prod(A,x)-b;
		double error = norm_inf(prod(A,diff));
		BOOST_CHECK_SMALL(error,1.e-12);
	}
	//xA=b
	{
		vector<double> x = solve(A,b, symm_semi_pos_def(),right());
		vector<double> diff = prod(A,x)-b;
		double error = norm_inf(prod(A,diff));
		BOOST_CHECK_SMALL(error,1.e-12);
	}
	
	//xA^T=b
	{
		vector<double> x = solve(trans(A),b, symm_semi_pos_def(),right());
		vector<double> diff = prod(A,x)-b;
		double error = norm_inf(prod(A,diff));
		BOOST_CHECK_SMALL(error,1.e-12);
	}
}

BOOST_AUTO_TEST_CASE_TEMPLATE(Solve_Symm_Semi_Pos_Def_Matrix_Rank_Deficient, Orientation,result_orientations) {
	std::size_t Dimensions = 128;
	std::size_t Rank = 50;
	std::size_t k = 151;
	
	std::cout<<"solve Symmetric semi pos def matrix, rank deficient"<<std::endl;
	matrix<double> A = createSymm(Dimensions, Rank);
	matrix<double,Orientation> B(Dimensions,k);
	for(std::size_t i = 0; i != Dimensions; ++i){
		for(std::size_t j = 0; j < k; ++j){
			B(i,j) = 0.1*j+(1.0/Dimensions) * i;
		}
	}
	matrix<double,Orientation> Bright = trans(B);
	//Ax=b
	{
		matrix<double,Orientation> X= solve(A,B, symm_semi_pos_def(),left());
		matrix<double> diff = prod(A,X) - B;
		double error = max(abs(prod(A,diff)));
		BOOST_CHECK_SMALL(error,1.e-12);
	}
	//A^Tx=b
	{
		matrix<double,Orientation> X = solve(trans(A),B, symm_semi_pos_def(),left());
		matrix<double> diff = prod(A,X) - B;
		double error = max(abs(prod(A,diff)));
		BOOST_CHECK_SMALL(error,1.e-12);
	}
	//xA=b
	{
		matrix<double,Orientation> X = solve(A,Bright, symm_semi_pos_def(),right());
		matrix<double> diff = prod(X,A) - Bright;
		double error = max(abs(prod(diff,A)));
		BOOST_CHECK_SMALL(error,1.e-12);
	}
	
	//xA^T=b
	{
		matrix<double,Orientation> X = solve(trans(A),Bright, symm_semi_pos_def(),right());
		matrix<double> diff = prod(X,A) - Bright;
		double error = max(abs(prod(diff,A)));
		BOOST_CHECK_SMALL(error,1.e-12);
	}
}

BOOST_AUTO_TEST_CASE( Solve_Symm_Conjugate_Gradient_Vector ){
	std::size_t Dimensions = 128;
	
	std::cout<<"solve Symmetric conjugate gradient, Vector"<<std::endl;
	matrix<double> A = createSymm(Dimensions);
	vector<double> b(Dimensions);
	for(std::size_t i = 0; i != Dimensions; ++i){
		b(i) = (1.0/Dimensions) * i;
	}
	
	//if A is not full rank and b i not orthogonal to the nullspace,
	//Ax -b does not hold for the right solution.
	//instead A(Ax-b) must be small (the residual of Ax-b must be orthogonal to A)
	
	//Ax=b
	{
		vector<double> x = solve(A,b, conjugate_gradient(1.e-9),left());
		vector<double> xprod = prod(inv(A,conjugate_gradient(1.e-9)),b);
		BOOST_CHECK_SMALL(norm_inf(x-xprod),1.e-15);
		vector<double> diff = prod(A,x)-b;
		double error = norm_inf(diff);
		BOOST_CHECK_SMALL(error,1.e-8);
	}
	//A^Tx=b
	{
		vector<double> x = solve(trans(A),b, conjugate_gradient(1.e-9),left());
		vector<double> xprod = prod(trans(inv(A,conjugate_gradient(1.e-9))),b);
		BOOST_CHECK_SMALL(norm_inf(x-xprod),1.e-15);
		vector<double> diff = prod(A,x)-b;
		double error = norm_inf(diff);
		BOOST_CHECK_SMALL(error,1.e-8);
	}
	//xA=b
	{
		vector<double> x = solve(A,b, conjugate_gradient(1.e-9),right());
		vector<double> xprod = prod(b,inv(A,conjugate_gradient(1.e-9)));
		BOOST_CHECK_SMALL(norm_inf(x-xprod),1.e-14);
		vector<double> diff = prod(A,x)-b;
		double error = norm_inf(diff);
		BOOST_CHECK_SMALL(error,1.e-8);
	}
	
	//xA^T=b
	{
		vector<double> x = solve(trans(A),b, conjugate_gradient(1.e-9),right());
		vector<double> xprod = prod(b,trans(inv(A,conjugate_gradient(1.e-9))));
		BOOST_CHECK_SMALL(norm_inf(x-xprod),1.e-14);
		vector<double> diff = prod(A,x)-b;
		double error = norm_inf(diff);
		BOOST_CHECK_SMALL(error,1.e-8);
	}
}

BOOST_AUTO_TEST_CASE_TEMPLATE(Solve_Symm_Conjugate_Gradient, Orientation,result_orientations) {
	std::size_t Dimensions = 128;
	std::size_t k = 151;
	
	std::cout<<"solve Symmetric semi pos def matrix, full rank"<<std::endl;
	matrix<double> A = createSymm(Dimensions);
	matrix<double,Orientation> B(Dimensions,k);
	for(std::size_t i = 0; i != Dimensions; ++i){
		for(std::size_t j = 0; j < k; ++j){
			B(i,j) = 0.1*j+(1.0/Dimensions) * i;
		}
	}
	matrix<double,Orientation> Bright = trans(B);
	//Ax=b
	{
		matrix<double,Orientation> X= solve(A,B, conjugate_gradient(1.e-9),left());
		matrix<double,Orientation> Xprod = prod(inv(A,conjugate_gradient(1.e-9)),B);
		BOOST_CHECK_SMALL(max(abs(X-Xprod)),1.e-15);
		matrix<double> test = prod(A,X);
		double error = max(abs(test-B));
		BOOST_CHECK_SMALL(error,1.e-8);
	}
	//A^Tx=b
	{
		matrix<double,Orientation> X = solve(trans(A),B, conjugate_gradient(1.e-9),left());
		matrix<double,Orientation> Xprod = prod(inv(trans(A),conjugate_gradient(1.e-9)),B);
		BOOST_CHECK_SMALL(max(abs(X-Xprod)),1.e-15);
		matrix<double> test = prod(trans(A),X);
		double error = max(abs(test-B));
		BOOST_CHECK_SMALL(error,1.e-8);
	}
	//xA=b
	{
		matrix<double,Orientation> X = solve(A,Bright, conjugate_gradient(1.e-9),right());
		matrix<double,Orientation> Xprod = prod(Bright, inv(A,conjugate_gradient(1.e-9)));
		BOOST_CHECK_SMALL(max(abs(X-Xprod)),1.e-15);
		matrix<double> test = prod(X,A);
		double error = max(abs(test-Bright));
		BOOST_CHECK_SMALL(error,1.e-8);
	}
	
	//xA^T=b
	{
		matrix<double,Orientation> X = solve(trans(A),Bright, conjugate_gradient(1.e-9),right());
		matrix<double,Orientation> Xprod = prod(Bright, inv(trans(A),conjugate_gradient(1.e-9)));
		BOOST_CHECK_SMALL(max(abs(X-Xprod)),1.e-15);
		matrix<double> test = prod(X,trans(A));
		double error = max(abs(test-Bright));
		BOOST_CHECK_SMALL(error,1.e-8);
	}
}

BOOST_AUTO_TEST_SUITE_END()
