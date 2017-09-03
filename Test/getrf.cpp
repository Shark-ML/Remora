#define BOOST_TEST_MODULE Remora_Getrf
#include <boost/test/unit_test.hpp>
#include <boost/test/floating_point_comparison.hpp>

#include <remora/kernels/getrf.hpp>
#include <remora/dense.hpp>
#include <remora/matrix_expression.hpp>

#include <iostream>
#include <boost/mpl/list.hpp>
#include <boost/math/special_functions/fpclassify.hpp>

using namespace remora;

//the matrix is designed such that permutation will always give the next row
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
	return A;
}
typedef boost::mpl::list<row_major,column_major> result_orientations;


BOOST_AUTO_TEST_SUITE (Remora_Cholesky)

BOOST_AUTO_TEST_CASE_TEMPLATE(Remora_Potrf, Orientation,result_orientations) {
	std::size_t Dimensions = 123;
	//first generate a suitable eigenvalue problem matrix A
	matrix<double,Orientation> A = createMatrix(Dimensions);
	//calculate lu decomposition
	permutation_matrix P(Dimensions);
	matrix<double,Orientation> dec = A;
	kernels::getrf(dec,P);

	//copy upper matrix to temporary
	matrix<double> upper(Dimensions,Dimensions,0.0);
	for (size_t row = 0; row < Dimensions; row++){
		for (size_t col = row; col < Dimensions ; col++){
			upper(row, col) = dec(row, col);
		}
	}
	
	//create reconstruction of A
	matrix<double> testA = triangular_prod<unit_lower>(dec,upper);
	swap_rows_inverted(P,testA);
	
	//test reconstruction error
	double error = max(abs(A - testA));
	BOOST_CHECK_SMALL(error,1.e-12);
	BOOST_CHECK(!(boost::math::isnan)(norm_frobenius(testA)));//test for nans
}

BOOST_AUTO_TEST_SUITE_END()
