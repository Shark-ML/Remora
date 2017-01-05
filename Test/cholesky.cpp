#define BOOST_TEST_MODULE Remora_Cholesky
#include <boost/test/unit_test.hpp>
#include <boost/test/floating_point_comparison.hpp>

#include <remora/solve.hpp>
#include <remora/matrix.hpp>
#include <remora/matrix_expression.hpp>
#include <remora/matrix_proxy.hpp>


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


BOOST_AUTO_TEST_SUITE (Remora_Cholesky)

BOOST_AUTO_TEST_CASE_TEMPLATE(Remora_Potrf, Orientation,result_orientations) {
	std::size_t Dimensions = 123;
	//first generate a suitable eigenvalue problem matrix A
	matrix<double,Orientation> A = createSymm(Dimensions);
	//calculate Cholesky
	matrix<double,Orientation> lowDec = A;
	matrix<double,Orientation> upDec = A;
	kernels::potrf<lower>(lowDec);
	kernels::potrf<upper>(upDec);
	
	//check that upper diagonal elements are correct and set them to zero
	for (size_t row = 0; row < Dimensions; row++){
		for (size_t col = row+1; col < Dimensions ; col++){
			BOOST_CHECK_CLOSE(lowDec(row, col), A(row,col),1.e-12);
			BOOST_CHECK_CLOSE(upDec(col, row), A(col,row),1.e-12);
			lowDec(row, col) = 0.0;
			upDec(col, row) = 0.0;
		}
	}
	
	//create reconstruction of A
	matrix<double> lowA = prod(lowDec,trans(lowDec));
	matrix<double> upA = prod(trans(upDec),upDec);
	
	//test reconstruction error
	double lowError = max(abs(A - lowA));
	double upError = max(abs(A - upA));
	BOOST_CHECK_SMALL(lowError,1.e-12);
	BOOST_CHECK_SMALL(upError,1.e-12);
	BOOST_CHECK(!(boost::math::isnan)(norm_frobenius(lowA)));//test for nans
	BOOST_CHECK(!(boost::math::isnan)(norm_frobenius(upA)));//test for nans
	
	//check that cholesky decomposition yields the same
	cholesky_decomposition<matrix<double,Orientation> > dec(A);
	for (size_t row = 0; row < Dimensions; row++){
		for (size_t col = 0; col <= row; col++){
			BOOST_CHECK_CLOSE(lowDec(row, col),dec.lower_factor()(row,col),1.e-12);
			BOOST_CHECK_CLOSE(upDec(col, row),dec.upper_factor()(col,row),1.e-12);
		}
	}
}

BOOST_AUTO_TEST_CASE_TEMPLATE(Remora_Pstrf, Orientation,result_orientations) {
	std::size_t Dimensions = 123;
	//first generate a suitable eigenvalue problem matrix A
	matrix<double,Orientation> A = createSymm(Dimensions);
	//calculate Cholesky
	matrix<double,Orientation> lowDec = A;
	matrix<double,Orientation> upDec = A;
	permutation_matrix lowP(Dimensions);
	permutation_matrix upP(Dimensions);
	std::size_t lowRank = kernels::pstrf<lower>(lowDec, lowP);
	std::size_t upRank = kernels::pstrf<upper>(upDec, upP);
	
	//test whether result is full rank
	BOOST_CHECK_EQUAL(lowRank,Dimensions);
	BOOST_CHECK_EQUAL(upRank,Dimensions);
		
	//create reconstruction of A
	matrix<double> lowA = prod(lowDec,trans(lowDec));
	swap_full_inverted(lowP,lowA);
	matrix<double> upA = prod(trans(upDec),upDec);
	swap_full_inverted(upP,upA);
	
	
	//test reconstruction error
	double lowError = max(abs(A - lowA));
	double upError = max(abs(A - upA));
	BOOST_CHECK_SMALL(lowError,1.e-12);
	BOOST_CHECK_SMALL(upError,1.e-12);
	BOOST_CHECK(!(boost::math::isnan)(norm_frobenius(lowA)));//test for nans
	BOOST_CHECK(!(boost::math::isnan)(norm_frobenius(upA)));//test for nans
}

BOOST_AUTO_TEST_CASE_TEMPLATE(Remora_Pstrf_Semi_Definite, Orientation,result_orientations) {
	std::size_t Dimensions = 123;
	std::size_t Rank = 50;
	//first generate a suitable eigenvalue problem matrix A
	matrix<double,Orientation> A = createSymm(Dimensions, Rank);
	//calculate Cholesky
	matrix<double,Orientation> lowDec = A;
	matrix<double,Orientation> upDec = A;
	permutation_matrix lowP(Dimensions);
	permutation_matrix upP(Dimensions);
	std::size_t lowRank = kernels::pstrf<lower>(lowDec, lowP);
	std::size_t upRank = kernels::pstrf<upper>(upDec, upP);
	
	//test whether result is full rank
	BOOST_CHECK_EQUAL(lowRank,Rank);
	BOOST_CHECK_EQUAL(upRank,Rank);
		
	//create reconstruction of A
	matrix<double> lowA = prod(columns(lowDec,0,Rank),trans(columns(lowDec,0,Rank)));
	swap_full_inverted(lowP,lowA);
	matrix<double> upA = prod(trans(rows(upDec,0,Rank)),rows(upDec,0,Rank));
	swap_full_inverted(upP,upA);
	
	
	//test reconstruction error
	double lowError = max(abs(A - lowA));
	double upError = max(abs(A - upA));
	BOOST_CHECK_SMALL(lowError,1.e-12);
	BOOST_CHECK_SMALL(upError,1.e-12);
	BOOST_CHECK(!(boost::math::isnan)(norm_frobenius(lowA)));//test for nans
	BOOST_CHECK(!(boost::math::isnan)(norm_frobenius(upA)));//test for nans
}

BOOST_AUTO_TEST_CASE_TEMPLATE(Remora_Cholesky_Update, Orientation,result_orientations) {
	std::size_t Dimensions = 123;
	//first generate a suitable eigenvalue problem matrix A
	matrix<double,Orientation> A = createSymm(Dimensions);
	vector<double> v(Dimensions);
	for(std::size_t i = 0; i != Dimensions; ++i){
		v(i) = 0.1/Dimensions*i - 0.5;
	}
	double alpha = 0.1;
	double beta = 0.5;
	matrix<double,Orientation> Aupdate = alpha * A + beta * outer_prod(v,v);
	//check that cholesky decomposition yields the same
	cholesky_decomposition<matrix<double,Orientation> > decUpdate(Aupdate);
	cholesky_decomposition<matrix<double,Orientation> > dec(A);
	dec.update(alpha,beta,v);
	
	for (size_t row = 0; row < Dimensions; row++){
		for (size_t col = 0; col <= row; col++){
			BOOST_CHECK_CLOSE(decUpdate.lower_factor()(row,col),dec.lower_factor()(row,col),1.e-11);
		}
	}
}

BOOST_AUTO_TEST_SUITE_END()
