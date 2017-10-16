#define BOOST_TEST_MODULE Remora_Cholesky
#define BOOST_COMPUTE_DEBUG_KERNEL_COMPILATION

#include <remora/dense.hpp>
#include <remora/device_copy.hpp>
#include <remora/kernels/potrf.hpp>
#include <remora/matrix_expression.hpp>
#include <remora/io.hpp>

#include <boost/test/unit_test.hpp>
#include <boost/test/floating_point_comparison.hpp>
#include <boost/mpl/list.hpp>
#include <boost/math/special_functions/fpclassify.hpp>
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
			R(i,i) = 1+0.5/dimensions*i+1;
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
	matrix<double,Orientation, gpu_tag> lowDec_gpu = copy_to_gpu(A);
	matrix<double,Orientation, gpu_tag> upDec_gpu = copy_to_gpu(A);
	kernels::potrf<lower>(lowDec_gpu);
	kernels::potrf<upper>(upDec_gpu);
	matrix<double,Orientation> lowDec = copy_to_cpu(lowDec_gpu);
	matrix<double,Orientation> upDec = copy_to_cpu(upDec_gpu);
	matrix<double,Orientation> lowDec_test = A;
	matrix<double,Orientation> upDec_test = A;
	kernels::potrf<lower>(lowDec_test);
	kernels::potrf<upper>(upDec_test);
	//check that upper diagonal elements are correct and set them to zero
	for (size_t row = 0; row < Dimensions; row++){
		for (size_t col =0; col < Dimensions ; col++){
			BOOST_CHECK_CLOSE(lowDec(row, col), lowDec_test(row,col),1.e-12);
			BOOST_CHECK_CLOSE(upDec(row, col), upDec_test(row,col),1.e-12);
		}
	}
	
}

BOOST_AUTO_TEST_SUITE_END()
