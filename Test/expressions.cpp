#define BOOST_TEST_MODULE Remora_Tensor_Expression
#include <boost/test/unit_test.hpp>
#include <boost/test/floating_point_comparison.hpp>

#include <remora/expressions.hpp>
#include <remora/dense.hpp>
#include <boost/mpl/list.hpp>

using namespace remora;

template<class Operation, class Result>
void checkDenseBlockAssign(
	vector_expression<Operation, cpu_tag> const& op, Result const& result
){
	remora::vector<typename Result::value_type> res1(result.shape()[0],1.0);
	remora::vector<typename Result::value_type> res2(result.shape()[0],1.0);
	op().assign_to(res1);
	op().plus_assign_to(res2);
	
	for(std::size_t i = 0; i != op().shape()[0]; ++i){
		BOOST_CHECK_SMALL(res1(i) - result(i),typename Result::value_type(1.e-7));
		BOOST_CHECK_SMALL(res2(i) - result(i) - 1,typename Result::value_type(1.e-7));
	}
}

template<class Operation, class Result>
void checkDenseBlockAssign(
	matrix_expression<Operation, cpu_tag> const& op, Result const& result
){
	remora::matrix<typename Result::value_type> res1(result.shape(),1.0);
	remora::matrix<typename Result::value_type> res2(result.shape(),1.0);
	op().assign_to(res1);
	op().plus_assign_to(res2);
	
	for(std::size_t i = 0; i != op().shape()[0]; ++i){
		for(std::size_t j = 0; j != op().shape()[1]; ++j){
			BOOST_CHECK_SMALL(res1(i,j) - result(i,j),typename Result::value_type(1.e-7));
			BOOST_CHECK_SMALL(res2(i,j) - result(i,j) - 1,typename Result::value_type(1.e-7));
		}
	}
}

template<class Operation, class Result>
void checkDenseExpressionEquality(
	matrix_expression<Operation, cpu_tag> const& op, Result const& result
){
	BOOST_REQUIRE_EQUAL(op().shape()[0], result.shape()[0]);
	BOOST_REQUIRE_EQUAL(op().shape()[1], result.shape()[1]);
	//check that elements() works
	auto op_elem = op().elements();
	for(std::size_t i = 0; i != op().shape()[0]; ++i){
		for(std::size_t j = 0; j != op().shape()[1]; ++j){
			BOOST_CHECK_CLOSE(result(i,j), op_elem(i,j),1.e-5);
		}
	}
	checkDenseBlockAssign(op,result);
}

template<class Operation, class Result>
void checkDenseExpressionEquality(
	scalar_expression<Operation, cpu_tag> const& op, Result const& result
){
	typedef typename Operation::value_type value_type;
	value_type res_conversion = op();
	scalar<value_type> res_tensor = op;
	BOOST_CHECK_SMALL(res_conversion - result,1.e-7);
	BOOST_CHECK_SMALL(value_type(res_tensor) - result,1.e-7);
}


template<class Operation, class Result>
void checkDenseExpressionEquality(
	vector_expression<Operation, cpu_tag> const& op, Result const& result
){
	BOOST_REQUIRE_EQUAL(op().shape()[0], result.shape()[0]);
	//check that elements() works
	auto op_elem = op().elements();
	for(std::size_t i = 0; i != op().shape()[0]; ++i){
		BOOST_CHECK_CLOSE(result(i), op_elem(i),1.e-5);
	}
	checkDenseBlockAssign(op,result);
}



// template<class M, class D>
// void checkDiagonalMatrix(M const& diagonal, D const& diagonalElements){
	// BOOST_REQUIRE_EQUAL(diagonal.shape()[0],diagonalElements.size());
	// BOOST_REQUIRE_EQUAL(diagonal.shape()[1],diagonalElements.size());
	// auto diag_elem = diagonal.elements();
	// for(std::size_t i = 0; i != diagonalElements.size(); ++i){
		// for(std::size_t j = 0; j != diagonalElements.size(); ++j){
			// if(i != j)
				// BOOST_CHECK_EQUAL(diag_elem(i,j),0);
			// else
				// BOOST_CHECK_EQUAL(diag_elem(i,i),diagonalElements(i));
		// }
		// auto major_begin = diagonal.major_begin(i);
		// BOOST_CHECK_EQUAL(major_begin.index(),i);
		// BOOST_CHECK_EQUAL(std::distance(major_begin, diagonal.major_end(i)),1);
		// BOOST_CHECK_EQUAL(*major_begin,diagonalElements(i));
	// }
	
// }

std::size_t Dimension1 = 25;
std::size_t Dimension2 = 50;

typedef boost::mpl::list<row_major,column_major> result_orientations;

BOOST_AUTO_TEST_SUITE (Remora_expressions_test)

/////////////////////////////////////////////////////////////
//////Vector->Matrix expansions///////
////////////////////////////////////////////////////////////
/*
BOOST_AUTO_TEST_CASE( Remora_dense_matrix_Outer_Prod ){
	std::size_t Dimension2 = 50;
	vector<double> x(Dimension1); 
	vector<double> y(Dimension2); 
	matrix<double> result({Dimension1, Dimension2});
	matrix<double> result2({Dimension1, Dimension2});
	
	
	for (size_t i = 0; i < Dimension1; i++)
		x(i) = i-3.0;
	for (size_t j = 0; j < Dimension2; j++)
		y(j) = 2*j;
	
	for (size_t i = 0; i < Dimension1; i++){
		for (size_t j = 0; j < Dimension2; j++){
			result(i,j)= x(i)*y(j);
			result2(i,j)= 2.0*x(i)*y(j) +6*result(i,j);
		}
	}

	checkDenseExpressionEquality(outer_prod(x,y),result);
	checkDenseExpressionEquality(2.0 * ( outer_prod( x, y ) + 3.0 * result),result2);
}

BOOST_AUTO_TEST_CASE( Remora_Diagonal_Matrix ){
	vector<double> diagonalElements(Dimension1);
	matrix<double> result(Dimension1,Dimension2,0.0);
	for(std::size_t i = 0; i != Dimension1; ++i){
		diagonalElements(i) = i;
		result(i,i) = i;
	}
	
	diagonal_matrix<vector<double> > diagonal(diagonalElements);
	checkDiagonalMatrix(diagonal,diagonalElements);
}

BOOST_AUTO_TEST_CASE( Remora_Identity_Matrix ){
	vector<double> diagonalElements(Dimension1);
	for(std::size_t i = 0; i != Dimension1; ++i)
		diagonalElements(i) = 1;
	
	identity_matrix<double > diagonal(Dimension1);
	checkDiagonalMatrix(diagonal,diagonalElements);
}*/



BOOST_AUTO_TEST_CASE_TEMPLATE( Remora_dense_matrix_broadcast, Axis, result_orientations ){
	matrix<int, Axis> x({Dimension1, Dimension2}); 
	tensorN<int, 4> result({Dimension1, 10,  Dimension2, 8});
	
	for (size_t i = 0; i < Dimension1; i++){
		for (size_t j = 0; j < 10; j++){
			for(std::size_t k = 0; k != Dimension2; ++k){
				x(i,k) = i-3+k;
				for(std::size_t l = 0; l != 8; ++l){
					result(i, j, k, l)= x(i,k);
				}
			}
		}
	}
	
	auto op =  broadcast(x, ax::same, 10, ax::same, 8);
	BOOST_REQUIRE_EQUAL(op.shape().size(), 4);
	BOOST_REQUIRE_EQUAL(op.shape()[0], Dimension1);
	BOOST_REQUIRE_EQUAL(op.shape()[1], 10);
	BOOST_REQUIRE_EQUAL(op.shape()[2], Dimension2);
	BOOST_REQUIRE_EQUAL(op.shape()[3], 8);
	auto elements = op.elements();
	tensorN<int, 4> op_elem_assign = op; //difference to above: tests assignment using identity permute and slice
	tensor<int,axis<2,0,1, 3>, cpu_tag > op_perm_elem_assignf = op; //permute moves broadcasted dimension to the front
	tensor<int,axis<0,2,1, 3>, cpu_tag > op_perm_elem_assignb = op; //permute moves broadcasted dimension to the back
	tensor<int,axis<3,1,2, 0>, cpu_tag > op_perm_elem_assignd = op; //permute moves both bc-dims to the front, slice will remove them.
	tensorN<int,4> op_block_assign(op.shape(), 0);
	tensorN<int,4> op_block_plus_assign(op.shape(), 1);
	op.assign_to(op_block_assign);
	op.plus_assign_to(op_block_plus_assign);
	
	for (size_t i = 0; i < Dimension1; i++){
		for (size_t j = 0; j < 10; j++){
			for(std::size_t k = 0; k != Dimension2; ++k){
				for(std::size_t l = 0; l != 8; ++l){
					BOOST_CHECK_EQUAL(result(i, j, k, l), elements(i, j, k, l));
					BOOST_CHECK_EQUAL(result(i, j, k, l), op_elem_assign(i, j, k, l));
					BOOST_CHECK_EQUAL(result(i, j, k, l), op_perm_elem_assignf(i, j, k, l));
					BOOST_CHECK_EQUAL(result(i, j, k, l), op_perm_elem_assignb(i, j, k, l));
					BOOST_CHECK_EQUAL(result(i, j, k, l), op_perm_elem_assignd(i, j, k, l));
					BOOST_CHECK_EQUAL(result(i, j, k, l), op_block_assign(i, j, k, l));
					BOOST_CHECK_EQUAL(result(i, j, k, l) + 1, op_block_plus_assign(i, j, k, l));
				}
			}
		}
	}
}

BOOST_AUTO_TEST_CASE_TEMPLATE( Remora_dense_matrix_broadcast_split, Axis, result_orientations )
{
	matrix<int, Axis> x({Dimension1, Dimension2}); 
	tensorN<int, 4> result({Dimension1, 10,  Dimension2, 8});
	
	for (size_t i = 0; i < Dimension1; i++){
		for (size_t j = 0; j < 10; j++){
			for(std::size_t k = 0; k != Dimension2; ++k){
				x(i,k) = i-3+k;
				for(std::size_t l = 0; l != 8; ++l){
					result(i, j, k, l)= x(i,k);
				}
			}
		}
	}
	
	auto op =  broadcast(x, ax::same, 10, ax::same, 8);
	//split check
	auto op_split1 = reshape(op, ax::same, ax::same, ax::split<2>(10,5), ax::same); 
	auto op_split2 = reshape(op, ax::same, ax::same, ax::same, ax::split<2>(2,4)); 
	auto split1_elem = op_split1.elements();
	auto split2_elem = op_split2.elements();
	BOOST_REQUIRE_EQUAL(op_split1.shape().size(), 5);
	BOOST_REQUIRE_EQUAL(op_split1.shape()[0], Dimension1);
	BOOST_REQUIRE_EQUAL(op_split1.shape()[1], 10);
	BOOST_REQUIRE_EQUAL(op_split1.shape()[2], 10);
	BOOST_REQUIRE_EQUAL(op_split1.shape()[3], 5);
	BOOST_REQUIRE_EQUAL(op_split1.shape()[4], 8);
	BOOST_REQUIRE_EQUAL(op_split2.shape().size(), 5);
	BOOST_REQUIRE_EQUAL(op_split2.shape()[0], Dimension1);
	BOOST_REQUIRE_EQUAL(op_split2.shape()[1], 10);
	BOOST_REQUIRE_EQUAL(op_split2.shape()[2], Dimension2);
	BOOST_REQUIRE_EQUAL(op_split2.shape()[3], 2);
	BOOST_REQUIRE_EQUAL(op_split2.shape()[4], 4);
	
	for (size_t i = 0; i < Dimension1; i++){
		for (size_t j = 0; j < 10; j++){
			for(std::size_t k = 0; k != Dimension2; ++k){
				for(std::size_t l = 0; l != 8; ++l){
					BOOST_CHECK_EQUAL(result(i, j, k, l), split1_elem(i, j, k / 5, k % 5, l));
					BOOST_CHECK_EQUAL(result(i, j, k, l), split2_elem(i, j, k, l / 4, l % 4));
				}
			}
		}
	}
}

BOOST_AUTO_TEST_CASE_TEMPLATE( Remora_dense_matrix_broadcast_slice, Axis, result_orientations )
{
	matrix<int, Axis> x({Dimension1, Dimension2}); 
	tensorN<int, 4> result({Dimension1, 10,  Dimension2, 8});
	
	for (size_t i = 0; i < Dimension1; i++){
		for (size_t j = 0; j < 10; j++){
			for(std::size_t k = 0; k != Dimension2; ++k){
				x(i,k) = i-3+k;
				for(std::size_t l = 0; l != 8; ++l){
					result(i, j, k, l)= x(i,k);
				}
			}
		}
	}
	
	{
		auto op =  broadcast(x, ax::same, 10, ax::same, 8);
		auto op_slice = slice(op, 3, ax::same, ax::same, ax::same); //slice a matrix-dimension 
		auto slice_elem = op_slice.elements();

		BOOST_REQUIRE_EQUAL(op_slice.shape()[0], 10);
		BOOST_REQUIRE_EQUAL(op_slice.shape()[1], Dimension2);
		BOOST_REQUIRE_EQUAL(op_slice.shape()[2], 8);
		
		for (size_t j = 0; j < 10; j++){
			for(std::size_t k = 0; k != Dimension2; ++k){
				for(std::size_t l = 0; l != 8; ++l){
					BOOST_CHECK_EQUAL(result(3, j, k, l), slice_elem(j, k, l));
				}
			}
		}
	}
	{
		auto op =  broadcast(x, ax::same, 10, ax::same, 8);
		auto op_slice = slice(op, ax::same, 3, ax::same, ax::same); //slice a bc-dimension
		auto slice_elem = op_slice.elements();

		BOOST_REQUIRE_EQUAL(op_slice.shape()[0], Dimension1);
		BOOST_REQUIRE_EQUAL(op_slice.shape()[1], Dimension2);
		BOOST_REQUIRE_EQUAL(op_slice.shape()[2], 8);
		
		for (size_t i = 0; i < Dimension1; i++){
			for(std::size_t k = 0; k != Dimension2; ++k){
				for(std::size_t l = 0; l != 8; ++l){
					BOOST_CHECK_EQUAL(result(i, 3, k, l), slice_elem(i, k, l));
				}
			}
		}
	}
	{
		auto op =  broadcast(x, ax::same, 10, ax::same, 8);
		scalar_tensor<int, axis<0,1>, cpu_tag> op_slice = slice(op, 3, ax::same, 5, ax::same); //slice until all matrix dimensions are gone(result should be a scalar tensor)
		auto slice_elem = op_slice.elements();

		BOOST_REQUIRE_EQUAL(op_slice.shape()[0], 10);
		BOOST_REQUIRE_EQUAL(op_slice.shape()[1], 8);
		
		for (size_t j = 0; j < 10; j++){
			for(std::size_t l = 0; l != 8; ++l){
				BOOST_CHECK_EQUAL(result(3, j, 5, l), slice_elem(j, l));
			}
		}
	}
	{
		auto op =  broadcast(x, ax::same, 10, ax::same, 8);
		typename matrix<int, Axis>::const_closure_type op_slice = slice(op, ax::same, 3, ax::same, 5); //slice until all bc dimensions are gone (result should be original matrix)
		auto slice_elem = op_slice.elements();

		BOOST_REQUIRE_EQUAL(op_slice.shape()[0], Dimension1);
		BOOST_REQUIRE_EQUAL(op_slice.shape()[1], Dimension2);
		for (size_t i = 0; i < Dimension1; i++){
			for(std::size_t k = 0; k != Dimension2; ++k){
				BOOST_CHECK_EQUAL(result(i, 3, k, 5), slice_elem(i, k));
			}
		}
		
	}
}



BOOST_AUTO_TEST_CASE_TEMPLATE( Remora_dense_matrix_broadcast_permute, Axis, result_orientations )
{
	matrix<int, Axis> x({Dimension1, Dimension2}); 
	tensorN<int, 4> result({Dimension1, 10,  Dimension2, 8});
	
	for (size_t i = 0; i < Dimension1; i++){
		for (size_t j = 0; j < 10; j++){
			for(std::size_t k = 0; k != Dimension2; ++k){
				x(i,k) = i-3+k;
				for(std::size_t l = 0; l != 8; ++l){
					result(i, j, k, l)= x(i,k);
				}
			}
		}
	}
	{
		auto op =  broadcast(x, ax::same, 10, ax::same, 8);
		typedef typename decltype(op)::axis op_axis;
		tensor_broadcast<
			typename matrix<int, Axis >::const_closure_type, 
			typename op_axis::template permute_t<0,3,1,2>,
			integer_list<bool,0,1,1,0>
		> op_permute = permute(op, axis<0,3,1,2>()); 
		auto permute_elem = op_permute.elements();
		BOOST_REQUIRE_EQUAL(op_permute.shape().size(), 4);
		BOOST_REQUIRE_EQUAL(op_permute.shape()[0], Dimension1);
		BOOST_REQUIRE_EQUAL(op_permute.shape()[1], 8);
		BOOST_REQUIRE_EQUAL(op_permute.shape()[2], 10);
		BOOST_REQUIRE_EQUAL(op_permute.shape()[3], Dimension2);
		
		for (size_t i = 0; i < Dimension1; i++){
			for (size_t j = 0; j < 10; j++){
				for(std::size_t k = 0; k != Dimension2; ++k){
					for(std::size_t l = 0; l != 8; ++l){
						BOOST_CHECK_EQUAL(result(i, j, k, l), permute_elem(i, l, j, k));
					}
				}
			}
		}
	}
	//also permutes the inner tensor
	{
		auto op =  broadcast(x, ax::same, 10, ax::same, 8);
		typedef typename decltype(op)::axis op_axis;
		tensor_broadcast<
			typename matrix<int, typename Axis::template permute_t<1,0> >::const_closure_type, 
			typename op_axis::template permute_t<2,3,1,0>,
			integer_list<bool,0,1,1,0>
		> op_permute = permute(op, axis<2,3,1,0>()); 
		auto permute_elem = op_permute.elements();
		BOOST_REQUIRE_EQUAL(op_permute.shape().size(), 4);
		BOOST_REQUIRE_EQUAL(op_permute.shape()[0], Dimension2);
		BOOST_REQUIRE_EQUAL(op_permute.shape()[1], 8);
		BOOST_REQUIRE_EQUAL(op_permute.shape()[2], 10);
		BOOST_REQUIRE_EQUAL(op_permute.shape()[3], Dimension1);
		
		for (size_t i = 0; i < Dimension1; i++){
			for (size_t j = 0; j < 10; j++){
				for(std::size_t k = 0; k != Dimension2; ++k){
					for(std::size_t l = 0; l != 8; ++l){
						BOOST_CHECK_EQUAL(result(i, j, k, l), permute_elem(k, l, j, i));
					}
				}
			}
		}
	}
}

BOOST_AUTO_TEST_CASE_TEMPLATE( Remora_dense_matrix_broadcast_diagonal, Axis, result_orientations )
{
	matrix<int, Axis> x({Dimension1, Dimension2}); 
	
	for (size_t i = 0; i < Dimension1; i++){
		for (size_t j = 0; j < 10; j++){
			for(std::size_t k = 0; k != Dimension2; ++k){
				x(i,k) = i-3+k;
			}
		}
	}
	//case 1: both diag dimensions are broadcasted
	{
		auto op =  broadcast(x, ax::same, ax::same, 10, 8);
		tensor_broadcast<
			typename matrix<int, Axis >::const_closure_type, 
			axis<Axis::template element_v<0>, Axis::template element_v<1>, 2>,
			integer_list<bool,0,0,1>
		> op_diag =  diag(op, 3);
		auto op_diag2 =  diag(op, axis_set<3,2>(), -3);
		auto op_elem = op_diag.elements();
		auto op_elem2 = op_diag2.elements();
		BOOST_REQUIRE_EQUAL(op_diag.shape()[0], Dimension1);
		BOOST_REQUIRE_EQUAL(op_diag.shape()[1], Dimension2);
		BOOST_REQUIRE_EQUAL(op_diag.shape()[2], 5);
		
		BOOST_REQUIRE_EQUAL(op_diag2.shape()[0], Dimension1);
		BOOST_REQUIRE_EQUAL(op_diag2.shape()[1], Dimension2);
		BOOST_REQUIRE_EQUAL(op_diag2.shape()[2], 5);
		
		for (size_t i = 0; i < Dimension1; i++){
			for (size_t j = 0; j < Dimension2; j++){
				for(std::size_t k = 0; k != 5; ++k){
					BOOST_CHECK_EQUAL(x(i, j), op_elem(i, j, k));
					BOOST_CHECK_EQUAL(x(i, j), op_elem2(i, j, k));
				}
			}
		}
	}
	//case 2: both diag dimensions are of a matrix
	{
		auto op =  broadcast(x, 10, 8, ax::same, ax::same);
		auto op_diag =  diag(op, 10);
		auto op_diag2 =  diag(op, axis_set<3,2>(), -10);
		auto op_elem = op_diag.elements();
		auto op_elem2 = op_diag2.elements();
		BOOST_REQUIRE_EQUAL(op_diag.shape()[0], 10);
		BOOST_REQUIRE_EQUAL(op_diag.shape()[1], 8);
		BOOST_REQUIRE_EQUAL(op_diag.shape()[2], 25);
		BOOST_REQUIRE_EQUAL(op_diag2.shape()[0], 10);
		BOOST_REQUIRE_EQUAL(op_diag2.shape()[1], 8);
		BOOST_REQUIRE_EQUAL(op_diag2.shape()[2], 25);
		for (size_t i = 0; i < 10; i++){
			for (size_t j = 0; j < 8; j++){
				for(std::size_t k = 0; k != 20; ++k){
					BOOST_CHECK_EQUAL(x(k, k + 10), op_elem(i, j, k));
					BOOST_CHECK_EQUAL(x(k, k + 10), op_elem2(i, j, k));
				}
			}
		}
	}
	
	//case 3: second diag dimension is of a matrix
	{
		auto op =  broadcast(x, 10, ax::same, 8, ax::same);
		auto op_diag =  diag(op, -3);
		auto op_diag2 =  diag(op, axis_set<3,2>(), 3);
		auto op_elem = op_diag.elements();
		auto op_elem2 = op_diag2.elements();
		BOOST_REQUIRE_EQUAL(op_diag.shape()[0], 10);
		BOOST_REQUIRE_EQUAL(op_diag.shape()[1], Dimension1);
		BOOST_REQUIRE_EQUAL(op_diag.shape()[2], 5);
		BOOST_REQUIRE_EQUAL(op_diag2.shape()[0], 10);
		BOOST_REQUIRE_EQUAL(op_diag2.shape()[1], Dimension1);
		BOOST_REQUIRE_EQUAL(op_diag2.shape()[2], 5);
		for (size_t i = 0; i < 10; i++){
			for (size_t j = 0; j < Dimension1; j++){
				for(std::size_t k = 0; k != 5; ++k){
					BOOST_CHECK_EQUAL(x(j,k), op_elem(i, j, k));
					BOOST_CHECK_EQUAL(x(j,k), op_elem2(i, j, k));
				}
			}
		}
	}
	//case 4: first diag dimension is of a matrix
	{
		auto op =  broadcast(x, 10, ax::same, ax::same, 8);
		auto op_diag =  diag(op, -3);
		auto op_diag2 =  diag(op, axis_set<3,2>(), 3);
		auto op_elem = op_diag.elements();
		auto op_elem2 = op_diag2.elements();
		BOOST_REQUIRE_EQUAL(op_diag.shape()[0], 10);
		BOOST_REQUIRE_EQUAL(op_diag.shape()[1], Dimension1);
		BOOST_REQUIRE_EQUAL(op_diag.shape()[2], 8);
		BOOST_REQUIRE_EQUAL(op_diag2.shape()[0], 10);
		BOOST_REQUIRE_EQUAL(op_diag2.shape()[1], Dimension1);
		BOOST_REQUIRE_EQUAL(op_diag2.shape()[2], 8);
		for (size_t i = 0; i < 10; i++){
			for (size_t j = 0; j < Dimension1; j++){
				for(std::size_t k = 0; k != 8; ++k){
					BOOST_CHECK_EQUAL(x(j,k), op_elem(i, j, k));
					BOOST_CHECK_EQUAL(x(j,k), op_elem2(i, j, k));
				}
			}
		}
	}
}

//basic functionality check for scalars. individual operations are tested below for matrices
BOOST_AUTO_TEST_CASE( Remora_dense_scalar_general){
	scalar<double> x(23.0);
	scalar<double> y(5.0);
	vector<double> z(3,5.0);
	vector<double> x_vec(3,5.0);
	
	//scalar
	scalar_tensor<double,axis<>, cpu_tag> s({},5.0);
	checkDenseExpressionEquality(s,5.0);
	//unary
	checkDenseExpressionEquality(max(x, 5.0),23.0);
	checkDenseExpressionEquality(max(5.0, x),23.0);
	checkDenseExpressionEquality(max(x_vec, z),x_vec);
	checkDenseExpressionEquality(max(z, x_vec),x_vec);
	//binary
	checkDenseExpressionEquality(max(x,y),23.0);
	
	//scalar multiply
	checkDenseExpressionEquality(x*y,23.0*5.0);
	checkDenseExpressionEquality(x*5.0,23.0*5.0);
	checkDenseExpressionEquality(23*y,23.0*5.0);
	
	//vector-scalar multiply
	vector<double> mul_res(3,5.0*23.0);
	checkDenseExpressionEquality(x*z,mul_res);
	checkDenseExpressionEquality(z*x,mul_res);
	
	
	//scalar add
	checkDenseExpressionEquality(x+y,23.0+5.0);
	checkDenseExpressionEquality(x+5.0,23.0+5.0);
	checkDenseExpressionEquality(23+y,23.0+5.0);
	
	//vector-scalar add
	vector<double> add_res(3,5.0+23.0);
	checkDenseExpressionEquality(x+z,add_res);
	checkDenseExpressionEquality(z+x,add_res);
	
	//scalar subtract
	checkDenseExpressionEquality(x-y,23.0-5.0);
	checkDenseExpressionEquality(x-5.0,23.0-5.0);
	checkDenseExpressionEquality(23.0-y,23.0-5.0);
	
	//vector-scalar subtract
	vector<double> sub_res1(3,23.0-5.0);
	vector<double> sub_res2(3,5.0-23.0);
	checkDenseExpressionEquality(x-z,sub_res1);
	checkDenseExpressionEquality(z-x,sub_res2);
	
	//division
	checkDenseExpressionEquality(safe_div(x,y, 5.0), 23.0/5.0);
	checkDenseExpressionEquality(safe_div(x,y, y), 23.0/5.0);
	checkDenseExpressionEquality(x/y, 23.0/5.0);

	vector<double> div_res1(3,5.0/23.0);	
	vector<double> div_res2(3,23.0/5.0);
	checkDenseExpressionEquality(z/x, div_res1);
	checkDenseExpressionEquality(x/z, div_res2);
	
	
	//broadcast
	matrix<double> res({10, 8},23.0*5.0); 
	auto op =  broadcast(x*y, 10, 8);
	checkDenseExpressionEquality(op, res);
}



/////////////////////////////////////////////////////////////
//////UNARY TRANSFORMATIONS///////
////////////////////////////////////////////////////////////

BOOST_AUTO_TEST_CASE_TEMPLATE( Remora_dense_matrix_Unary_Minus, Axis, result_orientations )
{
	matrix<double, Axis> x({Dimension1, Dimension2}); 
	matrix<double> result({Dimension1, Dimension2});
	
	for (size_t i = 0; i < Dimension1; i++){
		for (size_t j = 0; j < Dimension2; j++){
			x(i,j) = i-3.0+j;
			result(i,j)= -x(i,j);
		}
	}
	checkDenseExpressionEquality(-x,result);
}
BOOST_AUTO_TEST_CASE_TEMPLATE( Remora_dense_matrix_Scalar_Multiply, Axis, result_orientations )
{
	matrix<double, Axis> x({Dimension1, Dimension2}); 
	matrix<double> result({Dimension1, Dimension2});
	
	for (size_t i = 0; i < Dimension1; i++){
		for (size_t j = 0; j < Dimension2; j++){
			x(i,j) = i-3.0+j;
			result(i,j)= 5.0* x(i,j);
		}
	}
	
	checkDenseExpressionEquality(5.0*x,result);
	checkDenseExpressionEquality(x*5.0,result);
}
BOOST_AUTO_TEST_CASE_TEMPLATE( Remora_dense_matrix_Scalar_Add, Axis, result_orientations )
{
	matrix<double, Axis> x({Dimension1, Dimension2}); 
	matrix<double> result({Dimension1, Dimension2});
	
	for (size_t i = 0; i < Dimension1; i++){
		for (size_t j = 0; j < Dimension2; j++){
			x(i,j) = i-3.0+j;
			result(i,j)= 5.0 + x(i,j);
		}
	}
	checkDenseExpressionEquality(5.0+x,result);
	checkDenseExpressionEquality(x+5.0,result);
}
BOOST_AUTO_TEST_CASE_TEMPLATE( Remora_dense_matrix_Scalar_Subtract, Axis, result_orientations )
{
	matrix<double, Axis> x({Dimension1, Dimension2}); 
	matrix<double> result1({Dimension1, Dimension2});
	matrix<double> result2({Dimension1, Dimension2});
	
	for (size_t i = 0; i < Dimension1; i++){
		for (size_t j = 0; j < Dimension2; j++){
			x(i,j) = i-3.0+j;
			result1(i,j)= 5.0 - x(i,j);
			result2(i,j)= x(i,j) - 5.0;
		}
	}
	checkDenseExpressionEquality(5.0- x,result1);
	checkDenseExpressionEquality(x - 5.0,result2);
}
BOOST_AUTO_TEST_CASE_TEMPLATE( Remora_dense_matrix_Scalar_Div, Axis, result_orientations )
{
	matrix<double, Axis> x({Dimension1, Dimension2}); 
	matrix<double> result({Dimension1, Dimension2});
	
	for (size_t i = 0; i < Dimension1; i++){
		for (size_t j = 0; j < Dimension2; j++){
			x(i,j) = i-3.0+j;
			result(i,j)= x(i,j)/5.0;
		}
	}
	checkDenseExpressionEquality(x/5.0f,result);
}
BOOST_AUTO_TEST_CASE_TEMPLATE( Remora_dense_matrix_Scalar_elem_inv, Axis, result_orientations )
{
	matrix<double, Axis> x({Dimension1, Dimension2}); 
	matrix<double> result({Dimension1, Dimension2});
	
	for (size_t i = 0; i < Dimension1; i++){
		for (size_t j = 0; j < Dimension2; j++){
			x(i,j) = i+j+3.0;
			result(i,j)= 1.0/x(i,j);
		}
	}
	checkDenseExpressionEquality(elem_inv(x),result);
}
BOOST_AUTO_TEST_CASE_TEMPLATE( Remora_dense_matrix_Abs, Axis, result_orientations )
{
	matrix<double, Axis> x({Dimension1, Dimension2}); 
	matrix<double> result({Dimension1, Dimension2});
	
	for (size_t i = 0; i < Dimension1; i++){
		for (size_t j = 0; j < Dimension2; j++){
			x(i,j) = i-3.0-j;
			result(i,j)= std::abs(x(i,j));
		}
	}
	checkDenseExpressionEquality(abs(x),result);
}
BOOST_AUTO_TEST_CASE_TEMPLATE( Remora_dense_matrix_Sqr, Axis, result_orientations )
{
	matrix<double, Axis> x({Dimension1, Dimension2}); 
	matrix<double> result({Dimension1, Dimension2});
	for (size_t i = 0; i < Dimension1; i++){
		for (size_t j = 0; j < Dimension2; j++){
			x(i,j) = i-3.0-j;
			result(i,j)= x(i,j) * x(i,j);
		}
	}
	checkDenseExpressionEquality(sqr(x),result);
}
BOOST_AUTO_TEST_CASE_TEMPLATE( Remora_dense_matrix_Sqrt, Axis, result_orientations )
{
	matrix<double, Axis> x({Dimension1, Dimension2}); 
	matrix<double> result({Dimension1, Dimension2});
	
	for (size_t i = 0; i < Dimension1; i++){
		for (size_t j = 0; j < Dimension2; j++){
			x(i,j) = i+j+3.0;
			result(i,j)= std::sqrt(x(i,j));
		}
	}
	checkDenseExpressionEquality(sqrt(x),result);
}
BOOST_AUTO_TEST_CASE_TEMPLATE( Remora_dense_matrix_Cbrt, Axis, result_orientations )
{
	matrix<double, Axis> x({Dimension1, Dimension2}); 
	matrix<double> result({Dimension1, Dimension2});
	
	for (size_t i = 0; i < Dimension1; i++){
		for (size_t j = 0; j < Dimension2; j++){
			x(i,j) = i+j+3.0;
			result(i,j)= std::cbrt(x(i,j));
		}
	}
	checkDenseExpressionEquality(cbrt(x),result);
}
BOOST_AUTO_TEST_CASE_TEMPLATE( Remora_dense_matrix_Exp, Axis, result_orientations )
{
	matrix<double, Axis> x({Dimension1, Dimension2}); 
	matrix<double> result({Dimension1, Dimension2});
	
	for (size_t i = 0; i < Dimension1; i++){
		for (size_t j = 0; j < Dimension2; j++){
			x(i,j) = 0.01*(i-3.0-double(j));
			result(i,j)= std::exp(x(i,j));
		}
	}
	checkDenseExpressionEquality(exp(x),result);
}
BOOST_AUTO_TEST_CASE_TEMPLATE( Remora_dense_matrix_Log, Axis, result_orientations )
{
	matrix<double, Axis> x({Dimension1, Dimension2}); 
	matrix<double> result({Dimension1, Dimension2});
	
	for (size_t i = 0; i < Dimension1; i++){
		for (size_t j = 0; j < Dimension2; j++){
			x(i,j) = i+j+3.0;
			result(i,j)= std::log(x(i,j));
		}
	}
	checkDenseExpressionEquality(log(x),result);
}
BOOST_AUTO_TEST_CASE_TEMPLATE( Remora_dense_matrix_sin, Axis, result_orientations )
{
	matrix<double, Axis> x({Dimension1, Dimension2}); 
	matrix<double> result({Dimension1, Dimension2});
	
	for (size_t i = 0; i < Dimension1; i++){
		for (size_t j = 0; j < Dimension2; j++){
			x(i,j) = double(i)/Dimension1-double(j)/Dimension2;
			result(i,j)= std::sin(x(i,j));
		}
	}
	checkDenseExpressionEquality(sin(x),result);
}

BOOST_AUTO_TEST_CASE_TEMPLATE( Remora_dense_matrix_cos, Axis, result_orientations )
{
	matrix<double, Axis> x({Dimension1, Dimension2}); 
	matrix<double> result({Dimension1, Dimension2});
	
	for (size_t i = 0; i < Dimension1; i++){
		for (size_t j = 0; j < Dimension2; j++){
			x(i,j) = double(i)/Dimension1-double(j)/Dimension2;
			result(i,j)= std::cos(x(i,j));
		}
	}
	checkDenseExpressionEquality(cos(x),result);
}

BOOST_AUTO_TEST_CASE_TEMPLATE( Remora_dense_matrix_tan, Axis, result_orientations )
{
	matrix<double, Axis> x({Dimension1, Dimension2}); 
	matrix<double> result({Dimension1, Dimension2});
	
	for (size_t i = 0; i < Dimension1; i++){
		for (size_t j = 0; j < Dimension2; j++){
			x(i,j) = double(i)/Dimension1-double(j)/Dimension2;
			result(i,j)= std::tan(x(i,j));
		}
	}
	checkDenseExpressionEquality(tan(x),result);
}

BOOST_AUTO_TEST_CASE_TEMPLATE( Remora_dense_matrix_asin, Axis, result_orientations )
{
	matrix<double, Axis> x({Dimension1, Dimension2}); 
	matrix<double> result({Dimension1, Dimension2});
	
	for (size_t i = 0; i < Dimension1; i++){
		for (size_t j = 0; j < Dimension2; j++){
			x(i,j) = double(i)/Dimension1-double(j)/Dimension2;
			result(i,j)= std::asin(x(i,j));
		}
	}
	checkDenseExpressionEquality(asin(x),result);
}

BOOST_AUTO_TEST_CASE_TEMPLATE( Remora_dense_matrix_acos, Axis, result_orientations )
{
	matrix<double, Axis> x({Dimension1, Dimension2}); 
	matrix<double> result({Dimension1, Dimension2});
	
	for (size_t i = 0; i < Dimension1; i++){
		for (size_t j = 0; j < Dimension2; j++){
			x(i,j) = double(i)/Dimension1-double(j)/Dimension2;
			result(i,j)= std::acos(x(i,j));
		}
	}
	checkDenseExpressionEquality(acos(x),result);
}

BOOST_AUTO_TEST_CASE_TEMPLATE( Remora_dense_matrix_atan, Axis, result_orientations )
{
	matrix<double, Axis> x({Dimension1, Dimension2}); 
	matrix<double> result({Dimension1, Dimension2});
	
	for (size_t i = 0; i < Dimension1; i++){
		for (size_t j = 0; j < Dimension2; j++){
			x(i,j) = double(i)/Dimension1-double(j)/Dimension2;
			result(i,j)= std::atan(x(i,j));
		}
	}
	checkDenseExpressionEquality(atan(x),result);
}

BOOST_AUTO_TEST_CASE_TEMPLATE( Remora_dense_matrix_erf, Axis, result_orientations )
{
	matrix<double, Axis> x({Dimension1, Dimension2}); 
	matrix<double> result({Dimension1, Dimension2});
	
	for (size_t i = 0; i < Dimension1; i++){
		for (size_t j = 0; j < Dimension2; j++){
			x(i,j) = double(i)/Dimension1-double(j)/Dimension2;
			result(i,j)= std::erf(x(i,j));
		}
	}
	checkDenseExpressionEquality(erf(x),result);
}

BOOST_AUTO_TEST_CASE_TEMPLATE( Remora_dense_matrix_erfc, Axis, result_orientations )
{
	matrix<double, Axis> x({Dimension1, Dimension2}); 
	matrix<double> result({Dimension1, Dimension2});
	
	for (size_t i = 0; i < Dimension1; i++){
		for (size_t j = 0; j < Dimension2; j++){
			x(i,j) = double(i)/Dimension1-double(j)/Dimension2;
			result(i,j)= std::erfc(x(i,j));
		}
	}
	checkDenseExpressionEquality(erfc(x),result);
}

BOOST_AUTO_TEST_CASE_TEMPLATE( Remora_dense_matrix_Tanh, Axis, result_orientations )
{
	matrix<double, Axis> x({Dimension1, Dimension2}); 
	matrix<double> result({Dimension1, Dimension2});
	
	for (size_t i = 0; i < Dimension1; i++){
		for (size_t j = 0; j < Dimension2; j++){
			x(i,j) = i-3.0-j;
			result(i,j)= std::tanh(x(i,j));
		}
	}
	checkDenseExpressionEquality(tanh(x),result);
}
BOOST_AUTO_TEST_CASE_TEMPLATE( Remora_dense_matrix_Sigmoid, Axis, result_orientations )
{
	matrix<double, Axis> x({Dimension1, Dimension2}); 
	matrix<double> result({Dimension1, Dimension2});
	
	for (size_t i = 0; i < Dimension1; i++){
		for (size_t j = 0; j < Dimension2; j++){
			x(i,j) = 0.01*(i-3.0-j);
			result(i,j)= 1.0/(1.0+std::exp(-x(i,j)));
		}
	}
	checkDenseExpressionEquality(sigmoid(x),result);
}
BOOST_AUTO_TEST_CASE_TEMPLATE( Remora_dense_matrix_SoftPlus, Axis, result_orientations )
{
	matrix<double, Axis> x({Dimension1, Dimension2}); 
	matrix<double> result({Dimension1, Dimension2});
	
	for (size_t i = 0; i < Dimension1; i++){
		for (size_t j = 0; j < Dimension2; j++){
			x(i,j) = i-3.0-j;
			result(i,j)= std::log(1+std::exp(x(i,j)));
		}
	}
	checkDenseExpressionEquality(softPlus(x),result);
}
BOOST_AUTO_TEST_CASE_TEMPLATE( Remora_dense_matrix_Pow, Axis, result_orientations )
{
	matrix<double, Axis> x({Dimension1, Dimension2}); 
	matrix<double> result({Dimension1, Dimension2});
	
	for (size_t i = 0; i < Dimension1; i++){
		for (size_t j = 0; j < Dimension2; j++){
			x(i,j) = 0.001*(i+j+2.0);
			result(i,j)= std::pow(x(i,j),3.2);
		}
	}
	checkDenseExpressionEquality(pow(x,3.2),result);
}

BOOST_AUTO_TEST_CASE_TEMPLATE( Remora_dense_matrix_Unary_Min, Axis, result_orientations )
{
	matrix<double, Axis> x({Dimension1, Dimension2}); 
	matrix<double> result({Dimension1, Dimension2});
	
	for (size_t i = 0; i < Dimension1; i++){
		for (size_t j = 0; j < Dimension2; j++){
			x(i,j) = i-1.0*j;
			result(i,j)= std::min(x(i,j),5.0);
		}
	}
	checkDenseExpressionEquality(min(x,5.0),result);
	checkDenseExpressionEquality(min(5.0,x),result);
}
BOOST_AUTO_TEST_CASE_TEMPLATE( Remora_dense_matrix_Unary_Max, Axis, result_orientations )
{
	matrix<double, Axis> x({Dimension1, Dimension2}); 
	matrix<double> result({Dimension1, Dimension2});
	
	for (size_t i = 0; i < Dimension1; i++){
		for (size_t j = 0; j < Dimension2; j++){
			x(i,j) = i-1.0*j;
			result(i,j)= std::max(x(i,j),5.0);
		}
	}
	checkDenseExpressionEquality(max(x,5.0),result);
	checkDenseExpressionEquality(max(5.0,x),result);
}

/////////////////////////////////////////////////////
///////BINARY OPERATIONS//////////
/////////////////////////////////////////////////////

BOOST_AUTO_TEST_CASE_TEMPLATE( Remora_dense_matrix_Binary_Plus, Axis, result_orientations )
{
	matrix<double, Axis> x({Dimension1, Dimension2}); 
	matrix<double, Axis> y({Dimension1, Dimension2}); 
	matrix<double> result({Dimension1, Dimension2});
	
	for (size_t i = 0; i < Dimension1; i++){
		for (size_t j = 0; j < Dimension2; j++){
			x(i,j) = i-3.0-j;
			y(i,j) = i+j+Dimension1;
			result(i,j)= x(i,j)+y(i,j);
		}
	}
	checkDenseExpressionEquality(x+y,result);
}
BOOST_AUTO_TEST_CASE_TEMPLATE( Remora_dense_matrix_Binary_Minus, Axis, result_orientations )
{
	matrix<double, Axis> x({Dimension1, Dimension2}); 
	matrix<double, Axis> y({Dimension1, Dimension2}); 
	matrix<double> result({Dimension1, Dimension2});
	
	for (size_t i = 0; i < Dimension1; i++){
		for (size_t j = 0; j < Dimension2; j++){
			x(i,j) = i-3.0-j;
			y(i,j) = i+j+Dimension1;
			result(i,j)= x(i,j)-y(i,j);
		}
	}
	checkDenseExpressionEquality(x-y,result);
}

BOOST_AUTO_TEST_CASE_TEMPLATE( Remora_dense_matrix_Binary_Multiply, Axis, result_orientations )
{
	matrix<double, Axis> x({Dimension1, Dimension2}); 
	matrix<double, Axis> y({Dimension1, Dimension2}); 
	matrix<double> result({Dimension1, Dimension2});
	
	for (size_t i = 0; i < Dimension1; i++){
		for (size_t j = 0; j < Dimension2; j++){
			x(i,j) = i-3.0-j;
			y(i,j) = i+j+Dimension1;
			result(i,j)= x(i,j)*y(i,j);
		}
	}
	checkDenseExpressionEquality(x*y,result);
}

BOOST_AUTO_TEST_CASE_TEMPLATE( Remora_dense_matrix_Binary_Div, Axis, result_orientations )
{
	matrix<double, Axis> x({Dimension1, Dimension2}); 
	matrix<double, Axis> y({Dimension1, Dimension2}); 
	matrix<double> result({Dimension1, Dimension2});
	
	for (size_t i = 0; i < Dimension1; i++){
		for (size_t j = 0; j < Dimension2; j++){
			x(i,j) = i-3.0-j;
			y(i,j) = i+j+1;
			result(i,j)= x(i,j)/y(i,j);
		}
	}
	checkDenseExpressionEquality(x/y,result);
}

BOOST_AUTO_TEST_CASE_TEMPLATE( Remora_dense_matrix_Safe_Div, Axis, result_orientations )
{
	matrix<double, Axis> x({Dimension1, Dimension2}); 
	matrix<double, Axis> y({Dimension1, Dimension2}); 
	matrix<double> result({Dimension1, Dimension2});
	for (size_t i = 0; i < Dimension1; i++){
		for (size_t j = 0; j < Dimension2; j++){
			x(i,j) = i+j;
			y(i,j) = (i+j)%3;
			result(i,j)= ((i+j) % 3 == 0)? 2.0: x(i,j)/y(i,j);
		}
	}
	checkDenseExpressionEquality(safe_div(x,y,2.0),result);
}

BOOST_AUTO_TEST_CASE_TEMPLATE( Remora_dense_matrix_Binary_Pow, Axis, result_orientations )
{
	matrix<double, Axis> x({Dimension1, Dimension2}); 
	matrix<double, Axis> y({Dimension1, Dimension2}); 
	matrix<double> result({Dimension1, Dimension2});
	
	for (size_t i = 0; i < Dimension1; i++){
		for (size_t j = 0; j < Dimension2; j++){
			x(i,j) = 0.1*(i+j+1);
			y(i,j) = 0.0001*(i+j-3);
			result(i,j)= std::pow(x(i,j),y(i,j));
		}
	}
	checkDenseExpressionEquality(pow(x,y),result);
}

BOOST_AUTO_TEST_CASE_TEMPLATE( Remora_dense_matrix_Binary_Max, Axis, result_orientations )
{
	matrix<double, Axis> x({Dimension1, Dimension2}); 
	matrix<double, Axis> y({Dimension1, Dimension2}); 
	matrix<double> result({Dimension1, Dimension2});
	
	for (size_t i = 0; i < Dimension1; i++){
		for (size_t j = 0; j < Dimension2; j++){
			x(i,j) = 50.0+i-j;
			y(i,j) = i+j+1;
			result(i,j)= std::max(x(i,j),y(i,j));
		}
	}
	checkDenseExpressionEquality(max(x,y),result);
}
BOOST_AUTO_TEST_CASE_TEMPLATE( Remora_dense_matrix_Binary_Min, Axis, result_orientations )
{
	matrix<double, Axis> x({Dimension1, Dimension2}); 
	matrix<double, Axis> y({Dimension1, Dimension2}); 
	matrix<double> result({Dimension1, Dimension2});
	
	for (size_t i = 0; i < Dimension1; i++){
		for (size_t j = 0; j < Dimension2; j++){
			x(i,j) = 50.0+i-j;
			y(i,j) = i+j+1;
			result(i,j)= std::min(x(i,j),y(i,j));
		}
	}
	checkDenseExpressionEquality(min(x,y),result);
}


/////////////////////////////////////////////////////////////
//////MATRIX CONCATENATION
/////////////////////////////////////////////////////////////
/*
BOOST_AUTO_TEST_CASE_TEMPLATE( Remora_Concat_Matrix_Matrix_Right, Axis, result_orientations )
{
	matrix<double> x({Dimension1, Dimension2}); 
	matrix<double, Axis> y(Dimension1,2 * Dimension2); 
	matrix<double> result(Dimension1, 3 * Dimension2);
	matrix<double> result2(Dimension1, 3 * Dimension2);
	
	for (size_t i = 0; i < Dimension1; i++){
		for (size_t j = 0; j < Dimension2; j++){
			x(i,j) = 50.0+i-j;
			y(i,j) = i+j+1;
			y(i,j+Dimension2) = i+j+2;
			result(i,j)= x(i,j);
			result(i,j+Dimension2)= y(i,j);
			result(i,j+2*Dimension2)= y(i,j+Dimension2);
			result2(i,j+2*Dimension2)= x(i,j);
			result2(i,j)= y(i,j);
			result2(i,j+Dimension2)= y(i,j+Dimension2);
		}
	}
	
	checkDenseBlockAssign(x|y,result);
	checkDenseBlockAssign(y|x,result2);
}

BOOST_AUTO_TEST_CASE_TEMPLATE( Remora_Concat_Matrix_Matrix_Bottom, Axis, result_orientations )
{
	matrix<double, Axis> x({Dimension1, Dimension2}); 
	matrix<double> y(2 * Dimension1,Dimension2); 
	matrix<double> result(3 *Dimension1, Dimension2);
	matrix<double> result2(3 *Dimension1, Dimension2);
	
	for (size_t i = 0; i < Dimension1; i++){
		for (size_t j = 0; j < Dimension2; j++){
			x(i,j) = 50.0+i-j;
			y(i,j) = i+j+1;
			y(i+Dimension1, j) = i+j+2;
			result(i,j)= x(i,j);
			result(i + Dimension1, j)= y(i,j);
			result(i + 2 * Dimension1, j)= y(i + Dimension1,j);
			result2(i  + 2 * Dimension1,j)= x(i,j);
			result2(i, j)= y(i,j);
			result2(i + Dimension1, j)= y(i + Dimension1,j);
		}
	}
	checkDenseBlockAssign(x & y,result);
	checkDenseBlockAssign(y & x,result2);
}

BOOST_AUTO_TEST_CASE_TEMPLATE( Remora_Concat_Matrix_Vector_Right, Axis, result_orientations )
{
	matrix<double, Axis> x({Dimension1, Dimension2}); 
	vector<double> y(Dimension1); 
	matrix<double> result(Dimension1, Dimension2+1);
	matrix<double> result2(Dimension1, Dimension2+1);
	
	for (size_t i = 0; i < Dimension1; i++){
		for (size_t j = 0; j < Dimension2; j++){
			x(i,j) = 50.0+i-j;
			result(i,j)= x(i,j);
			result2(i,j+1)= x(i,j);
		}
		y(i) = i;
		result(i,Dimension2) = y(i);
		result2(i,0) = y(i);
	}
	checkDenseBlockAssign(x | y,result);
	checkDenseBlockAssign(y | x,result2);
}

BOOST_AUTO_TEST_CASE_TEMPLATE( Remora_Concat_Matrix_Vector_Bottom, Axis, result_orientations )
{
	matrix<double, Axis> x({Dimension1, Dimension2}); 
	vector<double> y(Dimension2); 
	matrix<double> result(Dimension1 + 1, Dimension2);
	matrix<double> result2(Dimension1 + 1, Dimension2);
	
	for (size_t i = 0; i < Dimension1; i++){
		for (size_t j = 0; j < Dimension2; j++){
			x(i,j) = 50.0+i-j;
			result(i,j)= x(i,j);
			result2(i+1,j)= x(i,j);
		}
	}
	for (size_t i = 0; i < Dimension2; i++){
		y(i) = i;
		result(Dimension1,i) = y(i);
		result2(0,i) = y(i);
	}
	checkDenseBlockAssign(x & y,result);
	checkDenseBlockAssign(y & x,result2);
}

BOOST_AUTO_TEST_CASE_TEMPLATE( Remora_Concat_Matrix_Scalar_Left, Axis, result_orientations )
{
	matrix<double, Axis> x({Dimension1, Dimension2}); 
	double t = 2.0;
	matrix<double> result(Dimension1, Dimension2 + 1);
	matrix<double> result2(Dimension1, Dimension2 + 1);
	
	for (size_t i = 0; i < Dimension1; i++){
		for (size_t j = 0; j < Dimension2; j++){
			x(i,j) = 50.0+i-j;
			result(i,j+1)= x(i,j);
			result2(i,j)= x(i,j);
			result(i,0) = t;
			result2(i,Dimension2) = t;
		}
	}
	checkDenseBlockAssign(t | x,result);
	checkDenseBlockAssign(x | t,result2);
}

BOOST_AUTO_TEST_CASE_TEMPLATE( Remora_Concat_Matrix_Scalar_Top, Axis, result_orientations )
{
	matrix<double, Axis> x({Dimension1, Dimension2}); 
	double t = 2.0;
	matrix<double> result(Dimension1 + 1, Dimension2);
	matrix<double> result2(Dimension1 + 1, Dimension2);
	
	for (size_t i = 0; i < Dimension1; i++){
		for (size_t j = 0; j < Dimension2; j++){
			x(i,j) = 50.0+i-j;
			result(i + 1,j)= x(i,j);
			result2(i,j)= x(i,j);
			result(0, j) = t;
			result2(Dimension1, j) = t;
		}
	}
	checkDenseBlockAssign(t & x,result);
	checkDenseBlockAssign(x & t,result2);
}*/



BOOST_AUTO_TEST_SUITE_END()
