#define BOOST_TEST_MODULE Remora_Tensor_Reductions
#include <boost/test/unit_test.hpp>
#include <boost/test/floating_point_comparison.hpp>

#include <remora/expressions.hpp>
#include <remora/dense.hpp>
#include <boost/mpl/list.hpp>

#include <iostream>

using namespace remora;
typedef boost::mpl::list<axis<0,1,2>, axis<0,2,1>, axis<1,0,2>, axis<1,2,0>, axis<2,0,1>, axis<2,1,0> > axis_3d_types;

BOOST_AUTO_TEST_SUITE (Remora_reductions_test)


BOOST_AUTO_TEST_CASE_TEMPLATE( Tensor_Reduce_Sum_1, Axis, axis_3d_types ){
	
	std::vector<unsigned> values(23*22*18);
	for(std::size_t i = 0; i != values.size(); ++i){
		values[i] = i;
	}
	
	
	tensor_shape<3> shape = {23, 22, 18};
	std::array<std::size_t, 3> strides = {22*18, 18, 1};
	strides = Axis::to_axis(strides);
	shape = Axis::to_axis(shape);
	dense_tensor_adaptor<unsigned, Axis, integer_list<bool, 1,1,1>, cpu_tag> adaptor({values.data(), strides},no_queue(), shape);
	
	//compute ground truth
	tensorN<unsigned, 2, cpu_tag> result1(shape.slice(2));
	tensorN<unsigned, 2, cpu_tag> result2(shape.slice(1));
	tensorN<unsigned, 2, cpu_tag> result3(shape.slice(0));
	
	for(std::size_t i = 0; i != shape[0]; ++i){
		for(std::size_t j = 0; j != shape[1]; ++j){
			for(std::size_t k = 0; k != shape[2]; ++k){
				result1(i,j) += adaptor(i,j,k);
				result2(i,k) += adaptor(i,j,k);
				result3(j,k) += adaptor(i,j,k);
			}
		}
	}
	
	//evaluate expressions
	tensorN<unsigned, 2, cpu_tag> op1 = sum(adaptor, axis_set<2>());
	tensorN<unsigned, 2, cpu_tag> op1_plus(shape.slice(2),1);
	plus_assign(op1_plus, sum(adaptor, axis_set<2>()));
	tensorN<unsigned, 2, cpu_tag> op2 = sum(adaptor, axis_set<1>());
	tensorN<unsigned, 2, cpu_tag> op3 = sum(adaptor, axis_set<0>());
	
	//check column major to ensure that this also works.
	tensor<unsigned, axis<1,0>, cpu_tag> op1_trans = sum(adaptor, axis_set<2>());
	tensor<unsigned, axis<1,0>, cpu_tag> op2_trans = sum(adaptor, axis_set<1>());
	tensor<unsigned, axis<1,0>, cpu_tag> op3_trans = sum(adaptor, axis_set<0>());
	
	
	//check sizes
	BOOST_REQUIRE_EQUAL(op1.shape()[0], result1.shape()[0]);
	BOOST_REQUIRE_EQUAL(op1.shape()[1], result1.shape()[1]);
	BOOST_REQUIRE_EQUAL(op2.shape()[0], result2.shape()[0]);
	BOOST_REQUIRE_EQUAL(op2.shape()[1], result2.shape()[1]);
	BOOST_REQUIRE_EQUAL(op3.shape()[0], result3.shape()[0]);
	BOOST_REQUIRE_EQUAL(op3.shape()[1], result3.shape()[1]);
	
	for(std::size_t i = 0; i != shape[0]; ++i){
		for(std::size_t j = 0; j != shape[1]; ++j){
			BOOST_CHECK_EQUAL(result1(i,j), op1(i,j));
			BOOST_CHECK_EQUAL(result1(i,j), op1_trans(i,j));
			BOOST_CHECK_EQUAL(result1(i,j) + 1, op1_plus(i,j));
		}
	}
	for(std::size_t i = 0; i != shape[0]; ++i){
		for(std::size_t k = 0; k != shape[2]; ++k){
			BOOST_CHECK_EQUAL(result2(i,k), op2(i,k));
			BOOST_CHECK_EQUAL(result2(i,k), op2_trans(i,k));
		}
	}
	for(std::size_t j = 0; j != shape[1]; ++j){
		for(std::size_t k = 0; k != shape[2]; ++k){
			BOOST_CHECK_EQUAL(result3(j,k), op3(j,k));
			BOOST_CHECK_EQUAL(result3(j,k), op3_trans(j,k));
		}
	}
}


BOOST_AUTO_TEST_CASE_TEMPLATE( Tensor_Reduce_Sum_2, Axis, axis_3d_types ){
	
	std::vector<unsigned> values(23*22*18);
	for(std::size_t i = 0; i != values.size(); ++i){
		values[i] = i;
	}
	
	
	tensor_shape<3> shape = {23, 22, 18};
	std::array<std::size_t, 3> strides = {22*18, 18, 1};
	strides = Axis::to_axis(strides);
	shape = Axis::to_axis(shape);
	dense_tensor_adaptor<unsigned, Axis, integer_list<bool, 1,1,1>, cpu_tag> adaptor({values.data(), strides},no_queue(), shape);
	
	//compute ground truth
	tensorN<unsigned, 1, cpu_tag> result1(shape[2]);
	tensorN<unsigned, 1, cpu_tag> result2(shape[1]);
	tensorN<unsigned, 1, cpu_tag> result3(shape[0]);
	
	for(std::size_t i = 0; i != shape[0]; ++i){
		for(std::size_t j = 0; j != shape[1]; ++j){
			for(std::size_t k = 0; k != shape[2]; ++k){
				result1(k) += adaptor(i,j,k);
				result2(j) += adaptor(i,j,k);
				result3(i) += adaptor(i,j,k);
			}
		}
	}
	
	//evaluate expressions
	tensorN<unsigned, 1, cpu_tag> op1 = sum(adaptor, axis_set<0,1>());
	tensorN<unsigned, 1, cpu_tag> op11 = sum(adaptor, axis_set<1,0>());
	tensorN<unsigned, 1, cpu_tag> op2 = sum(adaptor, axis_set<2,0>());
	tensorN<unsigned, 1, cpu_tag> op21 = sum(adaptor, axis_set<0,2>());
	tensorN<unsigned, 1, cpu_tag> op3 = sum(adaptor, axis_set<1,2>());
	tensorN<unsigned, 1, cpu_tag> op31 = sum(adaptor, axis_set<2,1>());
	
	//check column major to ensure that this also works.
	tensor<unsigned, axis<0>, cpu_tag> op12 = sum(adaptor, axis_set<0,1>());
	tensor<unsigned, axis<0>, cpu_tag> op22 = sum(adaptor, axis_set<2,0>());
	tensor<unsigned, axis<0>, cpu_tag> op32 = sum(adaptor, axis_set<2,1>());
	
	
	//check sizes
	BOOST_REQUIRE_EQUAL(op1.shape()[0], result1.shape()[0]);
	BOOST_REQUIRE_EQUAL(op2.shape()[0], result2.shape()[0]);
	BOOST_REQUIRE_EQUAL(op3.shape()[0], result3.shape()[0]);
	BOOST_REQUIRE_EQUAL(op11.shape()[0], result1.shape()[0]);
	BOOST_REQUIRE_EQUAL(op21.shape()[0], result2.shape()[0]);
	BOOST_REQUIRE_EQUAL(op31.shape()[0], result3.shape()[0]);
	BOOST_REQUIRE_EQUAL(op12.shape()[0], result1.shape()[0]);
	BOOST_REQUIRE_EQUAL(op22.shape()[0], result2.shape()[0]);
	BOOST_REQUIRE_EQUAL(op32.shape()[0], result3.shape()[0]);
	
	for(std::size_t i = 0; i != shape[2]; ++i){
		BOOST_CHECK_EQUAL(result1(i), op1(i));
		BOOST_CHECK_EQUAL(result1(i), op11(i));
		BOOST_CHECK_EQUAL(result1(i), op12(i));
	}
	for(std::size_t i = 0; i != shape[1]; ++i){
		BOOST_CHECK_EQUAL(result2(i), op2(i));
		BOOST_CHECK_EQUAL(result2(i), op21(i));
		BOOST_CHECK_EQUAL(result2(i), op22(i));
	}
	for(std::size_t i = 0; i != shape[0]; ++i){
		BOOST_CHECK_EQUAL(result3(i), op3(i));
		BOOST_CHECK_EQUAL(result3(i), op31(i));
		BOOST_CHECK_EQUAL(result3(i), op32(i));
	}
}


BOOST_AUTO_TEST_CASE_TEMPLATE( Tensor_Reduce_Sum_scalar, Axis, axis_3d_types ){
	
	std::vector<unsigned> values(23*22*18);
	for(std::size_t i = 0; i != values.size(); ++i){
		values[i] = i;
	}	
	tensor_shape<3> shape = {23, 22, 18};
	std::array<std::size_t, 3> strides = {22*18, 18, 1};
	strides = Axis::to_axis(strides);
	shape = Axis::to_axis(shape);
	dense_tensor_adaptor<unsigned, Axis, integer_list<bool, 1,1,1>, cpu_tag> adaptor1({values.data(), strides},no_queue(), shape);
	auto adaptor2 = slice(adaptor1,ax::same, ax::same, 0);
	auto adaptor3 = slice(adaptor1, ax::same, 0, 0);
	auto adaptor4 = slice(adaptor1, 0, ax::same, ax::same);
	//compute ground truth
	unsigned result1 = 0;
	unsigned result2 = 0;
	unsigned result3 = 0;
	unsigned result4 = 0;
	
	for(std::size_t i = 0; i != shape[0]; ++i){
		result3 += adaptor1(i,0,0);
		result4 = 0;
		for(std::size_t j = 0; j != shape[1]; ++j){
			result2 += adaptor1(i,j,0);
			for(std::size_t k = 0; k != shape[2]; ++k){
				result1	+= adaptor1(i,j,k);
				result4	+= adaptor1(0,j,k);
			}
		}
	}
	//evaluate expressions
	tensorN<unsigned, 0, cpu_tag> op1 = sum(adaptor1);
	unsigned op2 = sum(adaptor2);
	unsigned op3 = sum(adaptor3);
	unsigned op4 = sum(adaptor4);
	
	//test for reduction of a scalar
	scalar<unsigned> x(5);
	unsigned op5 = sum(x);
	
	
	//check results
	BOOST_CHECK_EQUAL(result1, op1());
	BOOST_CHECK_EQUAL(result2, op2);
	BOOST_CHECK_EQUAL(result3, op3);
	BOOST_CHECK_EQUAL(result4, op4);
	BOOST_CHECK_EQUAL(5, op5);
}

BOOST_AUTO_TEST_CASE_TEMPLATE( Tensor_Reduce_Sum_Proxies, Axis, axis_3d_types ){
	
	std::vector<unsigned> values(20*22*18);
	for(std::size_t i = 0; i != values.size(); ++i){
		values[i] = i;
	}
	
	
	tensor_shape<3> shape = {20, 22, 18};
	std::array<std::size_t, 3> strides = Axis::compute_dense_strides(shape).shape_array;
	dense_tensor_adaptor<unsigned, Axis, integer_list<bool, 1,1,1>, cpu_tag> adaptor({values.data(), strides},no_queue(), shape);
	auto const& const_adaptor = adaptor;
	
	typedef device_traits<cpu_tag>::add<unsigned> F;
	
	auto op_sum = sum(adaptor, axis_set<2>());
	
	//define proxies:
	//split of the summed result
	tensor_reduce_last<
		dense_tensor_adaptor<unsigned const, typename Axis::template split_t<0>, integer_list<bool, 1, 1, 1, 1>, cpu_tag>
	,F> op_split = reshape(op_sum,ax::split<2>(4,5),ax::same);
	//slice of the result
	typedef integer_list<bool, 
		(Axis::template element_v<0> + 1 != Axis::template element_v<1>), 
		(Axis::template element_v<2> + 1 != Axis::template element_v<1>)
	> sliced_storage;
	tensor_reduce_last<
		dense_tensor_adaptor<unsigned const, typename Axis::template slice_t<1>, sliced_storage, cpu_tag>
	, F> op_slice = slice(op_sum,ax::same, 0);
	
	typedef decltype(reshape(const_adaptor,ax::merge<2>(), ax::same)) merged_adaptor;
	tensor_reduce_last<merged_adaptor, F> op_merge = reshape(op_sum,ax::merge<2>());
	
	//and evaluate
	tensorN<unsigned, 3, cpu_tag> eval_split = op_split;
	tensorN<unsigned, 1, cpu_tag> eval_slice = op_slice;
	tensorN<unsigned, 1, cpu_tag> eval_merge = op_merge;
	
	//compute ground truth
	tensorN<unsigned, 2, cpu_tag> result_sum = op_sum; 
	tensorN<unsigned, 3, cpu_tag> result_split = reshape(result_sum,ax::split<2>(4,5),ax::same);
	tensorN<unsigned, 1, cpu_tag> result_slice = slice(result_sum,ax::same, 0);
	tensorN<unsigned, 1, cpu_tag> result_merge = reshape(result_sum,ax::merge<2>());
	
	
	//check sizes
	BOOST_REQUIRE_EQUAL(result_split.shape()[0], eval_split.shape()[0]);
	BOOST_REQUIRE_EQUAL(result_split.shape()[1], eval_split.shape()[1]);
	BOOST_REQUIRE_EQUAL(result_split.shape()[2], eval_split.shape()[2]);
	BOOST_REQUIRE_EQUAL(result_slice.shape()[0], eval_slice.shape()[0]);
	BOOST_REQUIRE_EQUAL(result_merge.shape()[0], eval_merge.shape()[0]);
	
	
	for(std::size_t i = 0; i != 4; ++i){
		for(std::size_t j = 0; j != 5; ++j){
			for(std::size_t k = 0; k != 22; ++k){
				BOOST_CHECK_EQUAL(result_split(i,j,k), eval_split(i,j,k));
			}
		}
	}
	
	
	for(std::size_t i = 0; i != shape[0]; ++i){
		BOOST_CHECK_EQUAL(result_slice(i), eval_slice(i));
	}
	
	for(std::size_t i = 0; i != shape[0]*shape[1]; ++i){
		BOOST_CHECK_EQUAL(result_merge(i), eval_merge(i));
	}
}


//test that automatic merging works
BOOST_AUTO_TEST_CASE( Tensor_Reduce_Sum_Multi_Merge){
	
	std::vector<unsigned> values(4*5*22*18);
	for(std::size_t i = 0; i != values.size(); ++i){
		values[i] = i;
	}
	
	//create a maximally mergeable adaptor
	tensor_shape<4> shape = {4, 5, 22, 18};
	std::array<std::size_t, 4> strides = default_axis<4>::compute_dense_strides(shape).shape_array;
	
	dense_tensor_adaptor<unsigned, default_axis<4>, integer_list<bool, 1,1,1,1>, cpu_tag> adaptor({values.data(), strides},no_queue(), shape);
	
	typedef device_traits<cpu_tag>::add<unsigned> F;
	typedef tensor_reduce_last<
		dense_tensor_adaptor<unsigned const, axis<0, 1>, integer_list<bool, 1, 1>, cpu_tag>,
		F
	> expr_type;
	expr_type op_merge1 = sum(adaptor, axis_set<3,2,1>());//most simple case. all direct merges possible
	expr_type op_merge2 = sum(adaptor, axis_set<1,2,3>());//should result in 2 permuted-merges
	
	//compute result
	tensorN<unsigned, 1> result({shape[0]},0);
	for(std::size_t i = 0; i != shape[0]; ++i){
		for(std::size_t j = 0; j != shape[1]; ++j){
			for(std::size_t k = 0; k != shape[2]; ++k){
				for(std::size_t l = 0; l != shape[3]; ++l){
					result(i) += adaptor(i,j,k, l);
				}
			}
		}
	}
	
	tensorN<unsigned, 1> eval1 = op_merge1;
	tensorN<unsigned, 1> eval2 = op_merge2;
	for(std::size_t i = 0; i != shape[0]; ++i){
		BOOST_CHECK_EQUAL(result(i), eval1(i));
		BOOST_CHECK_EQUAL(result(i), eval2(i));
	}
}

//All other instances of reductions use the same code & logic. So, after performing all the tests above, only a simple sanity check needed for the rest.

BOOST_AUTO_TEST_CASE( Remora_Max )
{
	matrix<int> x({10, 20}); 
	vector<int> result({10}, -3);
	for (size_t i = 0; i < 10; i++){
		for (size_t j = 0; j < 20; j++){
			x(i,j) = 2*i-3-j;
			result(i) = std::max(result(i), x(i,j));
		}
	}
	vector<int> op = max(x, axis_set<1>());
	for(std::size_t i = 0; i != 10; ++i)
		BOOST_CHECK_EQUAL(result(i),op(i));
}

BOOST_AUTO_TEST_CASE( Remora_Min )
{
	matrix<int> x({10, 20}); 
	vector<int> result({10}, -3);
	for (size_t i = 0; i < 10; i++){
		for (size_t j = 0; j < 20; j++){
			x(i,j) = 2*i-3-j;
			result(i) = std::min(result(i), x(i,j));
		}
	}
	vector<int> op = min(x, axis_set<1>());
	for(std::size_t i = 0; i != 10; ++i)
		BOOST_CHECK_EQUAL(result(i),op(i));
}
//remainder are all instances of tests that lead to a scalar. This is not implemented, yet.
/*
BOOST_AUTO_TEST_CASE_TEMPLATE( Remora_trace, Axis, result_orientations )
{
	matrix<double, Axis> x(Dimension1, Dimension1); 
	double result = 0.0f;
	for (size_t i = 0; i < Dimension1; i++){
		for (size_t j = 0; j < Dimension1; j++){
			x(i,j) = 2*i-3.0-j;
		}
		result += x(i,i);
	}
	BOOST_CHECK_CLOSE(trace(x),result, 1.e-6);
}
*/

BOOST_AUTO_TEST_SUITE_END()
