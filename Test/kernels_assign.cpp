#define BOOST_TEST_MODULE Remora_Kernels_Assign
#include <boost/test/unit_test.hpp>
#include <boost/test/floating_point_comparison.hpp>
#include <boost/mpl/list.hpp>

#include <remora/kernels/assign.hpp>
#include <remora/dense.hpp>

using namespace remora;
struct ProxyFixture
{
	std::vector<unsigned> values;
	ProxyFixture():values(4*20*8){
		for(std::size_t i = 0; i != 4*20*8; ++i){
			values[i] = i;
		}
	}
};

BOOST_FIXTURE_TEST_SUITE (Remora_Kernels_Assign_Test, ProxyFixture);

typedef boost::mpl::list<axis<0,1>, axis<1,0> > axis_types_2d;
typedef boost::mpl::list<axis<0,1,2>, axis<0,2,1>, axis<1,0,2>, axis<1,2,0>, axis<2,0,1>, axis<2,1,0> > axis_types_3d;
typedef boost::mpl::list<axis<0,1,2,3>, axis<0,3,2,1>, axis<1,3,2,0>, axis<3,2,1,0>, axis<0,3,1,2> > axis_types_4d;

///////////////////////////////////////////////////////////////////////////////
//////FUNCTOR APPLY TO TENSOR
//////////////////////////////////////////////////////////////////////////////

//test for assign of the form m_ij=f(m_ij,t) for constant t//
BOOST_AUTO_TEST_CASE_TEMPLATE( Remora_Dense_Apply, Axis, axis_types_3d ){
	std::array<std::size_t, 3> strides = {160, 8, 1};
	tensor_shape<3> shape = {2, 10, 4};
	strides = Axis::to_axis(strides);
	shape = Axis::to_axis(shape);
	
	std::vector<unsigned> target_results = values;
	for(std::size_t i = 0; i != 2; ++i)
		for(std::size_t j = 0; j != 10; ++j)
			for(std::size_t k = 0; k != 4; ++k)
				target_results[i*160+j*8+k] *= 3;
	
	dense_tensor_adaptor<unsigned, Axis, dense_tag, cpu_tag> adaptor({values.data(), strides},no_queue(), shape);
	device_traits<cpu_tag>::multiply_scalar<unsigned> f(3);
	kernels::apply(adaptor, f);
	
	
	for(std::size_t i = 0; i != 4*20*8; ++i){
		BOOST_CHECK_EQUAL(values[i], target_results[i]);
	}
}

///////////////////////////////////////////////////////////////////////////////
//////FUNCTOR TENSOR ASSIGNMENT
//////////////////////////////////////////////////////////////////////////////

BOOST_AUTO_TEST_CASE( Remora_Dense_Functor_Assign_1D){
	// typedef axis<0,1,2,3> axis_permute;
	tensor_shape<1> shape = {5};
	std::array<std::size_t, 1> strides_rhs = {20};
	std::array<std::size_t, 1> strides_lhs =  {10};
	//compute ground truth
	std::vector<unsigned> target_results(50, 0);
	for(std::size_t i = 0; i != 5; ++i){
		target_results.at(strides_lhs[0] * i) = 3*values.at(strides_rhs[0] * i);
	}
	
	std::vector<unsigned> results(target_results.size(), 0);
	dense_tensor_adaptor<unsigned, axis<0>, dense_tag, cpu_tag> lhs({results.data(), strides_lhs},no_queue(), shape);
	dense_tensor_adaptor<unsigned, axis<0>, dense_tag, cpu_tag> rhs({values.data(), strides_rhs},no_queue(), shape);
	device_traits<cpu_tag>::multiply_assign<unsigned> f(3);
	kernels::assign(lhs, rhs, f);
	
	for(std::size_t i = 0; i != target_results.size(); ++i){
		BOOST_CHECK_EQUAL(results[i], target_results[i]);
	}
}
BOOST_AUTO_TEST_CASE_TEMPLATE( Remora_Dense_Functor_Assign_2D, Axis, axis_types_2d ){
	typedef axis<1,0> axis_permute;
	// typedef axis<0,1,2,3> axis_permute;
	tensor_shape<2> shape = {5, 3};
	std::array<std::size_t, 2> strides_rhs = {20, 5};
	auto strides_lhs = Axis::compute_dense_strides(shape).shape_array;
	//compute ground truth
	std::vector<unsigned> target_results(30, 0);
	for(std::size_t i = 0; i != 5; ++i){
		for(std::size_t j = 0; j != 3; ++j){
			auto ind_lhs = strides_lhs[0] * i + strides_lhs[1] * j;
			auto ind_rhs = strides_rhs[0] * i + strides_rhs[1] * j;
			target_results.at(ind_lhs) = 3*values.at(ind_rhs);
		}
	}
	
	//apply Axis permutation
	typedef axis<
		Axis::template element_v<axis_permute::element_v<0> >,
		Axis::template element_v<axis_permute::element_v<1> >
	> axis_lhs;
	strides_lhs = axis_permute::to_axis(strides_lhs);
	strides_rhs = axis_permute::to_axis(strides_rhs);
	shape = axis_permute::to_axis(shape);
	
	std::vector<unsigned> results(target_results.size(), 0);
	dense_tensor_adaptor<unsigned, axis_lhs, dense_tag, cpu_tag> lhs({results.data(), strides_lhs},no_queue(), shape);
	dense_tensor_adaptor<unsigned, axis_permute, dense_tag, cpu_tag> rhs({values.data(), strides_rhs},no_queue(), shape);
	device_traits<cpu_tag>::multiply_assign<unsigned> f(3);
	kernels::assign(lhs, rhs, f);
	
	for(std::size_t i = 0; i != target_results.size(); ++i){
		BOOST_CHECK_EQUAL(results[i], target_results[i]);
	}
}
BOOST_AUTO_TEST_CASE_TEMPLATE( Remora_Dense_Functor_Assign_4D, Axis, axis_types_4d ){
	typedef axis<3,1,2,0> axis_permute;
	// typedef axis<0,1,2,3> axis_permute;
	tensor_shape<4> shape = {5, 3, 2, 4};
	std::array<std::size_t, 4> strides_rhs = {75, 16, 4, 1};
	// std::array<std::size_t, 4> strides_rhs = {24, 8, 4, 1};
	// std::array<std::size_t, 4> strides_lhs = {48, 16, 8, 2};
	auto strides_lhs = Axis::compute_dense_strides(shape).shape_array;
	//compute ground truth
	std::vector<unsigned> target_results(300, 0);
	for(std::size_t i = 0; i != 5; ++i){
		for(std::size_t j = 0; j != 3; ++j){
			for(std::size_t k = 0; k != 2; ++k){
				for(std::size_t l = 0; l != 4; ++l){
					auto ind_lhs = strides_lhs[0] * i + strides_lhs[1] * j + strides_lhs[2] * k + strides_lhs[3] * l;
					auto ind_rhs = strides_rhs[0] * i + strides_rhs[1] * j + strides_rhs[2] * k + strides_rhs[3] * l;
					target_results.at(ind_lhs) = 3*values.at(ind_rhs);
				}
			}
		}
	}
	
	//apply Axis permutation
	typedef axis<
		Axis::template element_v<axis_permute::element_v<0> >,
		Axis::template element_v<axis_permute::element_v<1> >,
		Axis::template element_v<axis_permute::element_v<2> >,
		Axis::template element_v<axis_permute::element_v<3> >
	> axis_lhs;
	strides_lhs = axis_permute::to_axis(strides_lhs);
	strides_rhs = axis_permute::to_axis(strides_rhs);
	shape = axis_permute::to_axis(shape);
	
	std::vector<unsigned> results(target_results.size(), 0);
	dense_tensor_adaptor<unsigned, axis_lhs, dense_tag, cpu_tag> lhs({results.data(), strides_lhs},no_queue(), shape);
	dense_tensor_adaptor<unsigned, axis_permute, dense_tag, cpu_tag> rhs({values.data(), strides_rhs},no_queue(), shape);
	device_traits<cpu_tag>::multiply_assign<unsigned> f(3);
	kernels::assign(lhs, rhs, f);
	
	for(std::size_t i = 0; i != target_results.size(); ++i){
		BOOST_CHECK_EQUAL(results[i], target_results[i]);
	}
}


//////////////////////////////////////////////////////
//////SIMPLE ASSIGNMENT
//////////////////////////////////////////////////////

BOOST_AUTO_TEST_CASE_TEMPLATE( Remora_Dense_Assign_4D, Axis, axis_types_4d ){
	typedef axis<3,1,2,0> axis_permute;
	// typedef axis<0,1,2,3> axis_permute;
	tensor_shape<4> shape = {5, 3, 2, 4};
	std::array<std::size_t, 4> strides_rhs = {75, 16, 4, 1};
	auto strides_lhs = Axis::compute_dense_strides(shape).shape_array;
	//compute ground truth
	std::vector<unsigned> target_results(300, 0);
	for(std::size_t i = 0; i != 5; ++i){
		for(std::size_t j = 0; j != 3; ++j){
			for(std::size_t k = 0; k != 2; ++k){
				for(std::size_t l = 0; l != 4; ++l){
					auto ind_lhs = strides_lhs[0] * i + strides_lhs[1] * j + strides_lhs[2] * k + strides_lhs[3] * l;
					auto ind_rhs = strides_rhs[0] * i + strides_rhs[1] * j + strides_rhs[2] * k + strides_rhs[3] * l;
					target_results.at(ind_lhs) = values.at(ind_rhs);
				}
			}
		}
	}
	
	//apply Axis permutation
	typedef axis<
		Axis::template element_v<axis_permute::element_v<0> >,
		Axis::template element_v<axis_permute::element_v<1> >,
		Axis::template element_v<axis_permute::element_v<2> >,
		Axis::template element_v<axis_permute::element_v<3> >
	> axis_lhs;
	strides_lhs = axis_permute::to_axis(strides_lhs);
	strides_rhs = axis_permute::to_axis(strides_rhs);
	shape = axis_permute::to_axis(shape);
	
	std::vector<unsigned> results(target_results.size(), 0);
	dense_tensor_adaptor<unsigned, axis_lhs, dense_tag, cpu_tag> lhs({results.data(), strides_lhs},no_queue(), shape);
	dense_tensor_adaptor<unsigned, axis_permute, dense_tag, cpu_tag> rhs({values.data(), strides_rhs},no_queue(), shape);
	kernels::assign(lhs, rhs);
	
	for(std::size_t i = 0; i != target_results.size(); ++i){
		BOOST_CHECK_EQUAL(results[i], target_results[i]);
	}
}
BOOST_AUTO_TEST_SUITE_END()
