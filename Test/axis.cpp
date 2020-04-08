#define BOOST_TEST_MODULE Remora_Axis
#include <boost/test/included/unit_test.hpp>
#include <boost/mpl/list.hpp>

#include <remora/detail/axis.hpp>
using namespace remora;

BOOST_AUTO_TEST_SUITE(Remora_Axis_Test)

BOOST_AUTO_TEST_CASE( Axis_Compile_Tests){
	// accessors
	{
		typedef axis<2,0,3,1> AX;
		static_assert(AX::num_dims ==  4);
		static_assert(AX::element_v<0> ==  2);
		static_assert(AX::element_v<1> ==  0);
		static_assert(AX::element_v<2> ==  3);
		static_assert(AX::element_v<3> ==  1);
		
		static_assert(AX::index_of_v<2> ==  0);
		static_assert(AX::index_of_v<0> ==  1);
		static_assert(AX::index_of_v<3> ==  2);
		static_assert(AX::index_of_v<1> ==  3);
	}
	
	//tests for index transformations
	{
		axis<0,1,2,3> def_ax = default_axis<4>();
		axis<3,2,1,0> ax = axis<3,2,1,0>::inverse_t();
		axis<2,0,3,1> ax1 = axis<1,3,0,2>::inverse_t();
		
		
		axis_set<3,2,0> ax_set = axis<3,2,1,0>::remove_t<2>();
		axis<1,2,0> ax_slice = axis<1,3,0,2>::slice_t<3>();
		axis<0,2,1> ax_slice1 = axis<1,3,0,2>::slice_t<2>();
		
		axis<3,4,1,2,0> ax_split = axis<2,3,1,0>::split_t<2>();
		
		
		axis_set<1,3,0> ax_select = axis<1,3,0,2>::select_t<0,1,2>();
		axis_set<2,3> ax_select1 = axis<1,3,0,2>::select_t<3,1>();
		
		axis_set<1> ax_select2 = axis<1,3,0,2>::front_t<1>();
		axis_set<1,3> ax_select3 = axis<1,3,0,2>::front_t<2>();
		axis_set<1,3, 0> ax_select4 = axis<1,3,0,2>::front_t<3>();
		axis_set<1,3,0,2> ax_select5 = axis<1,3,0,2>::front_t<4>();
	}
}


BOOST_AUTO_TEST_CASE( Axis_to_array){
	auto arr = axis<2,0,3,1>::to_array();
	BOOST_CHECK_EQUAL(arr[0],2);
	BOOST_CHECK_EQUAL(arr[1],0);
	BOOST_CHECK_EQUAL(arr[2],3);
	BOOST_CHECK_EQUAL(arr[3],1);
}

BOOST_AUTO_TEST_CASE( Axis_to_axis){
	std::array<unsigned, 4> input={3,0,1,2};
	std::array<unsigned, 4> arr = axis<2,0,3,1>::to_axis(input);
	BOOST_CHECK_EQUAL(arr[0],1);
	BOOST_CHECK_EQUAL(arr[1],3);
	BOOST_CHECK_EQUAL(arr[2],2);
	BOOST_CHECK_EQUAL(arr[3],0);
}

BOOST_AUTO_TEST_CASE( Axis_from_axis){
	std::array<unsigned, 4> input={3,0,1,2};
	std::array<unsigned, 4> arr = axis<1,3,0,2>::from_axis(input);
	BOOST_CHECK_EQUAL(arr[0],1);
	BOOST_CHECK_EQUAL(arr[1],3);
	BOOST_CHECK_EQUAL(arr[2],2);
	BOOST_CHECK_EQUAL(arr[3],0);
}

BOOST_AUTO_TEST_CASE( Axis_compute_dense_strides){
	std::array<unsigned, 4> input={3,5,2,4};
	std::array<unsigned, 4> arr = axis<1,3,0,2>::compute_dense_strides(input);
	BOOST_CHECK_EQUAL(arr[0],20);
	BOOST_CHECK_EQUAL(arr[1],1);
	BOOST_CHECK_EQUAL(arr[2],60);
	BOOST_CHECK_EQUAL(arr[3],5);
}

BOOST_AUTO_TEST_CASE( Axis_leading){
	std::array<unsigned, 4> strides={2,24,1,6};
	BOOST_CHECK_EQUAL((axis<1,3,0,2>::leading<0>(strides)),24);
	BOOST_CHECK_EQUAL((axis<1,3,0,2>::leading<1>(strides)),6);
	BOOST_CHECK_EQUAL((axis<1,3,0,2>::leading<2>(strides)),2);
	BOOST_CHECK_EQUAL((axis<1,3,0,2>::leading<3>(strides)),1);
}
BOOST_AUTO_TEST_CASE( Axis_element){
	std::array<unsigned, 4> strides={2,24,1,6};
	std::array<unsigned, 4> index={1,4,1,3};
	BOOST_CHECK_EQUAL((axis<1,3,0,2>::element(index,strides)),117);
}

BOOST_AUTO_TEST_SUITE_END();