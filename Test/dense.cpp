#define BOOST_TEST_MODULE Remora_Dense
#include <remora/dense.hpp>
#include <remora/assignment.hpp>

#include <vector>

#include <boost/test/included/unit_test.hpp>
#include <boost/mpl/list.hpp>
#include <algorithm>


using namespace remora;

struct ProxyFixture
{
	std::vector<unsigned> values;
	ProxyFixture():values(3*20*7){
		for(std::size_t i = 0; i != 3*20*7; ++i){
			values[i] = i;
		}
	}
};

BOOST_FIXTURE_TEST_SUITE (Remora_Dense_Tensor_Test, ProxyFixture);

BOOST_AUTO_TEST_CASE( Tensor_Adaptor_1D){
	dense_tensor_adaptor<unsigned, axis<0>, dense_tag, cpu_tag> adaptor({values.data(), {1}},no_queue(), 3*20*7);
	BOOST_CHECK_EQUAL(adaptor.shape().size(), 1);
	BOOST_CHECK_EQUAL(adaptor.shape()[0], 3*20*7);
	BOOST_CHECK_EQUAL(adaptor.raw_storage().values, values.data());
	BOOST_CHECK_EQUAL(adaptor.raw_storage().strides.size(), 1);
	BOOST_CHECK_EQUAL(adaptor.raw_storage().strides[0], 1);
	auto elem = adaptor.elements();
	for(std::size_t i = 0; i != 3*20*7; ++i){
		BOOST_CHECK_EQUAL(adaptor(i),values[i]);
		BOOST_CHECK_EQUAL(elem(i),values[i]);
	}
}



typedef boost::mpl::list<axis<0,1,2>, axis<0,2,1>, axis<1,0,2>, axis<1,2,0>, axis<2,0,1>, axis<2,1,0> > axis_types;
BOOST_AUTO_TEST_CASE_TEMPLATE( Tensor_Adaptor_3D, Axis, axis_types ){
	std::array<std::size_t, 3> strides = {140, 7, 1};
	strides = Axis::to_axis(strides);
	tensor_shape<3> shape = {3, 20, 7};
	shape = Axis::to_axis(shape);
	
	//check internal structure
	dense_tensor_adaptor<unsigned, Axis, dense_tag, cpu_tag> adaptor({values.data(), strides},no_queue(), shape);
	BOOST_CHECK_EQUAL(adaptor.shape().size(), 3);
	BOOST_CHECK_EQUAL(adaptor.raw_storage().strides.size(), 3);
	BOOST_CHECK_EQUAL(adaptor.raw_storage().values, values.data());
	for(std::size_t dim = 0; dim != 3; ++dim){
		BOOST_CHECK_EQUAL(adaptor.shape()[dim], shape[dim]);
		BOOST_CHECK_EQUAL(adaptor.raw_storage().strides[dim], strides[dim]);
	}
	
	//check element access
	auto elem = adaptor.elements();
	for(std::size_t i = 0; i != shape[0]; ++i){
		for(std::size_t j = 0; j != shape[1]; ++j){
			for(std::size_t k = 0; k != shape[2]; ++k){
				std::size_t pos = Axis::element(std::array<std::size_t, 3>{i,j,k}, strides);
				BOOST_CHECK_EQUAL(adaptor(i,j,k),values[pos]);
				BOOST_CHECK_EQUAL(elem(i,j,k),values[pos]);
			}
		}
	}
}

BOOST_AUTO_TEST_CASE_TEMPLATE( Tensor_3D, Axis, axis_types ){
	//shape and stride in standard axis
	std::array<std::size_t, 3> strides = {140, 7, 1};
	tensor_shape<3> shape = {3, 20, 7};
	//transform to current axis
	strides = Axis::to_axis(strides);
	shape = Axis::to_axis(shape);
	
	// construct a few tensors:
	tensor<unsigned, Axis, cpu_tag> A(shape);
	auto const& A_cnst = A;
	tensor<unsigned, Axis, cpu_tag> Azero(shape, 0.0);
	tensor<unsigned, Axis, cpu_tag> A_resize;
	A_resize.resize(shape);
	
	// fill storage of A with values
	std::copy(values.begin(), values.end(), A.raw_storage().values);
	std::copy(values.begin(), values.end(), A_resize.raw_storage().values);
	
	//check internal structure
	BOOST_CHECK_EQUAL(A.shape().size(), 3);
	BOOST_CHECK_EQUAL(A.raw_storage().strides.size(), 3);
	for(std::size_t dim = 0; dim != 3; ++dim){
		BOOST_CHECK_EQUAL(A.shape()[dim], shape[dim]);
		BOOST_CHECK_EQUAL(A.raw_storage().strides[dim], strides[dim]);
	}
	
	BOOST_CHECK_EQUAL(A_resize.shape().size(), 3);
	BOOST_CHECK_EQUAL(A_resize.raw_storage().strides.size(), 3);
	for(std::size_t dim = 0; dim != 3; ++dim){
		BOOST_CHECK_EQUAL(A_resize.shape()[dim], shape[dim]);
		BOOST_CHECK_EQUAL(A_resize.raw_storage().strides[dim], strides[dim]);
	}
	
	
	
	//check element access
	auto elem = A.elements();
	for(std::size_t i = 0; i != shape[0]; ++i){
		for(std::size_t j = 0; j != shape[1]; ++j){
			for(std::size_t k = 0; k != shape[2]; ++k){
				std::size_t pos = Axis::element(std::array<std::size_t, 3>{i,j,k}, strides);
				BOOST_CHECK_EQUAL(Azero(i,j,k),0.0);
				BOOST_CHECK_EQUAL(A(i,j,k),values[pos]);
				BOOST_CHECK_EQUAL(A_resize(i,j,k),values[pos]);
				BOOST_CHECK_EQUAL(A_cnst(i,j,k),values[pos]);
				BOOST_CHECK_EQUAL(elem(i,j,k),values[pos]);
			}
		}
	}
}

BOOST_AUTO_TEST_SUITE_END();