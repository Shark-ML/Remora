#define BOOST_TEST_MODULE Remora_Proxy
#include <remora/proxy_expressions.hpp>
#include <remora/dense.hpp>

#include <boost/test/included/unit_test.hpp>
#include <boost/mpl/list.hpp>
#include <algorithm>
using namespace remora;
using namespace remora::ax;


struct ProxyFixture
{
	std::vector<unsigned> values;
	ProxyFixture():values(3*20*7){
		for(std::size_t i = 0; i != 3*20*7; ++i){
			values[i] = i;
		}
	}
};

BOOST_FIXTURE_TEST_SUITE (Remora_Proxy_Test, ProxyFixture);

typedef boost::mpl::list<axis<0,1,2>, axis<0,2,1>, axis<1,0,2>, axis<1,2,0>, axis<2,0,1>, axis<2,1,0> > axis_types;
typedef boost::mpl::list<axis<0,1>, axis<1,0> > axis_2d_types;


////////////////////////////////////////////////////
//// PERMUTE
////////////////////////////////////////////////////


BOOST_AUTO_TEST_CASE_TEMPLATE( Dense_Permute, Axis, axis_types ){
	typedef axis<2, 0, 1> axis0;
	tensor_shape<3> shape = {3, 10, 7};
	std::array<std::size_t, 3> strides = {140, 7, 1};
	strides = axis0::to_axis(strides);
	shape = axis0::to_axis(shape);
	
	typedef axis<
		axis0::element_v<Axis::template element_v<0> >,
		axis0::element_v<Axis::template element_v<1> >,
		axis0::element_v<Axis::template element_v<2> >
	> axis_target;
	
	typedef integer_list<bool, 0,1,1>::select_t<2, 0, 1> storage_tag;
	typedef integer_list<bool, 
		storage_tag::element_v<Axis::template element_v<0> >,
		storage_tag::element_v<Axis::template element_v<1> >,
		storage_tag::element_v<Axis::template element_v<2> >
	> storage_tag_target;
	auto target_strides = Axis::to_axis(strides);
	auto target_shape = Axis::to_axis(shape);
	
	dense_tensor_adaptor<unsigned, axis0, storage_tag, cpu_tag> adaptor({values.data(), strides},no_queue(), shape);
	auto adaptor_move = adaptor;
	auto const& adaptor_const = adaptor;
	dense_tensor_adaptor<unsigned, axis_target, storage_tag_target, cpu_tag> result = permute(adaptor, Axis());
	dense_tensor_adaptor<unsigned const, axis_target, storage_tag_target, cpu_tag> result1 = permute(adaptor_const, Axis());
	dense_tensor_adaptor<unsigned const, axis_target, storage_tag_target, cpu_tag> result2 = permute(std::move(adaptor_move), Axis());
	(void)result1; (void)result2;
	
	BOOST_CHECK_EQUAL(result.shape().size(), 3);
	BOOST_CHECK_EQUAL(result.raw_storage().strides.size(), 3);
	BOOST_CHECK_EQUAL(result.raw_storage().values, values.data());
	for(std::size_t dim = 0; dim != 3; ++dim){
		BOOST_CHECK_EQUAL(result.shape()[dim], target_shape[dim]);
		BOOST_CHECK_EQUAL(result.raw_storage().strides[dim], target_strides[dim]);
	}
}

////////////////////////////////////////////////////
//// Diagonal
////////////////////////////////////////////////////


BOOST_AUTO_TEST_CASE( Dense_Diagonal2D ){
	
	std::array<std::size_t, 2> strides = {49, 7};
	tensor_shape<2> shape = {20, 7};
	typedef integer_list<bool, 1, 0> storage_tag;
	dense_tensor_adaptor<unsigned, axis<0,1>, storage_tag, cpu_tag> adaptor({values.data(), strides},no_queue(), shape);
	
	//diagonal will have at most 7 elements, but the lower diagonals should be full size for longer.
	{
		// compute ground truth
		typedef integer_list<bool, 0> storage_tag_target;
		typedef axis<0> axis_target;
		dense_tensor_storage<unsigned, storage_tag_target> target_storage;
		
		target_storage.strides[0] = strides[0]+strides[1];
		target_storage.values = adaptor.raw_storage().values;
		tensor_shape<1> target_shape = {7};

		
		dense_tensor_adaptor<unsigned, axis_target, storage_tag_target, cpu_tag> result = diag(adaptor);
		//shoudl give the same result (just more verbose)
		dense_tensor_adaptor<unsigned, axis_target, storage_tag_target, cpu_tag> result1 = diag(adaptor, axis<0,1>() );
		dense_tensor_adaptor<unsigned, axis_target, storage_tag_target, cpu_tag> result2 = diag(adaptor, axis<1,0>() );
		
		BOOST_CHECK_EQUAL(result.raw_storage().values, target_storage.values);
		BOOST_CHECK_EQUAL(result.shape()[0], target_shape[0]);
		BOOST_CHECK_EQUAL(result.raw_storage().strides[0], target_storage.strides[0]);
		
		BOOST_CHECK_EQUAL(result1.raw_storage().values, target_storage.values);
		BOOST_CHECK_EQUAL(result1.shape()[0], target_shape[0]);
		BOOST_CHECK_EQUAL(result1.raw_storage().strides[0], target_storage.strides[0]);
		
		BOOST_CHECK_EQUAL(result2.raw_storage().values, target_storage.values);
		BOOST_CHECK_EQUAL(result2.shape()[0], target_shape[0]);
		BOOST_CHECK_EQUAL(result2.raw_storage().strides[0], target_storage.strides[0]);
		
		
		//check values because diagonal is a bit complicated
		for(std::size_t i = 0; i != 7; ++i){
			BOOST_CHECK_EQUAL(result(i), adaptor(i,i));
			BOOST_CHECK_EQUAL(result1(i), adaptor(i,i));
			BOOST_CHECK_EQUAL(result2(i), adaptor(i,i));
		}
	}
	
	{
		// compute ground truth
		typedef integer_list<bool, 0> storage_tag_target;
		typedef axis<0> axis_target;
		dense_tensor_storage<unsigned, storage_tag_target> target_storage;
		
		std::ptrdiff_t k = 3;
		target_storage.strides[0] = strides[0]+strides[1];
		target_storage.values = adaptor.raw_storage().values + k * strides[1];
		tensor_shape<1> target_shape = {7 - k};
		
		
		dense_tensor_adaptor<unsigned, axis_target, storage_tag_target, cpu_tag> result = diag(adaptor, k);
		//shoudl give the same result (just more verbose)
		dense_tensor_adaptor<unsigned, axis_target, storage_tag_target, cpu_tag> result1 = diag(adaptor, axis<0,1>(), k );
		dense_tensor_adaptor<unsigned, axis_target, storage_tag_target, cpu_tag> result2 = diag(adaptor, axis<1,0>(), -k );
		
		BOOST_CHECK_EQUAL(result.raw_storage().values, target_storage.values);
		BOOST_CHECK_EQUAL(result.shape()[0], target_shape[0]);
		BOOST_CHECK_EQUAL(result.raw_storage().strides[0], target_storage.strides[0]);
		
		BOOST_CHECK_EQUAL(result1.raw_storage().values, target_storage.values);
		BOOST_CHECK_EQUAL(result1.shape()[0], target_shape[0]);
		BOOST_CHECK_EQUAL(result1.raw_storage().strides[0], target_storage.strides[0]);
		
		BOOST_CHECK_EQUAL(result2.raw_storage().values, target_storage.values);
		BOOST_CHECK_EQUAL(result2.shape()[0], target_shape[0]);
		BOOST_CHECK_EQUAL(result2.raw_storage().strides[0], target_storage.strides[0]);
		
		
		
		//check values because diagonal is a bit complicated
		for(std::size_t i = 0; i != 7; ++i){
			BOOST_CHECK_EQUAL(result(i), adaptor(i, i + k));
			BOOST_CHECK_EQUAL(result1(i), adaptor(i, i + k));
			BOOST_CHECK_EQUAL(result2(i), adaptor(i, i + k));
		}
	}	
	{
		// compute ground truth
		typedef integer_list<bool, 0> storage_tag_target;
		typedef axis<0> axis_target;
		dense_tensor_storage<unsigned, storage_tag_target> target_storage;
		
		std::ptrdiff_t k = -3;
		target_storage.strides[0] = strides[0]+strides[1];
		target_storage.values = adaptor.raw_storage().values - k * strides[0];
		tensor_shape<1> target_shape = {7};
		
		
		dense_tensor_adaptor<unsigned, axis_target, storage_tag_target, cpu_tag> result = diag(adaptor, k);
		//shoudl give the same result (just more verbose)
		dense_tensor_adaptor<unsigned, axis_target, storage_tag_target, cpu_tag> result1 = diag(adaptor, axis<0,1>(), k );
		dense_tensor_adaptor<unsigned, axis_target, storage_tag_target, cpu_tag> result2 = diag(adaptor, axis<1,0>(), -k );
		
		BOOST_CHECK_EQUAL(result.raw_storage().values, target_storage.values);
		BOOST_CHECK_EQUAL(result.shape()[0], target_shape[0]);
		BOOST_CHECK_EQUAL(result.raw_storage().strides[0], target_storage.strides[0]);
		
		BOOST_CHECK_EQUAL(result1.raw_storage().values, target_storage.values);
		BOOST_CHECK_EQUAL(result1.shape()[0], target_shape[0]);
		BOOST_CHECK_EQUAL(result1.raw_storage().strides[0], target_storage.strides[0]);
		
		BOOST_CHECK_EQUAL(result2.raw_storage().values, target_storage.values);
		BOOST_CHECK_EQUAL(result2.shape()[0], target_shape[0]);
		BOOST_CHECK_EQUAL(result2.raw_storage().strides[0], target_storage.strides[0]);
		
		
		
		//check values because diagonal is a bit complicated
		for(std::size_t i = 0; i != 7; ++i){
			BOOST_CHECK_EQUAL(result(i), adaptor(i - k, i));
			BOOST_CHECK_EQUAL(result1(i), adaptor(i - k, i));
			BOOST_CHECK_EQUAL(result2(i), adaptor(i - k, i));
		}
	}
}



BOOST_AUTO_TEST_CASE( Dense_Diagonal3D ){
	
	std::array<std::size_t, 3> strides = {14, 7, 1};
	tensor_shape<3> shape = {10, 2, 7};
	typedef integer_list<bool, 1, 1, 1> storage_tag;
	dense_tensor_adaptor<unsigned, axis<0,1, 2>, storage_tag, cpu_tag> adaptor({values.data(), strides},no_queue(), shape);
	
	{
		// compute ground truth
		typedef integer_list<bool, 0, 0> storage_tag_target;
		typedef axis<1, 0> axis_target;
		dense_tensor_storage<unsigned, storage_tag_target> target_storage;
		
		std::ptrdiff_t k = 3;
		target_storage.strides[0] = strides[1];
		target_storage.strides[1] = strides[0]+strides[2];
		target_storage.values = adaptor.raw_storage().values + k * strides[2];
		tensor_shape<2> target_shape = {2, 7 - k};
		
		//shoudl give the same result (just more verbose)
		dense_tensor_adaptor<unsigned, axis_target, storage_tag_target, cpu_tag> result1 = diag(adaptor, axis_set<0,2>(), k );
		dense_tensor_adaptor<unsigned, axis_target, storage_tag_target, cpu_tag> result2 = diag(adaptor, axis_set<2,0>(), -k );
		
		BOOST_CHECK_EQUAL(result1.raw_storage().values, target_storage.values);
		BOOST_CHECK_EQUAL(result1.shape()[0], target_shape[0]);
		BOOST_CHECK_EQUAL(result1.shape()[1], target_shape[1]);
		BOOST_CHECK_EQUAL(result1.raw_storage().strides[0], target_storage.strides[0]);
		BOOST_CHECK_EQUAL(result1.raw_storage().strides[1], target_storage.strides[1]);
		
		BOOST_CHECK_EQUAL(result2.raw_storage().values, target_storage.values);
		BOOST_CHECK_EQUAL(result2.shape()[0], target_shape[0]);
		BOOST_CHECK_EQUAL(result2.shape()[1], target_shape[1]);
		BOOST_CHECK_EQUAL(result2.raw_storage().strides[0], target_storage.strides[0]);
		BOOST_CHECK_EQUAL(result2.raw_storage().strides[1], target_storage.strides[1]);
		
		
		
		//check values because diagonal is a bit complicated
		for(std::size_t i = 0; i != target_shape[0]; ++i){
			for(std::size_t j = 0; j != target_shape[1]; ++j){
				BOOST_CHECK_EQUAL(result1(i,j), adaptor(j, i, j + k));
				BOOST_CHECK_EQUAL(result2(i, j), adaptor(j, i, j + k));
			}
		}
	}	
}

////////////////////////////////////////////////////
//// SLICE
////////////////////////////////////////////////////

BOOST_AUTO_TEST_CASE_TEMPLATE( Dense_Slice_First_3D, Axis, axis_types ){
	
	std::array<std::size_t, 3> strides = {140, 7, 1};
	tensor_shape<3> shape = {4, 20, 7};
	strides = Axis::to_axis(strides);
	shape = Axis::to_axis(shape);
	typedef integer_list<bool, 1,1,1> storage_tag;
	dense_tensor_adaptor<unsigned, Axis, storage_tag, cpu_tag> adaptor({values.data(), strides},no_queue(), shape);
	
	//offset we want to perform
	std::size_t offset = 3;
	// compute ground truth
	typedef integer_list<bool,
		!(Axis::template element_v<1> + 1 == Axis::template element_v<0>),
		!(Axis::template element_v<2> + 1 == Axis::template element_v<0>)
	> storage_tag_target;
	typedef typename Axis::template slice_t<0> axis_target;
	dense_tensor_storage<unsigned, storage_tag_target> target_storage;
	target_storage.strides[0] = strides[1];
	target_storage.strides[1] = strides[2];
	target_storage.values = adaptor.raw_storage().values + offset * strides[0];
	tensor_shape<2> target_shape = {shape[1], shape[2]};
	
	
	static_assert(std::is_same<typename decltype(slice(adaptor, offset ))::storage_type::dense_axis_tag, storage_tag_target>::value);
	dense_tensor_adaptor<unsigned, axis_target, storage_tag_target, cpu_tag> result = slice(adaptor, offset );
	//shoudl give the same result (just more verbose)
	dense_tensor_adaptor<unsigned, axis_target, storage_tag_target, cpu_tag> result1 = slice(adaptor, offset, same, same );
	
	BOOST_CHECK_EQUAL(result.shape().size(), 2);
	BOOST_CHECK_EQUAL(result.raw_storage().strides.size(), 2);
	BOOST_CHECK_EQUAL(result.raw_storage().values, target_storage.values);
	BOOST_CHECK_EQUAL(result.shape()[0], target_shape[0]);
	BOOST_CHECK_EQUAL(result.raw_storage().strides[0], target_storage.strides[0]);
	BOOST_CHECK_EQUAL(result.shape()[1], target_shape[1]);
	BOOST_CHECK_EQUAL(result.raw_storage().strides[1], target_storage.strides[1]);
	
	BOOST_CHECK_EQUAL(result1.shape().size(), 2);
	BOOST_CHECK_EQUAL(result1.raw_storage().strides.size(), 2);
	BOOST_CHECK_EQUAL(result1.raw_storage().values, target_storage.values);
	BOOST_CHECK_EQUAL(result1.shape()[0], target_shape[0]);
	BOOST_CHECK_EQUAL(result1.raw_storage().strides[0], target_storage.strides[0]);
	BOOST_CHECK_EQUAL(result1.shape()[1], target_shape[1]);
	BOOST_CHECK_EQUAL(result1.raw_storage().strides[1], target_storage.strides[1]);
}

BOOST_AUTO_TEST_CASE_TEMPLATE( Dense_Slice_Second_3D, Axis, axis_types ){
	
	std::array<std::size_t, 3> strides = {140, 7, 1};
	tensor_shape<3> shape = {4, 20, 7};
	typedef integer_list<bool, 1,1,1> storage_tag;
	strides = Axis::to_axis(strides);
	shape = Axis::to_axis(shape);
	dense_tensor_adaptor<unsigned, Axis, storage_tag, cpu_tag> adaptor({values.data(), strides},no_queue(), shape);
	
	//offset we want to perform
	std::size_t offset = 3;
	// compute ground truth
	typedef integer_list<bool,
		!(Axis::template element_v<0> + 1 == Axis::template element_v<1>),
		!(Axis::template element_v<2> + 1 == Axis::template element_v<1>)
	> storage_tag_target;
	typedef typename Axis::template slice_t<1> axis_target;
	dense_tensor_storage<unsigned, storage_tag_target> target_storage;
	target_storage.strides[0] = strides[0];
	target_storage.strides[1] = strides[2];
	target_storage.values = adaptor.raw_storage().values + offset * strides[1];
	tensor_shape<2> target_shape = {shape[0], shape[2]};

	
	dense_tensor_adaptor<unsigned, axis_target, storage_tag_target, cpu_tag> result = slice(adaptor, same, offset );
	//shoudl give the same result (just more verbose)
	dense_tensor_adaptor<unsigned, axis_target, storage_tag_target, cpu_tag> result1 = slice(adaptor, same, offset, same );
	
	BOOST_CHECK_EQUAL(result.shape().size(), 2);
	BOOST_CHECK_EQUAL(result.raw_storage().strides.size(), 2);
	BOOST_CHECK_EQUAL(result.raw_storage().values, target_storage.values);
	BOOST_CHECK_EQUAL(result.shape()[0], target_shape[0]);
	BOOST_CHECK_EQUAL(result.raw_storage().strides[0], target_storage.strides[0]);
	BOOST_CHECK_EQUAL(result.shape()[1], target_shape[1]);
	BOOST_CHECK_EQUAL(result.raw_storage().strides[1], target_storage.strides[1]);
	
	BOOST_CHECK_EQUAL(result1.shape().size(), 2);
	BOOST_CHECK_EQUAL(result1.raw_storage().strides.size(), 2);
	BOOST_CHECK_EQUAL(result1.raw_storage().values, target_storage.values);
	BOOST_CHECK_EQUAL(result1.shape()[0], target_shape[0]);
	BOOST_CHECK_EQUAL(result1.raw_storage().strides[0], target_storage.strides[0]);
	BOOST_CHECK_EQUAL(result1.shape()[1], target_shape[1]);
	BOOST_CHECK_EQUAL(result1.raw_storage().strides[1], target_storage.strides[1]);
}

BOOST_AUTO_TEST_CASE_TEMPLATE( Dense_Slice_01_3D, Axis, axis_types ){
	
	std::array<std::size_t, 3> strides = {140, 7, 1};
	tensor_shape<3> shape = {4, 20, 7};
	typedef integer_list<bool, 1,1,1> storage_tag;
	strides = Axis::to_axis(strides);
	shape = Axis::to_axis(shape);
	dense_tensor_adaptor<unsigned, Axis, storage_tag, cpu_tag> adaptor({values.data(), strides},no_queue(), shape);
	
	//offset we want to perform
	std::size_t offset_0 = 3;
	std::size_t offset_1 = 2;
	// compute ground truth
	typedef integer_list<bool,
		Axis::template element_v<2> == 2
	> storage_tag_target;
	dense_tensor_storage<unsigned, storage_tag_target> target_storage;
	target_storage.strides[0] = strides[2];
	target_storage.values = adaptor.raw_storage().values + offset_0 * strides[0] + offset_1 * strides[1];
	tensor_shape<1> target_shape = {shape[2]};
	
	dense_tensor_adaptor<unsigned, axis<0>, storage_tag_target, cpu_tag> result = slice(adaptor, offset_0, offset_1 );
	//shoudl give the same result (just more verbose)
	dense_tensor_adaptor<unsigned, axis<0>, storage_tag_target, cpu_tag> result1 = slice(adaptor, offset_0, offset_1, same );
	
	BOOST_CHECK_EQUAL(result.shape().size(), 1);
	BOOST_CHECK_EQUAL(result.raw_storage().strides.size(), 1);
	BOOST_CHECK_EQUAL(result.raw_storage().values, target_storage.values);
	BOOST_CHECK_EQUAL(result.shape()[0], target_shape[0]);
	BOOST_CHECK_EQUAL(result.raw_storage().strides[0], target_storage.strides[0]);
	
	BOOST_CHECK_EQUAL(result1.shape().size(), 1);
	BOOST_CHECK_EQUAL(result1.raw_storage().strides.size(), 1);
	BOOST_CHECK_EQUAL(result1.raw_storage().values, target_storage.values);
	BOOST_CHECK_EQUAL(result1.shape()[0], target_shape[0]);
	BOOST_CHECK_EQUAL(result1.raw_storage().strides[0], target_storage.strides[0]);
}

BOOST_AUTO_TEST_CASE_TEMPLATE( Dense_Slice_02_3D, Axis, axis_types ){
	
	std::array<std::size_t, 3> strides = {140, 7, 1};
	tensor_shape<3> shape = {4, 20, 7};
	typedef integer_list<bool, 1,1,1> storage_tag;
	strides = Axis::to_axis(strides);
	shape = Axis::to_axis(shape);
	dense_tensor_adaptor<unsigned, Axis, storage_tag, cpu_tag> adaptor({values.data(), strides},no_queue(), shape);
	
	//offset we want to perform
	std::size_t offset_0 = 3;
	std::size_t offset_2 = 2;
	// compute ground truth
	typedef integer_list<bool,
		Axis::template element_v<1> == 2
	> storage_tag_target;
	dense_tensor_storage<unsigned, storage_tag_target> target_storage;
	target_storage.strides[0] = strides[1];
	target_storage.values = adaptor.raw_storage().values + offset_0 * strides[0] + offset_2 * strides[2];
	tensor_shape<1> target_shape = {shape[1]};
	
	dense_tensor_adaptor<unsigned, axis<0>, storage_tag_target, cpu_tag> result = slice(adaptor, offset_0, same, offset_2 );
	
	BOOST_CHECK_EQUAL(result.shape().size(), 1);
	BOOST_CHECK_EQUAL(result.raw_storage().strides.size(), 1);
	BOOST_CHECK_EQUAL(result.raw_storage().values, target_storage.values);
	BOOST_CHECK_EQUAL(result.shape()[0], target_shape[0]);
	BOOST_CHECK_EQUAL(result.raw_storage().strides[0], target_storage.strides[0]);
}

BOOST_AUTO_TEST_CASE_TEMPLATE( Dense_Slice_12_3D, Axis, axis_types ){
	
	std::array<std::size_t, 3> strides = {140, 7, 1};
	tensor_shape<3> shape = {4, 20, 7};
	typedef integer_list<bool, 1,1,1> storage_tag;
	strides = Axis::to_axis(strides);
	shape = Axis::to_axis(shape);
	dense_tensor_adaptor<unsigned, Axis, storage_tag, cpu_tag> adaptor({values.data(), strides},no_queue(), shape);
	
	//offset we want to perform
	std::size_t offset_1 = 3;
	std::size_t offset_2 = 2;
	// compute ground truth
	typedef integer_list<bool,
		Axis::template element_v<0> == 2
	> storage_tag_target;
	dense_tensor_storage<unsigned, storage_tag_target> target_storage;
	target_storage.strides[0] = strides[0];
	target_storage.values = adaptor.raw_storage().values + offset_1 * strides[1] + offset_2 * strides[2];
	tensor_shape<1> target_shape = {shape[0]};
	
	dense_tensor_adaptor<unsigned, axis<0>, storage_tag_target, cpu_tag> result = slice(adaptor, same, offset_1, offset_2 );
	
	BOOST_CHECK_EQUAL(result.shape().size(), 1);
	BOOST_CHECK_EQUAL(result.raw_storage().strides.size(), 1);
	BOOST_CHECK_EQUAL(result.raw_storage().values, target_storage.values);
	BOOST_CHECK_EQUAL(result.shape()[0], target_shape[0]);
	BOOST_CHECK_EQUAL(result.raw_storage().strides[0], target_storage.strides[0]);
}

BOOST_AUTO_TEST_CASE_TEMPLATE( Dense_Slice_Full, Axis, axis_types ){
	
	std::array<std::size_t, 3> strides = {140, 7, 1};
	tensor_shape<3> shape = {4, 20, 7};
	strides = Axis::to_axis(strides);
	shape = Axis::to_axis(shape);
	dense_tensor_adaptor<unsigned, Axis, integer_list<bool, 1,1,1>, cpu_tag> adaptor({values.data(), strides},no_queue(), shape);
	
	//offset we want to perform
	std::size_t offset_0 = 2;
	std::size_t offset_1 = 1;
	std::size_t offset_2 = 3;
	// compute ground truth
	unsigned target_value = values[offset_0 * strides[0] + offset_1 * strides[1] + offset_2 * strides[2]];

	
	unsigned result = slice(adaptor, offset_0, offset_1, offset_2 );
	
	BOOST_CHECK_EQUAL(result, target_value);
}


BOOST_AUTO_TEST_CASE_TEMPLATE( Dense_Subrange_First_3D, Axis, axis_types ){
	
	std::array<std::size_t, 3> strides = {140, 7, 1};
	tensor_shape<3> shape = {4, 10, 7};
	typedef integer_list<bool, 0,1,1>::select_t<
		Axis::template element_v<0>,Axis::template element_v<1>, Axis::template element_v<2>
	> storage_tag;
	strides = Axis::to_axis(strides);
	shape = Axis::to_axis(shape);
	dense_tensor_adaptor<unsigned, Axis, storage_tag, cpu_tag> adaptor({values.data(), strides},no_queue(), shape);
	
	//offset we want to perform
	std::size_t start = 1;
	std::size_t end = 3;
	// compute ground truth
	typedef integer_list<bool,
		storage_tag::template element_v<0>,
		storage_tag::template element_v<1> && !(Axis::template element_v<1> + 1 == Axis::template element_v<0>),
		storage_tag::template element_v<2> && !(Axis::template element_v<2> + 1 == Axis::template element_v<0>)
	> storage_tag_target;
	auto target_storage = adaptor.raw_storage();
	target_storage.values  += start * strides[0];
	tensor_shape<3> target_shape = {end - start, shape[1], shape[2]};

	
	dense_tensor_adaptor<unsigned, Axis, storage_tag_target, cpu_tag> result = slice(adaptor, range(start,end) );
	//shoudl give the same result (just more verbose)
	dense_tensor_adaptor<unsigned, Axis, storage_tag_target, cpu_tag> result1 = slice(adaptor, range(start,end), same, same );
	
	BOOST_CHECK_EQUAL(result.shape().size(), 3);
	BOOST_CHECK_EQUAL(result.raw_storage().strides.size(), 3);
	BOOST_CHECK_EQUAL(result.raw_storage().values, target_storage.values);
	BOOST_CHECK_EQUAL(result.shape()[0], target_shape[0]);
	BOOST_CHECK_EQUAL(result.raw_storage().strides[0], target_storage.strides[0]);
	BOOST_CHECK_EQUAL(result.shape()[1], target_shape[1]);
	BOOST_CHECK_EQUAL(result.raw_storage().strides[1], target_storage.strides[1]);
	BOOST_CHECK_EQUAL(result.shape()[2], target_shape[2]);
	BOOST_CHECK_EQUAL(result.raw_storage().strides[2], target_storage.strides[2]);
	
	BOOST_CHECK_EQUAL(result1.shape().size(), 3);
	BOOST_CHECK_EQUAL(result1.raw_storage().strides.size(), 3);
	BOOST_CHECK_EQUAL(result1.raw_storage().values, target_storage.values);
	BOOST_CHECK_EQUAL(result1.shape()[0], target_shape[0]);
	BOOST_CHECK_EQUAL(result1.raw_storage().strides[0], target_storage.strides[0]);
	BOOST_CHECK_EQUAL(result1.shape()[1], target_shape[1]);
	BOOST_CHECK_EQUAL(result1.raw_storage().strides[1], target_storage.strides[1]);
	BOOST_CHECK_EQUAL(result1.shape()[2], target_shape[2]);
	BOOST_CHECK_EQUAL(result1.raw_storage().strides[2], target_storage.strides[2]);
}



//combination of split and subrange in second position to get a removed axis before subrange
BOOST_AUTO_TEST_CASE_TEMPLATE( Dense_Slice_Subrange, Axis, axis_types ){
	
	std::array<std::size_t, 3> strides = {140, 7, 1};
	tensor_shape<3> shape = {4, 10, 7};
	strides = Axis::to_axis(strides);
	shape = Axis::to_axis(shape);
	typedef integer_list<bool, 0,1,1>::select_t<
		Axis::template element_v<0>,Axis::template element_v<1>, Axis::template element_v<2>
	> storage_tag;
	dense_tensor_adaptor<unsigned, Axis, storage_tag, cpu_tag> adaptor({values.data(), strides},no_queue(), shape);
	
	//offset we want to perform
	std::size_t offset_0 = 2;
	std::size_t start_2 = 1;
	std::size_t end_2 = 3;
	// compute ground truth
	typedef integer_list<bool,
		storage_tag::template element_v<1> && (Axis::template element_v<1> == 2),
		storage_tag::template element_v<2> && (Axis::template element_v<2> == 2)
	> storage_tag_target;
	typedef typename Axis::template slice_t<0> axis_target;
	dense_tensor_storage< unsigned, storage_tag_target> target_storage;
	target_storage.values = adaptor.raw_storage().values + offset_0 * strides[0] + start_2 * strides[2];
	target_storage.strides[0] = strides[1];
	target_storage.strides[1] = strides[2];
	tensor_shape<2> target_shape = {shape[1], end_2 - start_2};

	
	dense_tensor_adaptor<unsigned, axis_target, storage_tag_target, cpu_tag> result = slice(adaptor, offset_0, same, range(start_2,end_2) );
	
	BOOST_CHECK_EQUAL(result.shape().size(), 2);
	BOOST_CHECK_EQUAL(result.raw_storage().strides.size(), 2);
	BOOST_CHECK_EQUAL(result.raw_storage().values, target_storage.values);
	BOOST_CHECK_EQUAL(result.shape()[0], target_shape[0]);
	BOOST_CHECK_EQUAL(result.raw_storage().strides[0], target_storage.strides[0]);
	BOOST_CHECK_EQUAL(result.shape()[1], target_shape[1]);
	BOOST_CHECK_EQUAL(result.raw_storage().strides[1], target_storage.strides[1]);
}

////////////////////////////////////////////////////
//// SPLIT
////////////////////////////////////////////////////

BOOST_AUTO_TEST_CASE( Dense_Split_1D){
	typedef integer_list<bool, 0> storage_tag;
	dense_tensor_adaptor<unsigned, axis<0>, storage_tag, cpu_tag> adaptor({values.data(), {3}},no_queue(), 4*5*7);
	typedef integer_list<bool, 1,1,0> target_storage_tag;
	dense_tensor_adaptor<unsigned, axis<0,1,2>, target_storage_tag, cpu_tag> result = reshape(adaptor, split<3>(4,5,7));
	
	
	BOOST_CHECK_EQUAL(result.raw_storage().values, values.data());
	BOOST_CHECK_EQUAL(result.raw_storage().strides[0], 3*7*5);
	BOOST_CHECK_EQUAL(result.raw_storage().strides[1], 3*7);
	BOOST_CHECK_EQUAL(result.raw_storage().strides[2], 3);
	BOOST_CHECK_EQUAL(result.shape()[0], 4);
	BOOST_CHECK_EQUAL(result.shape()[1], 5);
	BOOST_CHECK_EQUAL(result.shape()[2], 7);	
}

BOOST_AUTO_TEST_CASE_TEMPLATE( Dense_Split_3D, Axis, axis_types){
	tensor_shape<3> shape = {2,4*5*2,3};
	typedef integer_list<bool, 1, 1, 1> storage_tag;
	auto strides = Axis::compute_dense_strides(shape).shape_array;
	dense_tensor_adaptor<unsigned, Axis, storage_tag, cpu_tag> adaptor({values.data(), strides},no_queue(), shape);
	{
		typedef axis<
			Axis::template element_v<0> + 2 * (Axis::template element_v<0> > Axis::template element_v<1>),
			Axis::template element_v<1>,
			Axis::template element_v<1> + 1,
			Axis::template element_v<1> + 2,
			Axis::template element_v<2> + 2 * (Axis::template element_v<2> > Axis::template element_v<1>)
		> axis_target;
		typedef integer_list<bool, 1, 1, 1, 1, 1> target_storage_tag;
		dense_tensor_adaptor<unsigned, axis_target, target_storage_tag, cpu_tag> result = reshape(adaptor, same, split<3>(4,5,2), same);
		BOOST_CHECK_EQUAL(result.raw_storage().values, values.data());
		BOOST_CHECK_EQUAL(result.raw_storage().strides[0], adaptor.raw_storage().strides[0]);
		BOOST_CHECK_EQUAL(result.raw_storage().strides[1], adaptor.raw_storage().strides[1] * 2 * 5);
		BOOST_CHECK_EQUAL(result.raw_storage().strides[2], adaptor.raw_storage().strides[1] * 2);
		BOOST_CHECK_EQUAL(result.raw_storage().strides[3], adaptor.raw_storage().strides[1]);
		BOOST_CHECK_EQUAL(result.raw_storage().strides[4], adaptor.raw_storage().strides[2]);
		BOOST_CHECK_EQUAL(result.shape()[0], 2);
		BOOST_CHECK_EQUAL(result.shape()[1], 4);
		BOOST_CHECK_EQUAL(result.shape()[2], 5);
		BOOST_CHECK_EQUAL(result.shape()[3], 2);
		BOOST_CHECK_EQUAL(result.shape()[4], 3);
	}
}


////////////////////////////////////////////////////
//// MERGE
////////////////////////////////////////////////////

BOOST_AUTO_TEST_CASE( Dense_Merge_2){
	{
		typedef integer_list<bool, 0, 1, 0, 1> storage_tag;
		dense_tensor_adaptor<unsigned, axis<3, 1, 2, 0>, storage_tag, cpu_tag> adaptor({values.data(), {4, 27,9,108}},no_queue(),  {2, 4,3,3});
		typedef integer_list<bool, 0,0, 1> target_storage_tag;
		dense_tensor_adaptor<unsigned, axis<2,1,0>, target_storage_tag, cpu_tag> result = reshape(adaptor, same, merge<2>(), same);
		
		BOOST_CHECK_EQUAL(result.raw_storage().values, values.data());
		BOOST_CHECK_EQUAL(result.raw_storage().strides[0], 4);
		BOOST_CHECK_EQUAL(result.raw_storage().strides[1], 9);
		BOOST_CHECK_EQUAL(result.raw_storage().strides[2], 108);
		BOOST_CHECK_EQUAL(result.shape()[0], 2);
		BOOST_CHECK_EQUAL(result.shape()[1], 12);
		BOOST_CHECK_EQUAL(result.shape()[2], 3);
	}
	{
		typedef integer_list<bool, 0, 1, 1, 1> storage_tag;
		dense_tensor_adaptor<unsigned, axis<3, 1,2, 0>, storage_tag, cpu_tag> adaptor({values.data(), {4, 24,8,96}},no_queue(),  {2, 4,3,3});
		typedef integer_list<bool, 0,1, 1> target_storage_tag;
		dense_tensor_adaptor<unsigned, axis<2,1,0>, target_storage_tag, cpu_tag> result = reshape(adaptor, same, merge<2>(), same);
		
		
		BOOST_CHECK_EQUAL(result.raw_storage().values, values.data());
		BOOST_CHECK_EQUAL(result.raw_storage().strides[0], 4);
		BOOST_CHECK_EQUAL(result.raw_storage().strides[1], 8);
		BOOST_CHECK_EQUAL(result.raw_storage().strides[2], 96);
		BOOST_CHECK_EQUAL(result.shape()[0], 2);
		BOOST_CHECK_EQUAL(result.shape()[1], 12);
		BOOST_CHECK_EQUAL(result.shape()[2], 3);
	}
}

BOOST_AUTO_TEST_CASE( Dense_Merge_3){
	typedef integer_list<bool, 0, 1, 1, 0> storage_tag;
	dense_tensor_adaptor<unsigned, axis<3, 0, 1, 2>, storage_tag, cpu_tag> adaptor({values.data(), {4, 108, 27,9}},no_queue(),  {2, 3, 4,3});
	typedef integer_list<bool, 0,0> target_storage_tag;
	dense_tensor_adaptor<unsigned, axis<1,0>, target_storage_tag, cpu_tag> result = reshape(adaptor, same, merge<3>());
	
	
	BOOST_CHECK_EQUAL(result.raw_storage().values, values.data());
	BOOST_CHECK_EQUAL(result.raw_storage().strides[0], 4);
	BOOST_CHECK_EQUAL(result.raw_storage().strides[1], 9);
	BOOST_CHECK_EQUAL(result.shape()[0], 2);
	BOOST_CHECK_EQUAL(result.shape()[1], 36);
}


BOOST_AUTO_TEST_CASE( Dense_Merge_Proxy){
	typedef dense_tensor_adaptor<unsigned, axis<3, 1, 2, 0>, integer_list<bool, 0, 1, 0, 1>, cpu_tag> tensor_type;
	typedef dense_tensor_adaptor<unsigned, axis<1, 0>, integer_list<bool, 0, 0>, cpu_tag> sliced_tensor_type1;
	typedef dense_tensor_adaptor<unsigned, axis<1, 2, 0>, integer_list<bool, 1, 0, 1>, cpu_tag> sliced_tensor_type2;
	tensor_type adaptor({values.data(), {4, 27,9,108}},no_queue(),  {2, 4,3,2});
	merge_proxy<tensor_type, 2> op = reshape(adaptor, same, same, merge<2>());
	sliced_tensor_type1 op_sliced1 = slice(op,ax::same, ax::same, 5);
	merge_proxy<sliced_tensor_type2, 1> op_sliced2 = slice(op,1);
	BOOST_CHECK_EQUAL(op.shape()[0], 2);
	BOOST_CHECK_EQUAL(op.shape()[1], 4);
	BOOST_CHECK_EQUAL(op.shape()[2], 6);
	
	tensorN<int, 3> result({2,4,6},0);
	tensorN<int, 3> result_plus({2,4,6},1);
	assign(result, op);
	plus_assign(result_plus, op);
	
	//test proxies
	tensor<int, axis<2,0,1>, cpu_tag > result_permute = op;
	tensor<int, axis<1,0>, cpu_tag > result_slice1 = op_sliced1;
	tensor<int, axis<1,0>, cpu_tag > result_slice2 = op_sliced2;

	for(std::size_t i = 0; i != 2; ++i){
		for(std::size_t j = 0; j != 4; ++j){
			for(std::size_t k = 0; k != 3; ++k){
				for(std::size_t l = 0; l != 2; ++l){
					BOOST_CHECK_EQUAL(adaptor(i,j,k,l), result(i,j, k * 2 + l));
					BOOST_CHECK_EQUAL(adaptor(i,j,k,l)+1, result_plus(i,j, k * 2 + l));
					BOOST_CHECK_EQUAL(adaptor(i,j,k,l), result_permute(i,j, k * 2 + l));
					BOOST_CHECK_EQUAL(adaptor(1,j,k,l), result_slice2(j,k * 2 + l));
				}
			}
			BOOST_CHECK_EQUAL(adaptor(i,j,2,1), result_slice1(i,j));
		}
	}
}

//testing split after merge
BOOST_AUTO_TEST_CASE( Dense_Merge_Proxy_Split){
	typedef dense_tensor_adaptor<unsigned, axis<0, 2, 1>, integer_list<bool, 1, 1, 1>, cpu_tag> tensor_type;
	typedef dense_tensor_adaptor<unsigned, axis<0, 3, 1, 2>, integer_list<bool, 1, 1, 1, 1>, cpu_tag> result_tensor_type;
	tensor_type adaptor({values.data(), {24, 1, 4}},no_queue(),  {2, 4,6});
	merge_proxy<result_tensor_type, 0> op = reshape(adaptor, merge<2>(), split<2>(2,3));
	//advanced check. we first permute adaptor to a different shape, than merge/split according to permutaton and permute back.
	//this should give the same final type of expression as above and indeed everything should be equal.
	//However, it goes through a different construction path, so testing this invariance is important
	auto adapt_permuted = permute(adaptor, axis<2,0,1>());
	auto merged = reshape(adapt_permuted, ax::same, merge<2>());
	merge_proxy<result_tensor_type, 0> op2 = permute(reshape(merged, split<2>(2,3), ax::same),axis<2,0,1>());
	
	BOOST_CHECK_EQUAL(op.shape()[0], 8);
	BOOST_CHECK_EQUAL(op.shape()[1], 2);
	BOOST_CHECK_EQUAL(op.shape()[2], 3);
	
	tensorN<int, 3> result = op;
	tensorN<int, 3> result2 = op2;

	for(std::size_t i = 0; i != 2; ++i){
		for(std::size_t j = 0; j != 4; ++j){
			for(std::size_t k = 0; k != 2; ++k){
				for(std::size_t l = 0; l != 3; ++l){
					BOOST_CHECK_EQUAL(adaptor(i,j,k * 3 + l), result(i * 4 + j, k, l));
					BOOST_CHECK_EQUAL(adaptor(i,j,k * 3 + l), result2(i * 4 + j, k, l));
				}
			}
		}
	}
}
////////////////////////////////////////////////////
//// SCALAR SPECIAL CASES
////////////////////////////////////////////////////
BOOST_AUTO_TEST_CASE( Scalar_Proxy_Test){
	scalar<unsigned> x(23);
	//slice
	BOOST_CHECK_EQUAL(unsigned(slice(x)),23);
	BOOST_CHECK_EQUAL(unsigned(reshape(x)),23);
	BOOST_CHECK_EQUAL(unsigned(permute(x,axis<>())),23);
}
	
	


BOOST_AUTO_TEST_SUITE_END();