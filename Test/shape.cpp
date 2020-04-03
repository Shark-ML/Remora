#define BOOST_TEST_MODULE Remora_Shape
#include <boost/test/included/unit_test.hpp>

#include <remora/detail/shape.hpp>
using namespace remora;

BOOST_AUTO_TEST_SUITE(Remora_Shape_Test)


BOOST_AUTO_TEST_CASE( Shape_creation){
	tensor_shape<4> sh = {7,2,3,4};
	BOOST_CHECK_EQUAL(sh.size(), 4);
	BOOST_CHECK_EQUAL(sh.num_elements(), 168);
	auto const& sh_cnst = sh;
	BOOST_CHECK_EQUAL(sh[0], 7);
	BOOST_CHECK_EQUAL(sh[1], 2);
	BOOST_CHECK_EQUAL(sh[2], 3);
	BOOST_CHECK_EQUAL(sh[3], 4);
	
	BOOST_CHECK_EQUAL(sh_cnst[0], 7);
	BOOST_CHECK_EQUAL(sh_cnst[1], 2);
	BOOST_CHECK_EQUAL(sh_cnst[2], 3);
	BOOST_CHECK_EQUAL(sh_cnst[3], 4);
	
	tensor_shape<4> sh_copy = sh;
	tensor_shape<4> sh2 = {7,2,5,4};
	BOOST_CHECK(sh == sh_copy);
	BOOST_CHECK(!(sh == sh2));
	BOOST_CHECK(sh != sh2);
	BOOST_CHECK(!(sh != sh_copy));
}

BOOST_AUTO_TEST_SUITE_END();