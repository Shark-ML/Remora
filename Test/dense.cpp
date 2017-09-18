#define BOOST_TEST_MODULE Remora_Vector_Proxy
#include <boost/test/unit_test.hpp>

#include <remora/proxy_expressions.hpp>
#include <remora/dense.hpp>

using namespace remora;

std::size_t Dimensions1 =20;
std::size_t Dimensions2 = 10;
struct ProxyFixture
{
	matrix<int> denseData;
	vector<int> denseDataVec;
	ProxyFixture():denseData(Dimensions1,Dimensions2),denseDataVec(Dimensions1){
		for(std::size_t row=0;row!= Dimensions1;++row){
			for(std::size_t col=0;col!=Dimensions2;++col){
				denseData(row,col) = row*Dimensions2+col+5;
			}
			denseDataVec(row) = row*Dimensions2+6;
		}
	}
};

BOOST_FIXTURE_TEST_SUITE (Remora_vector_proxy, ProxyFixture);

BOOST_AUTO_TEST_CASE( Vector){
	auto const& constDataVec = denseDataVec;
	auto storageVec = denseDataVec.raw_storage();
	BOOST_CHECK_EQUAL(denseDataVec.size(),Dimensions1);
	BOOST_CHECK_EQUAL(storageVec.values,&denseDataVec(0));
	BOOST_CHECK_EQUAL(storageVec.stride,1);
	
	//check that values are correctly aligned in memory
	for(std::size_t i=0;i!= Dimensions1;++i){
		BOOST_CHECK_EQUAL(storageVec.values[i], i*Dimensions2+6);
	}
	
	//check that operator() works
	for(std::size_t i=0;i!= Dimensions1;++i){
		BOOST_CHECK_EQUAL(denseDataVec(i), i*Dimensions2+6);
		BOOST_CHECK_EQUAL(constDataVec(i), i*Dimensions2+6);
	}
	//Check that iterators work
	{
		std::size_t pos = 0;
		for(auto it = denseDataVec.begin();it != denseDataVec.end();++it, ++pos){
			BOOST_CHECK_EQUAL(*it, pos*Dimensions2+6);
			BOOST_CHECK_EQUAL(it.index(), pos);
		}
		BOOST_CHECK_EQUAL(pos, Dimensions1);
	}
	{
		
		std::size_t pos = 0;
		for(auto it = constDataVec.begin();it != constDataVec.end();++it, ++pos){
			BOOST_CHECK_EQUAL(*it, pos*Dimensions2+6);
			BOOST_CHECK_EQUAL(it.index(), pos);
		}
		BOOST_CHECK_EQUAL(pos, Dimensions1);
	}
}

BOOST_AUTO_TEST_CASE( Vector_Closure){
	auto storageVec = denseDataVec.raw_storage();
	{
		vector<int>::closure_type closure = denseDataVec;
		BOOST_CHECK_EQUAL(closure.size(),denseDataVec.size());
		auto storage = closure.raw_storage();
		BOOST_CHECK_EQUAL(storage.values,storageVec.values);
		BOOST_CHECK_EQUAL(storage.stride,1);
	}
	{
		vector<int>::const_closure_type closure = (vector<int> const&) denseDataVec;
		BOOST_CHECK_EQUAL(closure.size(),denseDataVec.size());
		auto storage = closure.raw_storage();
		BOOST_CHECK_EQUAL(storage.values,storageVec.values);
		BOOST_CHECK_EQUAL(storage.stride,1);
	}
	{
		vector<int>::const_closure_type closure = denseDataVec;
		BOOST_CHECK_EQUAL(closure.size(),denseDataVec.size());
		auto storage = closure.raw_storage();
		BOOST_CHECK_EQUAL(storage.values,storageVec.values);
		BOOST_CHECK_EQUAL(storage.stride,1);
	}
}

// vector subrange
BOOST_AUTO_TEST_CASE( Vector_Subrange){
	auto storageVec = denseDataVec.raw_storage();
	{
		vector<int>::closure_type closure = subrange(denseDataVec,3,7);
		BOOST_CHECK_EQUAL(closure.size(),4);
		auto storage = closure.raw_storage();
		BOOST_CHECK_EQUAL(storage.values,storageVec.values + 3);
		BOOST_CHECK_EQUAL(storage.stride,1);
	}
	{
		vector<int>::const_closure_type closure = subrange((vector<int> const&) denseDataVec, 3, 7);
		BOOST_CHECK_EQUAL(closure.size(),4);
		auto storage = closure.raw_storage();
		BOOST_CHECK_EQUAL(storage.values,storageVec.values + 3);
		BOOST_CHECK_EQUAL(storage.stride,1);
	}
	{
		vector<int>::const_closure_type closure = subrange(denseDataVec,3,7);
		BOOST_CHECK_EQUAL(closure.size(),4);
		auto storage = closure.raw_storage();
		BOOST_CHECK_EQUAL(storage.values,storageVec.values + 3);
		BOOST_CHECK_EQUAL(storage.stride,1);
	}
	//now also check from non-trivial proxy
	storageVec.values += 2;
	storageVec.stride = 2;
	vector<int>::closure_type proxy(storageVec, denseDataVec.queue(), (denseDataVec.size()-2)/2);
	vector<int>::const_closure_type const_proxy = proxy;
	{
		vector<int>::closure_type closure = subrange(proxy,3,7);
		BOOST_CHECK_EQUAL(closure.size(),4);
		auto storage = closure.raw_storage();
		BOOST_CHECK_EQUAL(storage.values,storageVec.values + 6);
		BOOST_CHECK_EQUAL(storage.stride,2);
	}
	{
		vector<int>::const_closure_type closure = subrange(const_proxy,3,7);
		BOOST_CHECK_EQUAL(closure.size(),4);
		auto storage = closure.raw_storage();
		BOOST_CHECK_EQUAL(storage.values,storageVec.values + 6);
		BOOST_CHECK_EQUAL(storage.stride,2);
	}
	
	//also check rvalue version
	{
		vector<int>::closure_type closure = subrange(vector<int>::closure_type(proxy),3,7);
		BOOST_CHECK_EQUAL(closure.size(),4);
		auto storage = closure.raw_storage();
		BOOST_CHECK_EQUAL(storage.values,storageVec.values + 6);
		BOOST_CHECK_EQUAL(storage.stride,2);
	}
	{
		vector<int>::const_closure_type closure = subrange(vector<int>::const_closure_type(proxy),3,7);
		BOOST_CHECK_EQUAL(closure.size(),4);
		auto storage = closure.raw_storage();
		BOOST_CHECK_EQUAL(storage.values,storageVec.values + 6);
		BOOST_CHECK_EQUAL(storage.stride,2);
	}
}

BOOST_AUTO_TEST_SUITE_END();