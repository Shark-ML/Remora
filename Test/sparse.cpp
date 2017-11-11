#define BOOST_TEST_MODULE Remora_Sparse_Test
#include <boost/test/unit_test.hpp>

#include <remora/sparse.hpp>

using namespace remora;

BOOST_AUTO_TEST_SUITE (Remora_Sparse)

BOOST_AUTO_TEST_CASE( compressed_vector_set_element){
	//first test for linear insertion
	compressed_vector<unsigned int> v(30);
	{
		std::size_t index[]={1,2,7,9,11,21};
		unsigned int data[]={2,4,8,3,5,2};
		auto pos = v.begin();
		for(int i = 0; i != 6; ++i)
			pos = v.set_element(pos,index[i],data[i]);

		BOOST_REQUIRE_EQUAL(v.nnz(),6);
		BOOST_REQUIRE_EQUAL(v.end() - v.begin(),6);
		BOOST_REQUIRE_EQUAL(v.nnz_capacity(),10);
		
		int i = 0;
		for(auto it = v.begin(); it != v.end(); ++it,++i){
			BOOST_CHECK_EQUAL(*it,data[i]);
			BOOST_CHECK_EQUAL(it.index(),index[i]);
		}
	}
	
	//add additional elements
	{
		std::size_t index[]={1,2,4,7,8, 9,11,16,21};
		unsigned int data[]={2,4,1,8,1,3,5,13,2};
		auto pos = v.begin()+2;
		pos = v.set_element(pos,4,1);
		pos +=1;
		pos = v.set_element(pos,8,1);
		pos +=2;
		pos = v.set_element(pos,16,13);

		BOOST_REQUIRE_EQUAL(v.nnz(),9);
		BOOST_REQUIRE_EQUAL(v.end() - v.begin(),9);
		BOOST_REQUIRE_EQUAL(v.nnz_capacity(),10);
		
		int i = 0;
		for(auto it = v.begin(); it != v.end(); ++it,++i){
			BOOST_CHECK_EQUAL(*it,data[i]);
			BOOST_CHECK_EQUAL(it.index(),index[i]);
		}
	}
}


BOOST_AUTO_TEST_CASE( compressed_vector_clear){
	std::size_t index[]={1,2,4,7,8, 9,11,16,21};
	unsigned int data[]={2,4,1,8,1,3,5,13,2};
	{
		compressed_vector<unsigned int> v(30);
		auto pos = v.begin();
		for(int i = 0; i != 9; ++i)
			pos = v.set_element(pos,index[i],data[i]);
		
		v.clear_range(v.begin()+2,v.begin()+5);

		BOOST_REQUIRE_EQUAL(v.nnz(),6);
		BOOST_REQUIRE_EQUAL(v.end() - v.begin(),6);
		
		int i = 0;
		for(auto it = v.begin(); it != v.end(); ++it,++i){
			BOOST_CHECK_EQUAL(*it,data[i]);
			BOOST_CHECK_EQUAL(it.index(),index[i]);
			if(i==1) i += 3;
		}
	}
	
	{
		compressed_vector<unsigned int> v(30);
		auto pos = v.begin();
		for(int i = 0; i != 9; ++i)
			pos = v.set_element(pos,index[i],data[i]);
		
		v.clear_element(v.begin()+2);

		BOOST_REQUIRE_EQUAL(v.nnz(),8);
		BOOST_REQUIRE_EQUAL(v.end() - v.begin(),8);
		
		int i = 0;
		for(auto it = v.begin(); it != v.end(); ++it,++i){
			BOOST_CHECK_EQUAL(*it,data[i]);
			BOOST_CHECK_EQUAL(it.index(),index[i]);
			if(i==1) ++i;
		}
	}
	
	{
		compressed_vector<unsigned int> v(30);
		auto pos = v.begin();
		for(int i = 0; i != 9; ++i)
			pos = v.set_element(pos,index[i],data[i]);
		
		v.clear();
		BOOST_REQUIRE_EQUAL(v.nnz(),0);
		BOOST_REQUIRE_EQUAL(v.end() - v.begin(),0);
	}
}

BOOST_AUTO_TEST_CASE( compressed_vector_assign){
	std::size_t index[]={1,2,4,7,8, 9,11,16,21};
	unsigned int data[]={2,4,1,8,1,3,5,13,2};
	compressed_vector<unsigned int> v(30);
	auto pos = v.begin();
	for(int i = 0; i != 9; ++i)
		pos = v.set_element(pos,index[i],data[i]);
	
	//copy ctor
	{
		compressed_vector<unsigned int> test(v);
		BOOST_CHECK_EQUAL(test.nnz_capacity(), v.nnz_capacity());
		BOOST_CHECK_EQUAL(test.nnz(), v.nnz());
		BOOST_CHECK_EQUAL(test.size(), v.size());
		
		int i = 0;
		for(auto it = test.begin(); it != test.end(); ++it,++i){
			BOOST_CHECK_EQUAL(*it,data[i]);
			BOOST_CHECK_EQUAL(it.index(),index[i]);
		}
	}
	
	//copy assigment
	{
		compressed_vector<unsigned int> test(10);
		auto pos = test.begin();
		for(int i = 0; i != 3; ++i)
			pos = test.set_element(pos,index[i+3],data[i+3]);
		test = v;
		BOOST_CHECK_EQUAL(test.nnz_capacity(), v.nnz_capacity());
		BOOST_CHECK_EQUAL(test.nnz(), v.nnz());
		BOOST_CHECK_EQUAL(test.size(), v.size());
		
		int i = 0;
		for(auto it = test.begin(); it != test.end(); ++it,++i){
			BOOST_CHECK_EQUAL(*it,data[i]);
			BOOST_CHECK_EQUAL(it.index(),index[i]);
		}
	}
	//copy assignment other container type
	{
		compressed_vector<unsigned long> test(10);
		auto pos = test.begin();
		for(int i = 0; i != 3; ++i)
			pos = test.set_element(pos,index[i+3],data[i+3]);
		test = v;
		BOOST_CHECK_EQUAL(test.nnz_capacity(), v.nnz_capacity());
		BOOST_CHECK_EQUAL(test.nnz(), v.nnz());
		BOOST_CHECK_EQUAL(test.size(), v.size());
		
		int i = 0;
		for(auto it = test.begin(); it != test.end(); ++it,++i){
			BOOST_CHECK_EQUAL(*it,data[i]);
			BOOST_CHECK_EQUAL(it.index(),index[i]);
		}
	}
}

BOOST_AUTO_TEST_CASE( compressed_vector_const_reference){
	std::size_t index[]={1,2,4,7,8, 9,11,16,21};
	unsigned int data[]={2,4,1,8,1,3,5,13,2};
	compressed_vector<unsigned int> v(30);
	auto pos = v.begin();
	for(int i = 0; i != 6; ++i)
		pos = v.set_element(pos,index[i],data[i]);
	auto storage = v.raw_storage();
	
	{
		compressed_vector_reference<unsigned int const, std::size_t const> ref(v);
		auto ref_storage = ref.raw_storage();
		BOOST_CHECK_EQUAL(ref_storage.values, storage.values);
		BOOST_CHECK_EQUAL(ref_storage.indices, storage.indices);
		BOOST_CHECK_EQUAL(ref_storage.nnz, storage.nnz);
		BOOST_CHECK_EQUAL(ref_storage.capacity, storage.capacity);
		
		BOOST_CHECK_EQUAL(ref.nnz_capacity(), ref.nnz());
		BOOST_CHECK_EQUAL(ref.nnz(), v.nnz());
		BOOST_CHECK_EQUAL(ref.size(), v.size());
		
		int i = 0;
		for(auto it = ref.begin(); it != ref.end(); ++it,++i){
			BOOST_CHECK_EQUAL(*it,data[i]);
			BOOST_CHECK_EQUAL(it.index(),index[i]);
		}
	}
	
	//construction from non-const reference
	{
		compressed_vector_reference<unsigned int, std::size_t> ref_non(v);
		compressed_vector_reference<unsigned int const, std::size_t const> ref(ref_non);
		auto ref_storage = ref.raw_storage();
		BOOST_CHECK_EQUAL(ref_storage.values, storage.values);
		BOOST_CHECK_EQUAL(ref_storage.indices, storage.indices);
		BOOST_CHECK_EQUAL(ref_storage.nnz, storage.nnz);
		BOOST_CHECK_EQUAL(ref_storage.capacity, storage.capacity);
		
		BOOST_CHECK_EQUAL(ref.nnz_capacity(), ref.nnz());
		BOOST_CHECK_EQUAL(ref.nnz(), v.nnz());
		BOOST_CHECK_EQUAL(ref.size(), v.size());
		
		int i = 0;
		for(auto it = ref.begin(); it != ref.end(); ++it,++i){
			BOOST_CHECK_EQUAL(*it,data[i]);
			BOOST_CHECK_EQUAL(it.index(),index[i]);
		}
	}
}


BOOST_AUTO_TEST_CASE( compressed_vector_reference_test){
	std::size_t index[]={1,2,4,7,8, 9,11,16,21};
	unsigned int data[]={2,4,1,8,1,3,5,13,2};
	compressed_vector<unsigned int> v(30);
	auto pos = v.begin();
	for(int i = 0; i != 6; ++i)
		pos = v.set_element(pos,index[i],data[i]);
	auto storage = v.raw_storage();
	
	{
		compressed_vector_reference<unsigned int, std::size_t> ref(v);
		auto ref_storage = ref.raw_storage();
		BOOST_CHECK_EQUAL(ref_storage.values, storage.values);
		BOOST_CHECK_EQUAL(ref_storage.indices, storage.indices);
		BOOST_CHECK_EQUAL(ref_storage.nnz, storage.nnz);
		BOOST_CHECK_EQUAL(ref_storage.capacity, storage.capacity);
		
		BOOST_CHECK_EQUAL(ref.nnz_capacity(), ref.nnz_capacity());
		BOOST_CHECK_EQUAL(ref.nnz(), v.nnz());
		BOOST_CHECK_EQUAL(ref.size(), v.size());
		
		int i = 0;
		for(auto it = ref.begin(); it != ref.end(); ++it,++i){
			BOOST_CHECK_EQUAL(*it,data[i]);
			BOOST_CHECK_EQUAL(it.index(),index[i]);
		}
	}
	
	{
		std::size_t index_new[]={0,1,3,6,7, 8};
		unsigned int data_new[]={1,3,0,7,0,2};
		compressed_vector<unsigned long> test(30);
		auto pos = test.begin();
		for(int i = 0; i != 6; ++i)
			pos = test.set_element(pos,index_new[i],data_new[i]);
		test = v;
		BOOST_CHECK_EQUAL(test.nnz(), v.nnz());
		BOOST_CHECK_EQUAL(test.size(), v.size());
		
		int i = 0;
		for(auto it = test.begin(); it != test.end(); ++it,++i){
			BOOST_CHECK_EQUAL(*it,data[i]);
			BOOST_CHECK_EQUAL(it.index(),index[i]);
		}
	}
}

BOOST_AUTO_TEST_SUITE_END();