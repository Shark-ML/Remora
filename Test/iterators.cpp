#define BOOST_TEST_MODULE Remora_Vector_vector_iterators
#include <boost/test/unit_test.hpp>
#include <boost/test/floating_point_comparison.hpp>

#include <remora/cpu/iterator.hpp>
#include <remora/detail/traits.hpp>

using namespace remora;

BOOST_AUTO_TEST_SUITE (Remora_iterators)

BOOST_AUTO_TEST_CASE( Remora_Dense_Storage_Iterator)
{
	double values[]={0.1,0.2,0.3,0.4,0.5,0.6};
	
	//reading
	{
		iterators::dense_storage_iterator<const double> iter(values+2,1,2);
		iterators::dense_storage_iterator<const double> start=iter;
		iterators::dense_storage_iterator<const double> end(values+6,3,2);
		BOOST_REQUIRE_EQUAL(end-start, 2);
		BOOST_REQUIRE_EQUAL(start-iter, 0);
		BOOST_REQUIRE(start == iter);
		BOOST_REQUIRE(end != start);
		BOOST_REQUIRE(end == start+2);
		std::size_t k = 1;
		while(iter != end){
			BOOST_CHECK_EQUAL(iter.index(),k);
			BOOST_CHECK_EQUAL(*iter,values[2*k]);
			BOOST_CHECK_EQUAL(start[k-1],values[2*k]);
			BOOST_CHECK_EQUAL(*(start+k-1),values[2*k]);
			BOOST_CHECK(iter < end);
			++iter;
			++k;
		}
		BOOST_REQUIRE_EQUAL(end-iter, 0);
		BOOST_REQUIRE_EQUAL(iter-start, 2);
		BOOST_REQUIRE_EQUAL(k, 3);
	}
	
	//writing
	{
		iterators::dense_storage_iterator<double> iter(values,0,2);
		iterators::dense_storage_iterator<double> end(values,3,2);
		std::size_t k = 0;
		while(iter != end){
			*iter = k;
			++k;
			++iter;
		}
		for(std::size_t i = 0; i != 6; ++i){
			if(i% 2 == 0)
				BOOST_CHECK_EQUAL(values[i],i/2);
			else
				BOOST_CHECK_CLOSE(values[i],i*0.1+0.1, 1.e-10);
		}
	}
}

BOOST_AUTO_TEST_CASE( Remora_Compressed_Storage_Iterator)
{
	double values[]={0.1,0.2,0.3,0.4,0.5,0.6};
	std::size_t indizes[]={3,8,11,12,15,16};
	
	//reading
	{
		iterators::compressed_storage_iterator<const double,const std::size_t> iter(values,indizes,1,2);
		iterators::compressed_storage_iterator<const double,const std::size_t> start=iter;
		iterators::compressed_storage_iterator<const double,const std::size_t> end(values,indizes,5,2);
		BOOST_REQUIRE_EQUAL(start.major_index(), 2);
		BOOST_REQUIRE_EQUAL(start-iter, 0);
		BOOST_REQUIRE_EQUAL(end-start, 4);
		BOOST_REQUIRE(start == iter);
		BOOST_REQUIRE(end != start);
		std::size_t k = 1;
		while(iter != end){
			BOOST_CHECK_EQUAL(iter.index(),indizes[k]);
			BOOST_CHECK_EQUAL(*iter,values[k]);
			++iter;
			++k;
		}
		BOOST_REQUIRE_EQUAL(end-iter, 0);
		BOOST_REQUIRE_EQUAL(iter-start, 4);
		BOOST_REQUIRE_EQUAL(k, 5);
	}
	//writing
	{
		iterators::compressed_storage_iterator<double,const std::size_t> iter(values,indizes,1,2);
		iterators::compressed_storage_iterator<double,const std::size_t> end(values,indizes,5,2);
		std::size_t k = 1;
		while(iter != end){
			*iter = 2*k;
			++iter;
			++k;
		}
		BOOST_REQUIRE_EQUAL(k, 5);
		for(std::size_t i = 1;  i !=5; ++i){
			BOOST_CHECK_EQUAL(values[i],2*i); 
		}
	}
}

struct IndexedMocup{
	typedef double value_type;
	typedef double& reference;
	typedef double const& const_reference;
	
	IndexedMocup(double* array,std::size_t size):m_array(array),m_size(size){}
	
	std::size_t size()const{
		return m_size;
	}
	reference operator()(std::size_t i)const{
		return m_array[i];
	}
	
	bool same_closure(IndexedMocup const& other)const{
		return m_array == other.m_array;
	}
	
	double* m_array;
	std::size_t m_size;
};

struct ConstIndexedMocup{
	typedef double value_type;
	typedef double& reference;
	typedef double const& const_reference;
	
	ConstIndexedMocup(double* array,std::size_t size):m_array(array),m_size(size){}
	
	std::size_t size()const{
		return m_size;
	}
	const_reference operator()(std::size_t i)const{
		return m_array[i];
	}
	
	bool same_closure(ConstIndexedMocup const& other)const{
		return m_array == other.m_array;
	}
	
	double* m_array;
	std::size_t m_size;
};




BOOST_AUTO_TEST_CASE( Remora_Indexed_Iterator)
{
	double values[]={0.1,0.2,0.3,0.4,0.5,0.6};
	
	//reading
	{
		ConstIndexedMocup mocup(values,6);
		iterators::indexed_iterator<const ConstIndexedMocup> iter(mocup,1);
		iterators::indexed_iterator<const ConstIndexedMocup> start=iter;
		iterators::indexed_iterator<const ConstIndexedMocup> end(mocup,5);
		BOOST_REQUIRE_EQUAL(end-start, 4);
		BOOST_REQUIRE_EQUAL(start-iter, 0);
		BOOST_REQUIRE(start == iter);
		BOOST_REQUIRE(end != start);
		BOOST_REQUIRE(start < end);
		BOOST_REQUIRE(end == start+4);
		std::size_t k = 1;
		while(iter != end){
			BOOST_CHECK_EQUAL(iter.index(),k);
			BOOST_CHECK_EQUAL(*iter,values[k]);
			BOOST_CHECK_EQUAL(start[k-1],values[k]);
			BOOST_CHECK_EQUAL(*(start+k-1),values[k]);
			BOOST_CHECK(iter < end);
			++iter;
			++k;
		}
		BOOST_REQUIRE_EQUAL(end-iter, 0);
		BOOST_REQUIRE_EQUAL(iter-start, 4);
		BOOST_REQUIRE_EQUAL(k, 5);
	}
	
	//writing
	{
		IndexedMocup mocup(values,6);
		iterators::indexed_iterator<IndexedMocup> iter(mocup,1);
		iterators::indexed_iterator<IndexedMocup> end(mocup,5);
		std::size_t k = 1;
		while(iter != end){
			*iter = 2*k;
			++iter;
			++k;
		}
		BOOST_REQUIRE_EQUAL(k, 5);
		for(std::size_t i = 1;  i !=5; ++i){
			BOOST_CHECK_EQUAL(values[i],2*i); 
		}
	}
	
}

BOOST_AUTO_TEST_CASE( Remora_Constant_Iterator)
{
	iterators::constant_iterator<double> iter(4.0,3);
	iterators::constant_iterator<double> start =iter;
	iterators::constant_iterator<double> end(4.0,10);
	BOOST_REQUIRE_EQUAL(start-iter, 0);
	BOOST_REQUIRE_EQUAL(end-start, 7);
	BOOST_REQUIRE(start == iter);
	BOOST_REQUIRE(end != start);
	BOOST_REQUIRE(start < end);
	std::size_t k = 3;
	while(iter != end){
		BOOST_CHECK_EQUAL(iter.index(),k);
		BOOST_CHECK_EQUAL(*iter,4.0);
		++iter;
		++k;
	}
	BOOST_REQUIRE_EQUAL(end-iter, 0);
	BOOST_REQUIRE_EQUAL(iter-start, 7);
	BOOST_REQUIRE_EQUAL(k, 10);
}

BOOST_AUTO_TEST_CASE( Remora_Transform_Iterator_Dense)
{
	double values[]={0.1,0.2,0.3,0.4,0.5,0.6};
	typedef device_traits<cpu_tag>::sqr<double> functor;

	typedef iterators::dense_storage_iterator<const double> iterator;
	iterator dense_iter(values,0);
	iterator dense_end(values,6);
	iterators::transform_iterator<iterator,functor > iter(dense_iter,functor());
	iterators::transform_iterator<iterator,functor > start = iter;
	iterators::transform_iterator<iterator,functor > end(dense_end,functor());
	
	BOOST_REQUIRE_EQUAL(end-start, 6);
	BOOST_REQUIRE_EQUAL(start-iter, 0);
	BOOST_REQUIRE(start == iter);
	BOOST_REQUIRE(start != end);
	BOOST_REQUIRE(start < end);
	BOOST_REQUIRE(end == start+6);
	std::size_t k = 0;
	while(iter != end){
		BOOST_CHECK_EQUAL(iter.index(),k);
		BOOST_CHECK_EQUAL(*iter,values[k]*values[k]);
		BOOST_CHECK_EQUAL(start[k],values[k]*values[k]);
		BOOST_CHECK_EQUAL(*(start+k),values[k]*values[k]);
		BOOST_CHECK(iter < end);
		++iter;
		++k;
	}
	BOOST_REQUIRE_EQUAL(k, 6);
	BOOST_REQUIRE_EQUAL(end-iter, 0);
	BOOST_REQUIRE_EQUAL(iter-start, 6);
}

BOOST_AUTO_TEST_CASE( Remora_Transform_Iterator_Compressed)
{
	double values[]={0.1,0.2,0.3,0.4,0.5,0.6};
	std::size_t indizes[]={3,8,11,12,15,16};
	typedef device_traits<cpu_tag>::sqr<double> functor;
	typedef iterators::compressed_storage_iterator<const double,const std::size_t> iterator;
	iterator compressed_iter(values,indizes,0);
	iterator compressed_end(values,indizes,6);
	iterators::transform_iterator<iterator,functor > iter(compressed_iter,functor());
	iterators::transform_iterator<iterator,functor > start = iter;
	iterators::transform_iterator<iterator,functor > end(compressed_end,functor());
	
	BOOST_REQUIRE_EQUAL(end-start, 6);
	BOOST_REQUIRE_EQUAL(start-iter, 0);
	BOOST_REQUIRE(start == iter);
	BOOST_REQUIRE(start != end);
	std::size_t k = 0;
	while(iter != end){
		BOOST_CHECK_EQUAL(iter.index(),indizes[k]);
		BOOST_CHECK_EQUAL(*iter,values[k]*values[k]);
		++iter;
		++k;
	}
	BOOST_REQUIRE_EQUAL(k, 6);
	BOOST_REQUIRE_EQUAL(end-iter, 0);
	BOOST_REQUIRE_EQUAL(iter-start, 6);
}

BOOST_AUTO_TEST_CASE( Remora_Binary_Transform_Iterator_Dense)
{
	double values1[]={0.1,0.2,0.3,0.4,0.5,0.6};
	double values2[]={0.3,0.5,0.7,0.9,1.1,1.3};
	

	typedef iterators::dense_storage_iterator<const double> iterator;
	iterator dense_iter1(values1,0);
	iterator dense_end1(values1,6);
	iterator dense_iter2(values2,0);
	iterator dense_end2(values2,6);
	typedef device_traits<cpu_tag>::add<double> functor;
	
	typedef iterators::binary_transform_iterator<iterator,iterator,functor > transform_iterator;
	
	transform_iterator iter(functor(),dense_iter1,dense_end1,dense_iter2,dense_end2);
	transform_iterator start = iter;
	transform_iterator end(functor(),dense_end1,dense_end1,dense_end2,dense_end2);
	
	BOOST_REQUIRE_EQUAL(end-start, 6);
	BOOST_REQUIRE_EQUAL(start-iter, 0);
	BOOST_REQUIRE(start == iter);
	BOOST_REQUIRE(start != end);
	BOOST_REQUIRE(start < end);
	BOOST_REQUIRE(end == start+6);
	std::size_t k = 0;
	while(iter != end){
		double value = values1[k]+values2[k];
		BOOST_CHECK_EQUAL(iter.index(),k);
		BOOST_CHECK_EQUAL(*iter,value);
		BOOST_CHECK_EQUAL(start[k],value);
		BOOST_CHECK_EQUAL(*(start+k),value);
		BOOST_CHECK(iter < end);
		++iter;
		++k;
	}
	BOOST_REQUIRE_EQUAL(k, 6);
	BOOST_REQUIRE_EQUAL(end-iter, 0);
	BOOST_REQUIRE_EQUAL(iter-start, 6);
}

BOOST_AUTO_TEST_CASE( Remora_Binary_Transform_Iterator_Compressed)
{
	double values1[]={0.1,0.2,0.3,0.4,0.5,0.6};
	double values2[]={0.3,0.5,0.7,0.9};
	double valuesResult[]={0.1,0.3,0.7,1.0,0.4,0.9,0.5,0.6};
	
	std::size_t indizes1[]={3,8,11,12,17,18};
	std::size_t indizes2[]={5,8,11,14};
	std::size_t indizesResult[]={3,5,8,11,12,14,17,18};
	
	typedef device_traits<cpu_tag>::add<double> functor;
	typedef iterators::compressed_storage_iterator<const double,const std::size_t> iterator;
	typedef iterators::binary_transform_iterator<iterator,iterator,functor > transform_iterator;
	
	//a+b
	{
		iterator iter1(values1,indizes1,0);
		iterator end1(values1,indizes1,6);
		iterator iter2(values2,indizes2,0);
		iterator end2(values2,indizes2,4);
		transform_iterator iter(functor(),iter1,end1,iter2,end2);
		transform_iterator start = iter;
		transform_iterator end(functor(),end1,end1,end2,end2);
		
		BOOST_REQUIRE(start == iter);
		BOOST_REQUIRE(start != end);
		std::size_t k = 0;
		while(iter != end){
			BOOST_CHECK_EQUAL(iter.index(),indizesResult[k]);
			BOOST_CHECK_EQUAL(*iter,valuesResult[k]);
			++iter;
			++k;
		}
		BOOST_REQUIRE_EQUAL(k, 8);
	}
	
	//b+a
	{
		iterator iter2(values1,indizes1,0);
		iterator end2(values1,indizes1,6);
		iterator iter1(values2,indizes2,0);
		iterator end1(values2,indizes2,4);
		transform_iterator iter(functor(),iter1,end1,iter2,end2);
		transform_iterator start = iter;
		transform_iterator end(functor(),end1,end1,end2,end2);
		
		BOOST_REQUIRE(start == iter);
		BOOST_REQUIRE(start != end);
		std::size_t k = 0;
		while(iter != end){
			BOOST_CHECK_EQUAL(iter.index(),indizesResult[k]);
			BOOST_CHECK_EQUAL(*iter,valuesResult[k]);
			++iter;
			++k;
		}
		BOOST_REQUIRE_EQUAL(k, 8);
	}
}
BOOST_AUTO_TEST_SUITE_END()
