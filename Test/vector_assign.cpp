#define BOOST_TEST_MODULE Remora_VectorAssign
#include <boost/test/unit_test.hpp>
#include <boost/test/floating_point_comparison.hpp>

#include <remora/assignment.hpp>
#include <remora/dense.hpp>
#include <remora/sparse.hpp>
#include <iostream>

using namespace remora;

template<class V1, class V2>
void checkVectorEqual(V1 const& v1, V2 const& v2){
	BOOST_REQUIRE_EQUAL(v1.size(),v2.size());
	vector<typename V2::value_type> w1(v2.size());
	vector<typename V2::value_type> w2(v2.size());
	for(auto pos = v1.begin(); pos != v1.end(); ++pos){
		w1(pos.index()) = *pos;
	}
	for(auto pos = v2.begin(); pos != v2.end(); ++pos){
		w2(pos.index()) = *pos;
	}
	for(std::size_t i = 0; i != v2.size(); ++i){
		BOOST_CHECK_EQUAL(w1(i),w2(i));
	}
}

BOOST_AUTO_TEST_SUITE (Remora_vector_assign)

BOOST_AUTO_TEST_CASE( Remora_Vector_Assign ){
	std::cout<<"testing direct assignment"<<std::endl;
	vector<unsigned int> source_dense(10);
	compressed_vector<unsigned int> source_sparse(10);
	for(std::size_t i = 0; i != 10; ++i){
		source_dense(i) = 2*i+1;
	}
	auto pos = source_sparse.begin();
	pos = source_sparse.set_element(pos,2,1);
	pos = source_sparse.set_element(pos,5,2);
	pos = source_sparse.set_element(pos,7,3);

	//test all 4 combinations
	{
		vector<unsigned int> target_dense(10);
		for(std::size_t i = 0; i != 10; ++i){
			target_dense(i) = 3*i+2;
		}
		std::cout<<"testing dense-dense"<<std::endl;
		kernels::assign(target_dense,source_dense);
		checkVectorEqual(target_dense,source_dense);
	}
	std::cout<<"testing dense-sparse"<<std::endl;
	{
		vector<unsigned int> target_dense(10);
		for(std::size_t i = 0; i != 10; ++i){
			target_dense(i) = 3*i+2;
		}
		
		kernels::assign(target_dense,source_sparse);
		checkVectorEqual(target_dense,source_sparse);
	}
	{
		vector<unsigned int> target_dense(10);
		for(std::size_t i = 0; i != 10; ++i){
			target_dense(i) = 3*i+2;
		}
		kernels::assign(target_dense,compressed_vector<unsigned int>::const_closure_type(source_sparse));
		checkVectorEqual(target_dense,source_sparse);
	}
	
	{
		compressed_vector<unsigned int> target_sparse(10);
		auto pos = target_sparse.begin();
		pos = target_sparse.set_element(pos,1,2);
		pos = target_sparse.set_element(pos,7,8);
		pos = target_sparse.set_element(pos,9,3);
		std::cout<<"testing sparse-dense"<<std::endl;
		kernels::assign(target_sparse,source_dense);
		BOOST_CHECK_EQUAL(target_sparse.nnz(), 10);
		checkVectorEqual(target_sparse,source_dense);
	}
	std::cout<<"testing sparse-sparse"<<std::endl;
	{
		compressed_vector<unsigned int> target_sparse(10);
		auto pos = target_sparse.begin();
		pos = target_sparse.set_element(pos,1,2);
		pos = target_sparse.set_element(pos,2,4);
		pos = target_sparse.set_element(pos,7,8);
		pos = target_sparse.set_element(pos,9,3);
		
		kernels::assign(target_sparse,source_sparse);
		BOOST_CHECK_EQUAL(target_sparse.nnz(), 3);
		checkVectorEqual(target_sparse,source_sparse);
	}
	{
		compressed_vector<unsigned int> target_sparse(10);
		auto pos = target_sparse.begin();
		pos = target_sparse.set_element(pos,1,2);
		pos = target_sparse.set_element(pos,2,4);
		pos = target_sparse.set_element(pos,7,8);
		pos = target_sparse.set_element(pos,9,3);
		compressed_vector<unsigned int>::closure_type target(target_sparse);
		kernels::assign(target,source_sparse);
		BOOST_CHECK_EQUAL(target_sparse.nnz(), 3);
		checkVectorEqual(target_sparse,source_sparse);
	}
}

BOOST_AUTO_TEST_CASE( Remora_Vector_Assign_Functor ){
	std::cout<<"testing += assignment"<<std::endl;
	vector<unsigned int> source_dense(10);
	compressed_vector<unsigned int> source_sparse(10);
	for(std::size_t i = 0; i != 10; ++i){
		source_dense(i) = 2*i+1;
	}
	auto pos = source_sparse.begin();
	pos = source_sparse.set_element(pos,2,1);
	pos = source_sparse.set_element(pos,5,2);
	pos = source_sparse.set_element(pos,7,3);
	typedef device_traits<cpu_tag>::add<unsigned int> functor;

	//test all 4 combinations
	{
		vector<unsigned int> target_dense(10);
		vector<unsigned int> result_dense(10);
		for(std::size_t i = 0; i != 10; ++i){
			target_dense(i) = 3*i+2;
			result_dense(i) = source_dense(i)+target_dense(i);
		}
		std::cout<<"testing dense-dense"<<std::endl;
		kernels::assign(target_dense,source_dense,functor());
		checkVectorEqual(target_dense,result_dense);
	}
	
	{
		vector<unsigned int> target_dense(10);
		for(std::size_t i = 0; i != 10; ++i){
			target_dense(i) = 3*i+2;
		}
		vector<unsigned int> result_dense(target_dense);
		result_dense(2) +=1;
		result_dense(5) +=2;
		result_dense(7) +=3;
		std::cout<<"testing dense-sparse"<<std::endl;
		kernels::assign(target_dense,source_sparse,functor());
		checkVectorEqual(target_dense,result_dense);
	}
	
	{
		compressed_vector<unsigned int> target_sparse(10);
		auto pos = target_sparse.begin();
		pos = target_sparse.set_element(pos,1,2);
		pos = target_sparse.set_element(pos,2,4);
		pos = target_sparse.set_element(pos,7,8);
		pos = target_sparse.set_element(pos,9,3);
		vector<unsigned int> result_dense = source_dense;
		result_dense(1) += 2;
		result_dense(2) += 4;
		result_dense(7) += 8;
		result_dense(9) += 3;
		std::cout<<"testing sparse-dense"<<std::endl;
		kernels::assign(target_sparse,source_dense,functor());
		BOOST_CHECK_EQUAL(target_sparse.nnz(), 10);
		checkVectorEqual(target_sparse,result_dense);
	}
	{
		compressed_vector<unsigned int> target_sparse(10);
		
		auto pos = target_sparse.begin();
		pos = target_sparse.set_element(pos,1,2);
		pos = target_sparse.set_element(pos,2,4);
		pos = target_sparse.set_element(pos,7,8);
		pos = target_sparse.set_element(pos,9,3);
		
		vector<unsigned int> result_dense(10);
		result_dense(1) = 2;
		result_dense(2) = 5;
		result_dense(5) = 2;
		result_dense(7) = 3+8;
		result_dense(9) = 3;
		std::cout<<"testing sparse-sparse"<<std::endl;
		kernels::assign(target_sparse,source_sparse,functor());
		BOOST_CHECK_EQUAL(target_sparse.nnz(), 5);
		checkVectorEqual(target_sparse,result_dense);
	}
}


BOOST_AUTO_TEST_SUITE_END()
