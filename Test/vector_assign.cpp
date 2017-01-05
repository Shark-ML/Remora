#define BOOST_TEST_MODULE Remora_VectorAssign
#include <boost/test/unit_test.hpp>
#include <boost/test/floating_point_comparison.hpp>

#include <remora/assignment.hpp>
#include <remora/vector.hpp>
#include <remora/compressed_vector.hpp>
#include <iostream>

using namespace remora;

template<class V1, class V2>
void checkVectorEqual(V1 const& v1, V2 const& v2){
	BOOST_REQUIRE_EQUAL(v1.size(),v2.size());
	for(std::size_t i = 0; i != v2.size(); ++i){
		BOOST_CHECK_EQUAL(v1(i),v2(i));
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
	source_sparse(2) = 1;
	source_sparse(5) = 2;
	source_sparse(7) = 3;

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
	
	{
		vector<unsigned int> target_dense(10);
		for(std::size_t i = 0; i != 10; ++i){
			target_dense(i) = 3*i+2;
		}
		std::cout<<"testing dense-sparse"<<std::endl;
		kernels::assign(target_dense,source_sparse);
		checkVectorEqual(target_dense,source_sparse);
	}
	
	{
		compressed_vector<unsigned int> target_sparse(10);
		target_sparse(1) = 2;
		target_sparse(7) = 8;
		target_sparse(9) = 3;
		std::cout<<"testing sparse-dense"<<std::endl;
		kernels::assign(target_sparse,source_dense);
		BOOST_CHECK_EQUAL(target_sparse.nnz(), 10);
		checkVectorEqual(target_sparse,source_dense);
	}
	{
		compressed_vector<unsigned int> target_sparse(10);
		target_sparse(1) = 2;
		target_sparse(2) = 2;
		target_sparse(7) = 8;
		target_sparse(9) = 3;
		std::cout<<"testing sparse-sparse"<<std::endl;
		kernels::assign(target_sparse,source_sparse);
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
	source_sparse(2) = 1;
	source_sparse(5) = 2;
	source_sparse(7) = 3;

	//test all 4 combinations
	{
		vector<unsigned int> target_dense(10);
		vector<unsigned int> result_dense(10);
		for(std::size_t i = 0; i != 10; ++i){
			target_dense(i) = 3*i+2;
			result_dense(i) = source_dense(i)+target_dense(i);
		}
		std::cout<<"testing dense-dense"<<std::endl;
		kernels::assign<functors::scalar_binary_plus<unsigned int> >(target_dense,source_dense);
		checkVectorEqual(target_dense,result_dense);
	}
	
	{
		vector<unsigned int> target_dense(10);
		vector<unsigned int> result_dense(10);
		for(std::size_t i = 0; i != 10; ++i){
			target_dense(i) = 3*i+2;
			result_dense(i) = source_sparse(i)+target_dense(i);
		}
		std::cout<<"testing dense-sparse"<<std::endl;
		kernels::assign<functors::scalar_binary_plus<unsigned int> >(target_dense,source_sparse);
		checkVectorEqual(target_dense,result_dense);
	}
	
	{
		compressed_vector<unsigned int> target_sparse(10);
		vector<unsigned int> result_dense(10);
		target_sparse(1) = 2;
		target_sparse(2) = 2;
		target_sparse(7) = 8;
		target_sparse(9) = 3;
		for(std::size_t i = 0; i != 10; ++i){
			result_dense(i) = source_dense(i)+target_sparse(i);
		}
		std::cout<<"testing sparse-dense"<<std::endl;
		kernels::assign<functors::scalar_binary_plus<unsigned int> >(target_sparse,source_dense);
		BOOST_CHECK_EQUAL(target_sparse.nnz(), 10);
		checkVectorEqual(target_sparse,result_dense);
	}
	{
		compressed_vector<unsigned int> target_sparse(10);
		vector<unsigned int> result_dense(10);
		target_sparse(1) = 2;
		target_sparse(2) = 2;
		target_sparse(7) = 8;
		target_sparse(9) = 3;
		for(std::size_t i = 0; i != 10; ++i){
			result_dense(i) = source_sparse(i)+target_sparse(i);
		}
		std::cout<<"testing sparse-sparse"<<std::endl;
		kernels::assign<functors::scalar_binary_plus<unsigned int> >(target_sparse,source_sparse);
		BOOST_CHECK_EQUAL(target_sparse.nnz(), 5);
		checkVectorEqual(target_sparse,result_dense);
	}
}


BOOST_AUTO_TEST_SUITE_END()
