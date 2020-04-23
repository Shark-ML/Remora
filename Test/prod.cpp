#define BOOST_TEST_MODULE Remora_prod
#include <boost/test/unit_test.hpp>
#include <boost/test/floating_point_comparison.hpp>

#include <remora/expressions.hpp>
#include <remora/dense.hpp>
// #include <remora/sparse.hpp>

#include <iostream>

using namespace remora;

template<class M, class V, class Result>
void checkMatrixVectorMultiply(M const& arg1, V const& arg2, Result const& result, double factor, double init = 0){
	BOOST_REQUIRE_EQUAL(arg1.shape()[0], result.shape()[0]);
	BOOST_REQUIRE_EQUAL(arg2.shape()[0], arg1.shape()[1]);
	
	remora::matrix<double> copy1(arg1);
	remora::vector<double> copy2(arg2);
	
	for(std::size_t i = 0; i != arg1.shape()[0]; ++i){
		double test_result = init;
		for(std::size_t k = 0; k != arg1.shape()[1]; ++k){
			test_result += factor * copy1(i,k)*copy2(k);
		}
		BOOST_CHECK_CLOSE(result(i), test_result,1.e-10);
	}
}

BOOST_AUTO_TEST_SUITE (Remora_prod)


BOOST_AUTO_TEST_CASE( Remora_prod_expressions){
	std::size_t rows = 80;
	std::size_t columns = 50;
	std::size_t middle = 33;
	matrix<double,row_major> arg1({rows,middle});
	for(std::size_t i = 0; i != rows; ++i){
		for(std::size_t j = 0; j != middle; ++j){
			arg1(i,j) = i*middle+0.2*j;
		}
	}
	matrix<double,row_major> arg2({middle,columns});
	for(std::size_t i = 0; i != middle; ++i){
		for(std::size_t j = 0; j != columns; ++j){
			arg2(i,j) = i*columns+1.5*j;
		}
	}
	
	std::cout<<"proxy tests"<<std::endl;
	std::cout<<"test transpose"<<std::endl;
	//test transpose
	{
		matrix<double, row_major> result = arg1 % arg2;
		matrix<double, row_major> test_result = trans(trans(arg2) % trans(arg1));
		
		BOOST_REQUIRE_EQUAL(test_result.shape()[0], rows);
		BOOST_REQUIRE_EQUAL(test_result.shape()[1], columns);
		for(std::size_t i = 0; i != result.shape()[0]; ++i){
			for(std::size_t j = 0; j != result.shape()[1]; ++j){
				BOOST_CHECK_CLOSE(result(i,j), test_result(i,j),1.e-10);
			}
		}
	}
	
	//merge is the default implementation, no need to test
	
	//test slice
	std::cout<<"test slice"<<std::endl;
	{
		vector<double> result1 = arg1 % slice(arg2,ax::same, 1);
		vector<double> result2 = slice(arg1, 1, ax::same) % arg2;
		vector<double> test_result1 = slice(arg1 % arg2, ax::same, 1);
		vector<double> test_result2 = slice(arg1 % arg2, 1, ax::same);
		//slice on transpose to test non-trivial permute parameter
		vector<double> test_result3 = slice(trans(arg1 % arg2), 1, ax::same);
		vector<double> test_result4 = slice(trans(arg1 % arg2), ax::same, 1);
		
		BOOST_REQUIRE_EQUAL(test_result1.shape()[0], rows);
		BOOST_REQUIRE_EQUAL(test_result2.shape()[0], columns);
		BOOST_REQUIRE_EQUAL(test_result3.shape()[0], rows);
		BOOST_REQUIRE_EQUAL(test_result4.shape()[0], columns);
		for(std::size_t i = 0; i != result1.shape()[0]; ++i){
			BOOST_CHECK_CLOSE(result1(i), test_result1(i),1.e-10);
			BOOST_CHECK_CLOSE(result1(i), test_result3(i),1.e-10);
		}
		for(std::size_t i = 0; i != result2.shape()[0]; ++i){
			BOOST_CHECK_CLOSE(result2(i), test_result2(i),1.e-10);
			BOOST_CHECK_CLOSE(result2(i), test_result4(i),1.e-10);
		}
	}
	
	//test split
	std::cout<<"test split"<<std::endl;
	{
		matrix<double> result_mat = arg1 % arg2;
		tensorN<double, 3> test_result1 = reshape(arg1 % arg2, ax::same, ax::split<2>(10,5));
		tensorN<double, 3> test_result2 = reshape(arg1 % arg2, ax::split<2>(8,10), ax::same);
		
		tensorN<double, 3> test_result3 = reshape(trans(arg1 % arg2), ax::split<2>(10,5), ax::same);
		tensorN<double, 3> test_result4 = reshape(trans(arg1 % arg2), ax::same, ax::split<2>(8,10));
		
		BOOST_REQUIRE_EQUAL(test_result1.shape()[0], rows);
		BOOST_REQUIRE_EQUAL(test_result1.shape()[1], 10);
		BOOST_REQUIRE_EQUAL(test_result1.shape()[2], 5);
		
		BOOST_REQUIRE_EQUAL(test_result2.shape()[0], 8);
		BOOST_REQUIRE_EQUAL(test_result2.shape()[1], 10);
		BOOST_REQUIRE_EQUAL(test_result2.shape()[2], columns);
		
		BOOST_REQUIRE_EQUAL(test_result3.shape()[2], rows);
		BOOST_REQUIRE_EQUAL(test_result3.shape()[0], 10);
		BOOST_REQUIRE_EQUAL(test_result3.shape()[1], 5);
		
		BOOST_REQUIRE_EQUAL(test_result4.shape()[1], 8);
		BOOST_REQUIRE_EQUAL(test_result4.shape()[2], 10);
		BOOST_REQUIRE_EQUAL(test_result4.shape()[0], columns);
		
		for(std::size_t i = 0; i != result_mat.shape()[0]; ++i){
			for(std::size_t j = 0; j != 10; ++j){
				for(std::size_t k = 0; k != 5; ++k){
					BOOST_CHECK_CLOSE(result_mat(i, j*5 + k), test_result1(i,j,k),1.e-10);
					BOOST_CHECK_CLOSE(result_mat(i, j*5 + k), test_result3(j,k,i),1.e-10);
				}
			}				
		}
		for(std::size_t j = 0; j != 8; ++j){
			for(std::size_t k = 0; k != 10; ++k){
				for(std::size_t i = 0; i != result_mat.shape()[1]; ++i){
					BOOST_CHECK_CLOSE(result_mat(j*10 + k, i), test_result2(j,k,i),1.e-10);
					BOOST_CHECK_CLOSE(result_mat(j*10 + k, i), test_result4(i,j,k),1.e-10);
				}
			}				
		}
	}
	
	//test subrange
	std::cout<<"test subrange"<<std::endl;
	{
		matrix<double> result_mat = slice(arg1,ax::range(10,25)) % slice(arg2,ax::same, ax::range(20,40));
		matrix<double> test_result1 = slice(arg1 % arg2, ax::range(10,25), ax::range(20,40));
		matrix<double> test_result2 = slice(trans(arg1 % arg2), ax::range(20,40), ax::range(10,25));

		BOOST_REQUIRE_EQUAL(test_result1.shape()[0], 15);
		BOOST_REQUIRE_EQUAL(test_result1.shape()[1], 20);
		BOOST_REQUIRE_EQUAL(test_result2.shape()[0], 20);
		BOOST_REQUIRE_EQUAL(test_result2.shape()[1], 15);
		for(std::size_t i = 0; i != result_mat.shape()[0]; ++i){
			for(std::size_t j = 0; j != result_mat.shape()[1]; ++j){
				BOOST_CHECK_CLOSE(result_mat(i, j), test_result1(i,j),1.e-10);
				BOOST_CHECK_CLOSE(result_mat(i, j), test_result2(j,i),1.e-10);
			}				
		}
	}
	
	//test scalar multiply
	std::cout<<"test scalar multiply"<<std::endl;
	{
		matrix<double, row_major> result = arg1 % arg2; result *=4;
		matrix<double, row_major> test_result1 = (4*arg1) %arg2;
		matrix<double, row_major> test_result2 = arg1 % (4*arg2);
		matrix<double, row_major> test_result3 = (2*arg1) % (2 * arg2);
		matrix<double, row_major> test_result4 = 4 * arg1 % arg2;
		
		BOOST_REQUIRE_EQUAL(test_result1.shape()[0], rows);
		BOOST_REQUIRE_EQUAL(test_result1.shape()[1], columns);
		BOOST_REQUIRE_EQUAL(test_result2.shape()[0], rows);
		BOOST_REQUIRE_EQUAL(test_result2.shape()[1], columns);
		BOOST_REQUIRE_EQUAL(test_result3.shape()[0], rows);
		BOOST_REQUIRE_EQUAL(test_result3.shape()[1], columns);
		BOOST_REQUIRE_EQUAL(test_result4.shape()[0], rows);
		BOOST_REQUIRE_EQUAL(test_result4.shape()[1], columns);
		
		for(std::size_t i = 0; i != result.shape()[0]; ++i){
			for(std::size_t j = 0; j != result.shape()[1]; ++j){
				BOOST_CHECK_CLOSE(result(i,j), test_result1(i,j),1.e-10);
				BOOST_CHECK_CLOSE(result(i,j), test_result2(i,j),1.e-10);
				BOOST_CHECK_CLOSE(result(i,j), test_result3(i,j),1.e-10);
				BOOST_CHECK_CLOSE(result(i,j), test_result4(i,j),1.e-10);
			}
		}
	}
	
	
	
}

BOOST_AUTO_TEST_CASE( Remora_prod_matrix_vector_dense ){
	std::size_t rows = 50;
	std::size_t columns = 80;
	//initialize the arguments in both row and column major as well as transposed
	matrix<double,row_major> arg1rm({rows,columns});
	matrix<double,column_major> arg1cm({rows,columns});
	matrix<double,row_major> arg1rmt({columns,rows});
	matrix<double,column_major> arg1cmt({columns,rows});
	for(std::size_t i = 0; i != rows; ++i){
		for(std::size_t j = 0; j != columns; ++j){
			arg1rm(i,j) = arg1cm(i,j) = i*columns+0.2*j;
			arg1rmt(j,i) = arg1cmt(j,i) = i*columns+0.2*j;
		}
	}
	vector<double> arg2(columns);
	for(std::size_t j = 0; j != columns; ++j){
		arg2(j)  = 1.5*j+2;
	}

	std::cout<<"\nchecking dense matrix vector plusassign multiply"<<std::endl;
	//test first expressions of the form A += alpha*B*C 
	{
		std::cout<<"row major Ax"<<std::endl;
		vector<double> result(rows,1.5);
		noalias(result) += -2 * arg1rm % arg2;
		checkMatrixVectorMultiply(arg1rm,arg2,result,-2.0,1.5);
	}
	{
		std::cout<<"column major Ax"<<std::endl;
		vector<double> result(rows,1.5);
		noalias(result) += -2* arg1cm % arg2;
		checkMatrixVectorMultiply(arg1cm,arg2,result,-2.0,1.5);
	}
	{
		std::cout<<"row major xA"<<std::endl;
		vector<double> result(rows,1.5);
		noalias(result) += -2 * arg2 % arg1rmt;
		checkMatrixVectorMultiply(arg1rm,arg2,result,-2.0,1.5);
	}
	{
		std::cout<<"column major xA"<<std::endl;
		vector<double> result(rows,1.5);
		noalias(result) += -2 * arg2 % arg1cmt;
		checkMatrixVectorMultiply(arg1cm,arg2,result,-2.0,1.5);
	}
	std::cout<<"\nchecking dense matrix vector assign multiply"<<std::endl;
	//test expressions of the form A=B*C
	{
		std::cout<<"row major Ax"<<std::endl;
		vector<double> result(rows,1.5);
		noalias(result) = -2 * arg1rm % arg2;
		checkMatrixVectorMultiply(arg1rm,arg2,result,-2.0,0);
	}
	{
		std::cout<<"column major Ax"<<std::endl;
		vector<double> result(rows,1.5);
		noalias(result) = -2 * arg1cm % arg2;
		checkMatrixVectorMultiply(arg1cm,arg2,result,-2.0,0);
	}
	{
		std::cout<<"row major xA"<<std::endl;
		vector<double> result(rows,1.5);
		noalias(result) = -2 * arg2 % arg1rmt;
		checkMatrixVectorMultiply(arg1rm,arg2,result,-2.0,0);
	}
	{
		std::cout<<"column major xA"<<std::endl;
		vector<double> result(rows,1.5);
		noalias(result) = -2 * arg2 % arg1cmt;
		checkMatrixVectorMultiply(arg1cm,arg2,result,-2.0,0);
	}
}
/*
BOOST_AUTO_TEST_CASE( Remora_prod_matrix_vector_dense_sparse ){
	std::size_t rows = 50;
	std::size_t columns = 80;
	//initialize the arguments in both row and column major as well as transposed
	matrix<double,row_major> arg1rm({rows,columns});
	matrix<double,column_major> arg1cm({rows,columns});
	matrix<double,row_major> arg1rmt({columns,rows});
	matrix<double,column_major> arg1cmt({columns,rows});
	for(std::size_t i = 0; i != rows; ++i){
		for(std::size_t j = 0; j != columns; ++j){
			arg1rm(i,j) = arg1cm(i,j) = i*columns+0.2*j;
			arg1rmt(j,i) = arg1cmt(j,i) = i*columns+0.2*j;
		}
	}
	compressed_vector<double> arg2(columns);
	auto pos = arg2.begin();
	for(std::size_t j = 1; j < columns; j+=3){
		pos = arg2.set_element(pos, j, 1.5*j+2);
	}

	std::cout<<"\nchecking dense-sparse matrix vector plusassign multiply"<<std::endl;
	//test first expressions of the form A += alpha*B*C 
	{
		std::cout<<"row major Ax"<<std::endl;
		vector<double> result(rows,1.5);
		noalias(result) += -2*(arg1rm % arg2);
		checkMatrixVectorMultiply(arg1rm,arg2,result,-2.0,1.5);
	}
	{
		std::cout<<"column major Ax"<<std::endl;
		vector<double> result(rows,1.5);
		noalias(result) += -2*(arg1cm % arg2);
		checkMatrixVectorMultiply(arg1cm,arg2,result,-2.0,1.5);
	}
	{
		std::cout<<"row major xA"<<std::endl;
		vector<double> result(rows,1.5);
		noalias(result) += -2*(arg2 % arg1rmt);
		checkMatrixVectorMultiply(arg1rm,arg2,result,-2.0,1.5);
	}
	{
		std::cout<<"column major xA"<<std::endl;
		vector<double> result(rows,1.5);
		noalias(result) += -2*(arg2 % arg1cmt);
		checkMatrixVectorMultiply(arg1cm,arg2,result,-2.0,1.5);
	}
	std::cout<<"\nchecking dense-sparse matrix vector assign multiply"<<std::endl;
	//test expressions of the form A=B*C
	{
		std::cout<<"row major Ax"<<std::endl;
		vector<double> result(rows,1.5);
		noalias(result) = -2*(arg1rm % arg2);
		checkMatrixVectorMultiply(arg1rm,arg2,result,-2.0,0);
	}
	{
		std::cout<<"column major Ax"<<std::endl;
		vector<double> result(rows,1.5);
		noalias(result) = -2*(arg1cm % arg2);
		checkMatrixVectorMultiply(arg1cm,arg2,result,-2.0,0);
	}
	{
		std::cout<<"row major xA"<<std::endl;
		vector<double> result(rows,1.5);
		noalias(result) = -2*(arg2 % arg1rmt);
		checkMatrixVectorMultiply(arg1rm,arg2,result,-2.0,0);
	}
	{
		std::cout<<"column major xA"<<std::endl;
		vector<double> result(rows,1.5);
		noalias(result) = -2*(arg2 % arg1cmt);
		checkMatrixVectorMultiply(arg1cm,arg2,result,-2.0,0);
	}
}

BOOST_AUTO_TEST_CASE( Remora_prod_matrix_vector_sparse_dense ){
	std::size_t rows = 50;
	std::size_t columns = 80;
	//initialize the arguments in both row and column major as well as transposed
	compressed_matrix<double> arg1rm({rows,columns});	
	for(std::size_t i = 0; i != 10; ++i){
		auto pos = arg1rm.major_begin(i);
		for(std::size_t j = i%3; j < 20; j+=(i+1)){
			pos = arg1rm.set_element(pos,j,2.0*(20*i+1)+1.0);
		}
	}
	compressed_matrix<double>  arg1rmt = trans(arg1rm);
	compressed_matrix<double, unsigned int, column_major>  arg1cm = arg1rm;
	compressed_matrix<double, unsigned int, column_major>  arg1cmt = trans(arg1rm);
	

	vector<double> arg2(columns);
	for(std::size_t j = 0; j != columns; ++j){
		arg2(j)  = 1.5*j+2;
	}

	std::cout<<"\nchecking sparse-dense matrix vector plusassign multiply"<<std::endl;
	//test first expressions of the form A += alpha*B*C 
	{
		std::cout<<"row major Ax"<<std::endl;
		vector<double> result(rows,1.5);
		noalias(result) += -2*(arg1rm % arg2);
		checkMatrixVectorMultiply(arg1rm,arg2,result,-2.0,1.5);
	}
	{
		std::cout<<"column major Ax"<<std::endl;
		vector<double> result(rows,1.5);
		noalias(result) += -2*(arg1cm % arg2);
		checkMatrixVectorMultiply(arg1cm,arg2,result,-2.0,1.5);
	}
	{
		std::cout<<"row major xA"<<std::endl;
		vector<double> result(rows,1.5);
		noalias(result) += -2*(arg2 % arg1rmt);
		checkMatrixVectorMultiply(arg1rm,arg2,result,-2.0,1.5);
	}
	{
		std::cout<<"column major xA"<<std::endl;
		vector<double> result(rows,1.5);
		noalias(result) += -2*(arg2 % arg1cmt);
		checkMatrixVectorMultiply(arg1cm,arg2,result,-2.0,1.5);
	}
	std::cout<<"\nchecking sparse-dense matrix vector assign multiply"<<std::endl;
	//test expressions of the form A=B*C
	{
		std::cout<<"row major Ax"<<std::endl;
		vector<double> result(rows,1.5);
		noalias(result) = -2*(arg1rm % arg2);
		checkMatrixVectorMultiply(arg1rm,arg2,result,-2.0,0);
	}
	{
		std::cout<<"column major Ax"<<std::endl;
		vector<double> result(rows,1.5);
		noalias(result) = -2*(arg1cm % arg2);
		checkMatrixVectorMultiply(arg1cm,arg2,result,-2.0,0);
	}
	{
		std::cout<<"row major xA"<<std::endl;
		vector<double> result(rows,1.5);
		noalias(result) = -2*(arg2 % arg1rmt);
		checkMatrixVectorMultiply(arg1rm,arg2,result,-2.0,0);
	}
	{
		std::cout<<"column major xA"<<std::endl;
		vector<double> result(rows,1.5);
		noalias(result) = -2*(arg2 % arg1cmt);
		checkMatrixVectorMultiply(arg1cm,arg2,result,-2.0,0);
	}
}

BOOST_AUTO_TEST_CASE( Remora_prod_matrix_vector_sparse_sparse ){
	std::size_t rows = 50;
	std::size_t columns = 80;
	//initialize the arguments in both row and column major as well as transposed
	compressed_matrix<double> arg1rm({rows,columns});	
	for(std::size_t i = 0; i != 10; ++i){
		auto pos = arg1rm.major_begin(i);
		for(std::size_t j = i%3; j < 20; j+=(i+1)){
			pos = arg1rm.set_element(pos,j,2.0*(20*i+1)+1.0);
		}
	}
	compressed_matrix<double, std::size_t, column_major> arg1cm = arg1rm;	
	compressed_matrix<double> arg1rmt = trans(arg1rm);	
	compressed_matrix<double, std::size_t, column_major> arg1cmt = trans(arg1rm);	

	compressed_vector<double> arg2(columns);
	auto pos = arg2.begin();
	for(std::size_t j = 1; j < columns; j+=3){
		pos = arg2.set_element(pos, j, 1.5*j+2);
	}

	std::cout<<"\nchecking sparse-sparse matrix vector plusassign multiply"<<std::endl;
	//test first expressions of the form A += alpha*B*C 
	{
		std::cout<<"row major Ax"<<std::endl;
		vector<double> result(rows,1.5);
		noalias(result) += -2*(arg1rm % arg2);
		checkMatrixVectorMultiply(arg1rm,arg2,result,-2.0,1.5);
	}
	{
		std::cout<<"column major Ax"<<std::endl;
		vector<double> result(rows,1.5);
		noalias(result) += -2*(arg1cm % arg2);
		checkMatrixVectorMultiply(arg1cm,arg2,result,-2.0,1.5);
	}
	{
		std::cout<<"row major xA"<<std::endl;
		vector<double> result(rows,1.5);
		noalias(result) += -2*(arg2 % arg1rmt);
		checkMatrixVectorMultiply(arg1rm,arg2,result,-2.0,1.5);
	}
	{
		std::cout<<"column major xA"<<std::endl;
		vector<double> result(rows,1.5);
		noalias(result) += -2*(arg2 % arg1cmt);
		checkMatrixVectorMultiply(arg1cm,arg2,result,-2.0,1.5);
	}
	std::cout<<"\nchecking sparse-sparse matrix vector assign multiply"<<std::endl;
	//test expressions of the form A=B*C
	{
		std::cout<<"row major Ax"<<std::endl;
		vector<double> result(rows,1.5);
		noalias(result) = -2*(arg1rm % arg2);
		checkMatrixVectorMultiply(arg1rm,arg2,result,-2.0,0);
	}
	{
		std::cout<<"column major Ax"<<std::endl;
		vector<double> result(rows,1.5);
		noalias(result) = -2*(arg1cm % arg2);
		checkMatrixVectorMultiply(arg1cm,arg2,result,-2.0,0);
	}
	{
		std::cout<<"row major xA"<<std::endl;
		vector<double> result(rows,1.5);
		noalias(result) = -2*(arg2 % arg1rmt);
		checkMatrixVectorMultiply(arg1rm,arg2,result,-2.0,0);
	}
	{
		std::cout<<"column major xA"<<std::endl;
		vector<double> result(rows,1.5);
		noalias(result) = -2*(arg2 % arg1cmt);
		checkMatrixVectorMultiply(arg1cm,arg2,result,-2.0,0);
	}
}*/

//we test using the textbook definition.
template<class Arg1, class Arg2, class Result>
void checkMatrixMatrixMultiply(Arg1 const& arg1, Arg2 const& arg2, Result const& result, double factor, double init = 0){
	BOOST_REQUIRE_EQUAL(arg1.shape()[0], result.shape()[0]);
	BOOST_REQUIRE_EQUAL(arg2.shape()[1], result.shape()[1]);
	
	remora::matrix<double> copy1(arg1);
	remora::matrix<double> copy2(arg2);
	
	for(std::size_t i = 0; i != arg1.shape()[0]; ++i){
		for(std::size_t j = 0; j != arg2.shape()[1]; ++j){
			double test_result = init;
			for(std::size_t k = 0; k != arg1.shape()[1]; ++k){
				test_result += factor * copy1(i,k)*copy2(k,j);
			}
			BOOST_CHECK_CLOSE(result(i,j), test_result,1.e-10);
		}
	}
}
BOOST_AUTO_TEST_CASE( Remora_prod_matrix_matrix_dense_dense ){
	std::size_t rows = 80;
	std::size_t columns = 50;
	std::size_t middle = 33;
	//initialize the arguments in both row and column major
	matrix<double,row_major> arg1rm({rows,middle});
	matrix<double,column_major> arg1cm({rows,middle});
	for(std::size_t i = 0; i != rows; ++i){
		for(std::size_t j = 0; j != middle; ++j){
			arg1rm(i,j) = arg1cm(i,j) = i*middle+0.2*j;
		}
	}
	matrix<double,row_major> arg2rm({middle,columns});
	matrix<double,column_major> arg2cm({middle,columns});
	for(std::size_t i = 0; i != middle; ++i){
		for(std::size_t j = 0; j != columns; ++j){
			arg2rm(i,j) = arg2cm(i,j) = i*columns+1.5*j;
		}
	}
	std::cout<<"\nchecking dense-dense matrix matrix plusassign multiply"<<std::endl;
	//test first expressions of the form A+=B*C
	{
		std::cout<<"rrr"<<std::endl;
		matrix<double,row_major> resultrm({rows,columns},1.5);
		noalias(resultrm) += -2.0 * (arg1rm % arg2rm);
		checkMatrixMatrixMultiply(arg1rm,arg2rm,resultrm,-2.0,1.5);
	}
	{
		std::cout<<"rrc"<<std::endl;
		matrix<double,column_major> resultcm({rows,columns},1.5);
		noalias(resultcm) += -2.0 * (arg1rm % arg2rm);
		checkMatrixMatrixMultiply(arg1rm,arg2rm,resultcm,-2.0,1.5);
	}
	{
		std::cout<<"rcr"<<std::endl;
		matrix<double,row_major> resultrm({rows,columns},1.5);
		noalias(resultrm) += -2.0 * (arg1rm % arg2cm);
		checkMatrixMatrixMultiply(arg1rm,arg2cm,resultrm,-2.0,1.5);
	}
	{
		std::cout<<"rcc"<<std::endl;
		matrix<double,column_major> resultcm({rows,columns},1.5);
		noalias(resultcm) += -2.0 * (arg1rm % arg2cm);
		checkMatrixMatrixMultiply(arg1rm,arg2cm,resultcm,-2.0,1.5);
	}
	
	{
		std::cout<<"crr"<<std::endl;
		matrix<double,row_major> resultrm({rows,columns},1.5);
		noalias(resultrm) += -2.0 * (arg1cm % arg2rm);
		checkMatrixMatrixMultiply(arg1cm,arg2rm,resultrm,-2.0,1.5);
	}
	{
		std::cout<<"crc"<<std::endl;
		matrix<double,column_major> resultcm({rows,columns},1.5);
		noalias(resultcm) += -2.0 * (arg1cm % arg2rm);
		checkMatrixMatrixMultiply(arg1cm,arg2rm,resultcm,-2.0,1.5);
	}
	{
		std::cout<<"ccr"<<std::endl;
		matrix<double,row_major> resultrm({rows,columns},1.5);
		noalias(resultrm) += -2.0 * (arg1cm % arg2cm);
		checkMatrixMatrixMultiply(arg1cm,arg2cm,resultrm,-2.0,1.5);
	}
	{
		std::cout<<"ccc"<<std::endl;
		matrix<double,column_major> resultcm({rows,columns},1.5);
		noalias(resultcm) += -2.0 * (arg1cm % arg2cm);
		checkMatrixMatrixMultiply(arg1cm,arg2cm,resultcm,-2.0,1.5);
	}
	
	std::cout<<"\nchecking dense-dense matrix matrix assign multiply"<<std::endl;
	//test expressions of the form A=B*C
	{
		std::cout<<"rrr"<<std::endl;
		matrix<double,row_major> resultrm({rows,columns},1.5);
		noalias(resultrm) = -2.0 * (arg1rm % arg2rm);
		checkMatrixMatrixMultiply(arg1rm,arg2rm,resultrm,-2.0,0);
	}
	{
		std::cout<<"rrc"<<std::endl;
		matrix<double,column_major> resultcm({rows,columns},1.5);
		noalias(resultcm) = -2.0 * (arg1rm % arg2rm);
		checkMatrixMatrixMultiply(arg1rm,arg2rm,resultcm,-2.0,0);
	}
	{
		std::cout<<"rcr"<<std::endl;
		matrix<double,row_major> resultrm({rows,columns},1.5);
		noalias(resultrm) = -2.0 * (arg1rm % arg2cm);
		checkMatrixMatrixMultiply(arg1rm,arg2cm,resultrm,-2.0,0);
	}
	{
		std::cout<<"rcc"<<std::endl;
		matrix<double,column_major> resultcm({rows,columns},1.5);
		noalias(resultcm) = -2.0 * (arg1rm % arg2cm);
		checkMatrixMatrixMultiply(arg1rm,arg2cm,resultcm,-2.0,0);
	}
	
	{
		std::cout<<"crr"<<std::endl;
		matrix<double,row_major> resultrm({rows,columns},1.5);
		noalias(resultrm) = -2.0 * (arg1cm % arg2rm);
		checkMatrixMatrixMultiply(arg1cm,arg2rm,resultrm,-2.0,0);
	}
	{
		std::cout<<"crc"<<std::endl;
		matrix<double,column_major> resultcm({rows,columns},1.5);
		noalias(resultcm) = -2.0 * (arg1cm % arg2rm);
		checkMatrixMatrixMultiply(arg1cm,arg2rm,resultcm,-2.0,0);
	}
	{
		std::cout<<"ccr"<<std::endl;
		matrix<double,row_major> resultrm({rows,columns},1.5);
		noalias(resultrm) = -2.0 * (arg1cm % arg2cm);
		checkMatrixMatrixMultiply(arg1cm,arg2cm,resultrm,-2.0,0);
	}
	{
		std::cout<<"ccc"<<std::endl;
		matrix<double,column_major> resultcm({rows,columns},1.5);
		noalias(resultcm) = -2.0 * (arg1cm % arg2cm);
		checkMatrixMatrixMultiply(arg1cm,arg2cm,resultcm,-2.0,0);
	}
}

/*
//second argument sparse
BOOST_AUTO_TEST_CASE( Remora_prod_matrix_matrix_dense_sparse ){
	std::size_t rows = 50;
	std::size_t columns = 80;
	std::size_t middle = 33;
	//initialize the arguments in both row and column major
	matrix<double,row_major> arg1rm(rows,middle);
	matrix<double,column_major> arg1cm(rows,middle);
	for(std::size_t i = 0; i != rows; ++i){
		for(std::size_t j = 0; j != middle; ++j){
			arg1rm(i,j) = arg1cm(i,j) = i*middle+0.2*j;
		}
	}
	
	compressed_matrix<double> arg2rm(middle,columns);
	for(std::size_t i = 0; i != middle; ++i){
		auto pos = arg2rm.major_begin(i);
		for(std::size_t j = i%3; j < columns; j+=(i+1)){
			pos = arg2rm.set_element(pos,j,2*(20*i+1)+1.0);
		}
	}
	compressed_matrix<double,std::size_t, column_major>  arg2cm = arg2rm;
	std::cout<<"\nchecking dense-sparse matrix matrix plusassign multiply"<<std::endl;
	//test first expressions of the form A+=B*C
	{
		std::cout<<"rrr"<<std::endl;
		matrix<double,row_major> resultrm({rows,columns},1.5);
		noalias(resultrm) += -2.0 * (arg1rm % arg2rm);
		checkMatrixMatrixMultiply(arg1rm,arg2rm,resultrm,-2.0,1.5);
	}
	{
		std::cout<<"rrc"<<std::endl;
		matrix<double,column_major> resultcm({rows,columns},1.5);
		noalias(resultcm) += -2.0 * (arg1rm % arg2rm);
		checkMatrixMatrixMultiply(arg1rm,arg2rm,resultcm,-2.0,1.5);
	}
	{
		std::cout<<"rcr"<<std::endl;
		matrix<double,row_major> resultrm({rows,columns},1.5);
		noalias(resultrm) += -2.0 * (arg1rm % arg2cm);
		checkMatrixMatrixMultiply(arg1rm,arg2cm,resultrm,-2.0,1.5);
	}
	{
		std::cout<<"rcc"<<std::endl;
		matrix<double,column_major> resultcm({rows,columns},1.5);
		noalias(resultcm) += -2.0 * (arg1rm % arg2cm);
		checkMatrixMatrixMultiply(arg1rm,arg2cm,resultcm,-2.0,1.5);
	}
	
	{
		std::cout<<"crr"<<std::endl;
		matrix<double,row_major> resultrm({rows,columns},1.5);
		noalias(resultrm) += -2.0 * (arg1cm % arg2rm);
		checkMatrixMatrixMultiply(arg1cm,arg2rm,resultrm,-2.0,1.5);
	}
	{
		std::cout<<"crc"<<std::endl;
		matrix<double,column_major> resultcm({rows,columns},1.5);
		noalias(resultcm) += -2.0 * (arg1cm % arg2rm);
		checkMatrixMatrixMultiply(arg1cm,arg2rm,resultcm,-2.0,1.5);
	}
	{
		std::cout<<"ccr"<<std::endl;
		matrix<double,row_major> resultrm({rows,columns},1.5);
		noalias(resultrm) += -2.0 * (arg1cm % arg2cm);
		checkMatrixMatrixMultiply(arg1cm,arg2cm,resultrm,-2.0,1.5);
	}
	{
		std::cout<<"ccc"<<std::endl;
		matrix<double,column_major> resultcm({rows,columns},1.5);
		noalias(resultcm) += -2.0 * (arg1cm % arg2cm);
		checkMatrixMatrixMultiply(arg1cm,arg2cm,resultcm,-2.0,1.5);
	}
	
	std::cout<<"\nchecking dense-sparse matrix matrix assign multiply"<<std::endl;
	//testexpressions of the form A=B*C
	{
		std::cout<<"rrr"<<std::endl;
		matrix<double,row_major> resultrm({rows,columns},1.5);
		noalias(resultrm) = -2.0 * (arg1rm % arg2rm);
		checkMatrixMatrixMultiply(arg1rm,arg2rm,resultrm,-2.0,0);
	}
	{
		std::cout<<"rrc"<<std::endl;
		matrix<double,column_major> resultcm({rows,columns},1.5);
		noalias(resultcm) = -2.0 * (arg1rm % arg2rm);
		checkMatrixMatrixMultiply(arg1rm,arg2rm,resultcm,-2.0,0);
	}
	{
		std::cout<<"rcr"<<std::endl;
		matrix<double,row_major> resultrm({rows,columns},1.5);
		noalias(resultrm) = -2.0 * (arg1rm % arg2cm);
		checkMatrixMatrixMultiply(arg1rm,arg2cm,resultrm,-2.0,0);
	}
	{
		std::cout<<"rcc"<<std::endl;
		matrix<double,column_major> resultcm({rows,columns},1.5);
		noalias(resultcm) = -2.0 * (arg1rm % arg2cm);
		checkMatrixMatrixMultiply(arg1rm,arg2cm,resultcm,-2.0,0);
	}
	
	{
		std::cout<<"crr"<<std::endl;
		matrix<double,row_major> resultrm({rows,columns},1.5);
		noalias(resultrm) = -2.0 * (arg1cm % arg2rm);
		checkMatrixMatrixMultiply(arg1cm,arg2rm,resultrm,-2.0,0);
	}
	{
		std::cout<<"crc"<<std::endl;
		matrix<double,column_major> resultcm({rows,columns},1.5);
		noalias(resultcm) = -2.0 * (arg1cm % arg2rm);
		checkMatrixMatrixMultiply(arg1cm,arg2rm,resultcm,-2.0,0);
	}
	{
		std::cout<<"ccr"<<std::endl;
		matrix<double,row_major> resultrm({rows,columns},1.5);
		noalias(resultrm) = -2.0 * (arg1cm % arg2cm);
		checkMatrixMatrixMultiply(arg1cm,arg2cm,resultrm,-2.0,0);
	}
	{
		std::cout<<"ccc"<<std::endl;
		matrix<double,column_major> resultcm({rows,columns},1.5);
		noalias(resultcm) = -2.0 * (arg1cm % arg2cm);
		checkMatrixMatrixMultiply(arg1cm,arg2cm,resultcm,-2.0,0);
	}
}

//first argument sparse
BOOST_AUTO_TEST_CASE( Remora_prod_matrix_matrix_sparse_dense ){
	std::size_t rows = 50;
	std::size_t columns = 80;
	std::size_t middle = 33;
	//initialize the arguments in both row and column major
	compressed_matrix<double> arg1rm(rows,middle);
	for(std::size_t i = 0; i != rows; ++i){
		auto pos = arg1rm.major_begin(i);
		for(std::size_t j = i%3; j < middle; j+=(i+1)){
			pos = arg1rm.set_element(pos,j,2*(20*i+1)+1.0);
		}
	}
	compressed_matrix<double,std::size_t, column_major>  arg1cm = arg1rm;
	matrix<double,row_major> arg2rm(middle,columns);
	matrix<double,column_major> arg2cm(middle,columns);
	for(std::size_t i = 0; i != middle; ++i){
		for(std::size_t j = 0; j != columns; ++j){
			arg2rm(i,j) = arg2cm(i,j) = i*columns+1.5*j;
		}
	}
	
	std::cout<<"\nchecking sparse-dense matrix matrix plusassign multiply"<<std::endl;
	//test first expressions of the form A+=B*C
	{
		std::cout<<"rrr"<<std::endl;
		matrix<double,row_major> resultrm({rows,columns},1.5);
		noalias(resultrm) += -2.0 * (arg1rm % arg2rm);
		checkMatrixMatrixMultiply(arg1rm,arg2rm,resultrm,-2.0,1.5);
	}
	{
		std::cout<<"rrc"<<std::endl;
		matrix<double,column_major> resultcm({rows,columns},1.5);
		noalias(resultcm) += -2.0 * (arg1rm % arg2rm);
		checkMatrixMatrixMultiply(arg1rm,arg2rm,resultcm,-2.0,1.5);
	}
	{
		std::cout<<"rcr"<<std::endl;
		matrix<double,row_major> resultrm({rows,columns},1.5);
		noalias(resultrm) += -2.0 * (arg1rm % arg2cm);
		checkMatrixMatrixMultiply(arg1rm,arg2cm,resultrm,-2.0,1.5);
	}
	{
		std::cout<<"rcc"<<std::endl;
		matrix<double,column_major> resultcm({rows,columns},1.5);
		noalias(resultcm) += -2.0 * (arg1rm % arg2cm);
		checkMatrixMatrixMultiply(arg1rm,arg2cm,resultcm,-2.0,1.5);
	}
	
	{
		std::cout<<"crr"<<std::endl;
		matrix<double,row_major> resultrm({rows,columns},1.5);
		noalias(resultrm) += -2.0 * (arg1cm % arg2rm);
		checkMatrixMatrixMultiply(arg1cm,arg2rm,resultrm,-2.0,1.5);
	}
	{
		std::cout<<"crc"<<std::endl;
		matrix<double,column_major> resultcm({rows,columns},1.5);
		noalias(resultcm) += -2.0 * (arg1cm % arg2rm);
		checkMatrixMatrixMultiply(arg1cm,arg2rm,resultcm,-2.0,1.5);
	}
	{
		std::cout<<"ccr"<<std::endl;
		matrix<double,row_major> resultrm({rows,columns},1.5);
		noalias(resultrm) += -2.0 * (arg1cm % arg2cm);
		checkMatrixMatrixMultiply(arg1cm,arg2cm,resultrm,-2.0,1.5);
	}
	{
		std::cout<<"ccc"<<std::endl;
		matrix<double,column_major> resultcm({rows,columns},1.5);
		noalias(resultcm) += -2.0 * (arg1cm % arg2cm);
		checkMatrixMatrixMultiply(arg1cm,arg2cm,resultcm,-2.0,1.5);
	}
	
	std::cout<<"\nchecking sparse-dense matrix matrix assign multiply"<<std::endl;
	//testexpressions of the form A=B*C
	{
		std::cout<<"rrr"<<std::endl;
		matrix<double,row_major> resultrm({rows,columns},1.5);
		noalias(resultrm) = -2.0 * (arg1rm % arg2rm);
		checkMatrixMatrixMultiply(arg1rm,arg2rm,resultrm,-2.0,0);
	}
	{
		std::cout<<"rrc"<<std::endl;
		matrix<double,column_major> resultcm({rows,columns},1.5);
		noalias(resultcm) = -2.0 * (arg1rm % arg2rm);
		checkMatrixMatrixMultiply(arg1rm,arg2rm,resultcm,-2.0,0);
	}
	{
		std::cout<<"rcr"<<std::endl;
		matrix<double,row_major> resultrm({rows,columns},1.5);
		noalias(resultrm) = -2.0 * (arg1rm % arg2cm);
		checkMatrixMatrixMultiply(arg1rm,arg2cm,resultrm,-2.0,0);
	}
	{
		std::cout<<"rcc"<<std::endl;
		matrix<double,column_major> resultcm({rows,columns},1.5);
		noalias(resultcm) = -2.0 * (arg1rm % arg2cm);
		checkMatrixMatrixMultiply(arg1rm,arg2cm,resultcm,-2.0,0);
	}
	
	{
		std::cout<<"crr"<<std::endl;
		matrix<double,row_major> resultrm({rows,columns},1.5);
		noalias(resultrm) = -2.0 * (arg1cm % arg2rm);
		checkMatrixMatrixMultiply(arg1cm,arg2rm,resultrm,-2.0,0);
	}
	{
		std::cout<<"crc"<<std::endl;
		matrix<double,column_major> resultcm({rows,columns},1.5);
		noalias(resultcm) = -2.0 * (arg1cm % arg2rm);
		checkMatrixMatrixMultiply(arg1cm,arg2rm,resultcm,-2.0,0);
	}
	{
		std::cout<<"ccr"<<std::endl;
		matrix<double,row_major> resultrm({rows,columns},1.5);
		noalias(resultrm) = -2.0 * (arg1cm % arg2cm);
		checkMatrixMatrixMultiply(arg1cm,arg2cm,resultrm,-2.0,0);
	}
	{
		std::cout<<"ccc"<<std::endl;
		matrix<double,column_major> resultcm({rows,columns},1.5);
		noalias(resultcm) = -2.0 * (arg1cm % arg2cm);
		checkMatrixMatrixMultiply(arg1cm,arg2cm,resultcm,-2.0,0);
	}
}

BOOST_AUTO_TEST_CASE( Remora_prod_matrix_matrix_sparse_sparse ){
	std::size_t rows = 50;
	std::size_t columns = 80;
	std::size_t middle = 33;
	//initialize the arguments in both row and column major
	compressed_matrix<double> arg1rm(rows,middle);	
	for(std::size_t i = 0; i != rows; ++i){
		auto pos = arg1rm.major_begin(i);
		for(std::size_t j = i%3; j < middle; j+=(i+1)){
			pos = arg1rm.set_element(pos,j,2.0*(20*i+1)+1.0);
		}
	}
	compressed_matrix<double, std::size_t, column_major> arg1cm = arg1rm;	

	compressed_matrix<double> arg2rm(middle,columns);
	for(std::size_t i = 0; i != middle; ++i){
		auto pos = arg2rm.major_begin(i);
		for(std::size_t j = i%3+1; j < columns; j+=(i/2 +1)){
			pos = arg2rm.set_element(pos,j,2*(20*i+1)+1.0);
		}
	}
	compressed_matrix<double, std::size_t, column_major> arg2cm = arg2rm;
	
	std::cout<<"\nchecking sparse-sparse matrix matrix plusassign multiply"<<std::endl;
	//test first expressions of the form A+=B*C
	{
		std::cout<<"rrr"<<std::endl;
		matrix<double,row_major> resultrm({rows,columns},1.5);
		noalias(resultrm) += -2.0 * (arg1rm % arg2rm);
		checkMatrixMatrixMultiply(arg1rm,arg2rm,resultrm,-2.0,1.5);
	}
	{
		std::cout<<"rrc"<<std::endl;
		matrix<double,column_major> resultcm({rows,columns},1.5);
		noalias(resultcm) += -2.0 * (arg1rm % arg2rm);
		checkMatrixMatrixMultiply(arg1rm,arg2rm,resultcm,-2.0,1.5);
	}
	{
		std::cout<<"rcr"<<std::endl;
		matrix<double,row_major> resultrm({rows,columns},1.5);
		noalias(resultrm) += -2.0 * (arg1rm % arg2cm);
		checkMatrixMatrixMultiply(arg1rm,arg2cm,resultrm,-2.0,1.5);
	}
	{
		std::cout<<"rcc"<<std::endl;
		matrix<double,column_major> resultcm({rows,columns},1.5);
		noalias(resultcm) += -2.0 * (arg1rm % arg2cm);
		checkMatrixMatrixMultiply(arg1rm,arg2cm,resultcm,-2.0,1.5);
	}
	
	{
		std::cout<<"crr"<<std::endl;
		matrix<double,row_major> resultrm({rows,columns},1.5);
		noalias(resultrm) += -2.0 * (arg1cm % arg2rm);
		checkMatrixMatrixMultiply(arg1cm,arg2rm,resultrm,-2.0,1.5);
	}
	{
		std::cout<<"crc"<<std::endl;
		matrix<double,column_major> resultcm({rows,columns},1.5);
		noalias(resultcm) += -2.0 * (arg1cm % arg2rm);
		checkMatrixMatrixMultiply(arg1cm,arg2rm,resultcm,-2.0,1.5);
	}
	{
		std::cout<<"ccr"<<std::endl;
		matrix<double,row_major> resultrm({rows,columns},1.5);
		noalias(resultrm) += -2.0 * (arg1cm % arg2cm);
		checkMatrixMatrixMultiply(arg1cm,arg2cm,resultrm,-2.0,1.5);
	}
	{
		std::cout<<"ccc"<<std::endl;
		matrix<double,column_major> resultcm({rows,columns},1.5);
		noalias(resultcm) += -2.0 * (arg1cm % arg2cm);
		checkMatrixMatrixMultiply(arg1cm,arg2cm,resultcm,-2.0,1.5);
	}
	
	std::cout<<"\nchecking sparse-sparse matrix matrix assign multiply"<<std::endl;
	//testexpressions of the form A=B*C
	{
		std::cout<<"rrr"<<std::endl;
		matrix<double,row_major> resultrm({rows,columns},1.5);
		noalias(resultrm) = -2.0 * (arg1rm % arg2rm);
		checkMatrixMatrixMultiply(arg1rm,arg2rm,resultrm,-2.0,0);
	}
	{
		std::cout<<"rrc"<<std::endl;
		matrix<double,column_major> resultcm({rows,columns},1.5);
		noalias(resultcm) = -2.0 * (arg1rm % arg2rm);
		checkMatrixMatrixMultiply(arg1rm,arg2rm,resultcm,-2.0,0);
	}
	{
		std::cout<<"rcr"<<std::endl;
		matrix<double,row_major> resultrm({rows,columns},1.5);
		noalias(resultrm) = -2.0 * (arg1rm % arg2cm);
		checkMatrixMatrixMultiply(arg1rm,arg2cm,resultrm,-2.0,0);
	}
	{
		std::cout<<"rcc"<<std::endl;
		matrix<double,column_major> resultcm({rows,columns},1.5);
		noalias(resultcm) = -2.0 * (arg1rm % arg2cm);
		checkMatrixMatrixMultiply(arg1rm,arg2cm,resultcm,-2.0,0);
	}
	
	{
		std::cout<<"crr"<<std::endl;
		matrix<double,row_major> resultrm({rows,columns},1.5);
		noalias(resultrm) = -2.0 * (arg1cm % arg2rm);
		checkMatrixMatrixMultiply(arg1cm,arg2rm,resultrm,-2.0,0);
	}
	{
		std::cout<<"crc"<<std::endl;
		matrix<double,column_major> resultcm({rows,columns},1.5);
		noalias(resultcm) = -2.0 * (arg1cm % arg2rm);
		checkMatrixMatrixMultiply(arg1cm,arg2rm,resultcm,-2.0,0);
	}
	{
		std::cout<<"ccr"<<std::endl;
		matrix<double,row_major> resultrm({rows,columns},1.5);
		noalias(resultrm) = -2.0 * (arg1cm % arg2cm);
		checkMatrixMatrixMultiply(arg1cm,arg2cm,resultrm,-2.0,0);
	}
	{
		std::cout<<"ccc"<<std::endl;
		matrix<double,column_major> resultcm({rows,columns},1.5);
		noalias(resultcm) = -2.0 * (arg1cm % arg2cm);
		checkMatrixMatrixMultiply(arg1cm,arg2cm,resultcm,-2.0,0);
	}
}*/


BOOST_AUTO_TEST_SUITE_END()