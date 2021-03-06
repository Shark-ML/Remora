#define BOOST_TEST_MODULE Remora_hip_prod
#include <boost/test/unit_test.hpp>
#include <boost/test/floating_point_comparison.hpp>

#include <remora/matrix_expression.hpp>
#include <remora/dense.hpp>
#include <remora/device_copy.hpp>

#include <iostream>
using namespace remora;

template<class M, class V, class Result>
void checkMatrixVectorMultiply(M const& arg1_hip, V const& arg2_hip, Result const& result_hip, float factor, float init = 0){
	BOOST_REQUIRE_EQUAL(arg1_hip.size1(), result_hip.size());
	BOOST_REQUIRE_EQUAL(arg2_hip.size(), arg1_hip.size2());
	
	matrix<float> arg1 = copy_to_cpu(arg1_hip);
	vector<float> arg2 = copy_to_cpu(arg2_hip);
	vector<float> result = copy_to_cpu(result_hip); 
	
	for(std::size_t i = 0; i != arg1.size1(); ++i){
		float test_result = init;
		for(std::size_t k = 0; k != arg1.size2(); ++k){
			test_result += factor * arg1(i,k)*arg2(k);
		}
		BOOST_CHECK_CLOSE(result(i), test_result,1.e-2);
	}
}

BOOST_AUTO_TEST_SUITE (Remora_hip_prod)

BOOST_AUTO_TEST_CASE( Remora_hip_prod_vector_dense ){
	std::size_t rows = 50;
	std::size_t columns = 40;
	//initialize the arguments in both row and column major as well as transposed
	matrix<float,row_major> arg1rm_cpu(rows,columns);
	matrix<float,column_major> arg1cm_cpu(rows,columns);
	matrix<float,row_major> arg1rmt_cpu(columns,rows);
	matrix<float,column_major> arg1cmt_cpu(columns,rows);
	for(std::size_t i = 0; i != rows; ++i){
		for(std::size_t j = 0; j != columns; ++j){
			arg1rm_cpu(i,j) = arg1cm_cpu(i,j) = 0.01*i*columns+0.1*j;
			arg1rmt_cpu(j,i) = arg1cmt_cpu(j,i) = 0.01*i*columns+0.1*j;
		}
	}
	vector<float> arg2_cpu(columns);
	for(std::size_t j = 0; j != columns; ++j){
		arg2_cpu(j)  = 0.1*j+0.1;
	}
	
	matrix<float,row_major, hip_tag> arg1rm = copy_to_device(arg1rm_cpu, hip_tag());
	matrix<float,column_major, hip_tag> arg1cm = copy_to_device(arg1cm_cpu, hip_tag());
	matrix<float,row_major, hip_tag> arg1rmt = copy_to_device(arg1rmt_cpu, hip_tag());
	matrix<float,column_major, hip_tag> arg1cmt = copy_to_device(arg1cmt_cpu, hip_tag());
	vector<float, hip_tag> arg2 = copy_to_device(arg2_cpu, hip_tag());
	std::cout<<"\nchecking dense matrix-vector plusassign multiply"<<std::endl;
    
	//test first expressions of the form A += alpha*B*C 
	{
		std::cout<<"row major Ax"<<std::endl;
		vector<float, hip_tag> result(rows,1.5);
		noalias(result) += -2*prod(arg1rm,arg2);
		checkMatrixVectorMultiply(arg1rm,arg2,result,-2.0,1.5);
	}
	{
		std::cout<<"column major Ax"<<std::endl;
		vector<float, hip_tag> result(rows,1.5);
		noalias(result) += -2*prod(arg1cm,arg2);
		checkMatrixVectorMultiply(arg1cm,arg2,result,-2.0,1.5);
	}
	{
		std::cout<<"row major xA"<<std::endl;
		vector<float, hip_tag> result(rows,1.5);
		noalias(result) += -2*prod(arg2,arg1rmt);
		checkMatrixVectorMultiply(arg1rm,arg2,result,-2.0,1.5);
	}
	{
		std::cout<<"column major xA"<<std::endl;
		vector<float, hip_tag> result(rows,1.5);
		noalias(result) += -2*prod(arg2,arg1cmt);
		checkMatrixVectorMultiply(arg1cm,arg2,result,-2.0,1.5);
	}
	std::cout<<"\nchecking dense matrix-vector assign multiply"<<std::endl;
	//test expressions of the form A=B*C
	{
		std::cout<<"row major Ax"<<std::endl;
		vector<float, hip_tag> result(rows,1.5);
		noalias(result) = -2*prod(arg1rm,arg2);
		checkMatrixVectorMultiply(arg1rm,arg2,result,-2.0,0);
	}
	{
		std::cout<<"column major Ax"<<std::endl;
		vector<float, hip_tag> result(rows,1.5);
		noalias(result) = -2*prod(arg1cm,arg2);
		checkMatrixVectorMultiply(arg1cm,arg2,result,-2.0,0);
	}
	{
		std::cout<<"row major xA"<<std::endl;
		vector<float, hip_tag> result(rows,1.5);
		noalias(result) = -2*prod(arg2,arg1rmt);
		checkMatrixVectorMultiply(arg1rm,arg2,result,-2.0,0);
	}
	{
		std::cout<<"column major xA"<<std::endl;
		vector<float, hip_tag> result(rows,1.5);
		noalias(result) = -2*prod(arg2,arg1cmt);
		checkMatrixVectorMultiply(arg1cm,arg2,result,-2.0,0);
	}
}

//we test using the textbook definition.
template<class Arg1, class Arg2, class Result>
void checkMatrixMatrixMultiply(Arg1 const& arg1_hip, Arg2 const& arg2_hip, Result const& result_hip, float factor, float init = 0){
	BOOST_REQUIRE_EQUAL(arg1_hip.size1(), result_hip.size1());
	BOOST_REQUIRE_EQUAL(arg2_hip.size2(), result_hip.size2());
	
	matrix<float> arg1 = copy_to_cpu(arg1_hip);
	matrix<float> arg2 = copy_to_cpu(arg2_hip);
	matrix<float> result = copy_to_cpu(result_hip); 
	
	for(std::size_t i = 0; i != arg1.size1(); ++i){
		for(std::size_t j = 0; j != arg2.size2(); ++j){
			float test_result = init;
			for(std::size_t k = 0; k != arg1.size2(); ++k){
				 test_result += factor * arg1(i,k)*arg2(k,j);
			}
			BOOST_CHECK_CLOSE(result(i,j), test_result,1.e-2);
		}
	}
}

BOOST_AUTO_TEST_CASE( Remora_prod_hip_matrix_dense_dense){
	std::size_t rows = 50;
	std::size_t columns = 80;
	std::size_t middle = 33;
	//initialize the arguments in both row and column major
	matrix<float,row_major> arg1rm_cpu(rows,middle);
	matrix<float,column_major> arg1cm_cpu(rows,middle);
	for(std::size_t i = 0; i != rows; ++i){
		for(std::size_t j = 0; j != middle; ++j){
			arg1rm_cpu(i,j) = arg1cm_cpu(i,j) = 0.1*i*middle+0.2*j;
		}
	}
	matrix<float,row_major> arg2rm_cpu(middle,columns);
	matrix<float,column_major> arg2cm_cpu(middle,columns);
	for(std::size_t i = 0; i != middle; ++i){
		for(std::size_t j = 0; j != columns; ++j){
			arg2rm_cpu(i,j) = arg2cm_cpu(i,j) = 0.1*i*columns+1.5*j;
		}
	}
	
	matrix<float,row_major, hip_tag> arg1rm = copy_to_device(arg1rm_cpu, hip_tag());
	matrix<float,column_major, hip_tag> arg1cm = copy_to_device(arg1cm_cpu, hip_tag());
	matrix<float,row_major, hip_tag> arg2rm = copy_to_device(arg2rm_cpu, hip_tag());
	matrix<float,column_major, hip_tag> arg2cm = copy_to_device(arg2cm_cpu, hip_tag());
	
	std::cout<<"\nchecking dense-dense matrix-matrix plusassign multiply"<<std::endl;
	//test first expressions of the form A+=B*C
	{
		std::cout<<"rrr"<<std::endl;
		matrix<float,row_major, hip_tag> resultrm(rows,columns,1.5);
		noalias(resultrm) += -2.0 * prod(arg1rm,arg2rm);
		checkMatrixMatrixMultiply(arg1rm,arg2rm,resultrm,-2.0,1.5);
	}
	{
		std::cout<<"rrc"<<std::endl;
		matrix<float,column_major, hip_tag> resultcm(rows,columns,1.5);
		noalias(resultcm) += -2.0 * prod(arg1rm,arg2rm);
		checkMatrixMatrixMultiply(arg1rm,arg2rm,resultcm,-2.0,1.5);
	}
	{
		std::cout<<"rcr"<<std::endl;
		matrix<float,row_major, hip_tag> resultrm(rows,columns,1.5);
		noalias(resultrm) += -2.0 * prod(arg1rm,arg2cm);
		checkMatrixMatrixMultiply(arg1rm,arg2cm,resultrm,-2.0,1.5);
	}
	{
		std::cout<<"rcc"<<std::endl;
		matrix<float,column_major, hip_tag> resultcm(rows,columns,1.5);
		noalias(resultcm) += -2.0 * prod(arg1rm,arg2cm);
		checkMatrixMatrixMultiply(arg1rm,arg2cm,resultcm,-2.0,1.5);
	}
	
	{
		std::cout<<"crr"<<std::endl;
		matrix<float,row_major, hip_tag> resultrm(rows,columns,1.5);
		noalias(resultrm) += -2.0 * prod(arg1cm,arg2rm);
		checkMatrixMatrixMultiply(arg1cm,arg2rm,resultrm,-2.0,1.5);
	}
	{
		std::cout<<"crc"<<std::endl;
		matrix<float,column_major, hip_tag> resultcm(rows,columns,1.5);
		noalias(resultcm) += -2.0 * prod(arg1cm,arg2rm);
		checkMatrixMatrixMultiply(arg1cm,arg2rm,resultcm,-2.0,1.5);
	}
	{
		std::cout<<"ccr"<<std::endl;
		matrix<float,row_major, hip_tag> resultrm(rows,columns,1.5);
		noalias(resultrm) += -2.0 * prod(arg1cm,arg2cm);
		checkMatrixMatrixMultiply(arg1cm,arg2cm,resultrm,-2.0,1.5);
	}
	{
		std::cout<<"ccc"<<std::endl;
		matrix<float,column_major, hip_tag> resultcm(rows,columns,1.5);
		noalias(resultcm) += -2.0 * prod(arg1cm,arg2cm);
		checkMatrixMatrixMultiply(arg1cm,arg2cm,resultcm,-2.0,1.5);
	}
	
	std::cout<<"\nchecking dense-dense matrix-matrix assign multiply"<<std::endl;
	//testexpressions of the form A=B*C
	{
		std::cout<<"rrr"<<std::endl;
		matrix<float,row_major, hip_tag> resultrm(rows,columns,1.5);
		noalias(resultrm) = -2.0 * prod(arg1rm,arg2rm);
		checkMatrixMatrixMultiply(arg1rm,arg2rm,resultrm,-2.0,0);
	}
	{
		std::cout<<"rrc"<<std::endl;
		matrix<float,column_major, hip_tag> resultcm(rows,columns,1.5);
		noalias(resultcm) = -2.0 * prod(arg1rm,arg2rm);
		checkMatrixMatrixMultiply(arg1rm,arg2rm,resultcm,-2.0,0);
	}
	{
		std::cout<<"rcr"<<std::endl;
		matrix<float,row_major, hip_tag> resultrm(rows,columns,1.5);
		noalias(resultrm) = -2.0 * prod(arg1rm,arg2cm);
		checkMatrixMatrixMultiply(arg1rm,arg2cm,resultrm,-2.0,0);
	}
	{
		std::cout<<"rcc"<<std::endl;
		matrix<float,column_major, hip_tag> resultcm(rows,columns,1.5);
		noalias(resultcm) = -2.0 * prod(arg1rm,arg2cm);
		checkMatrixMatrixMultiply(arg1rm,arg2cm,resultcm,-2.0,0);
	}
	
	{
		std::cout<<"crr"<<std::endl;
		matrix<float,row_major, hip_tag> resultrm(rows,columns,1.5);
		noalias(resultrm) = -2.0 * prod(arg1cm,arg2rm);
		checkMatrixMatrixMultiply(arg1cm,arg2rm,resultrm,-2.0,0);
	}
	{
		std::cout<<"crc"<<std::endl;
		matrix<float,column_major, hip_tag> resultcm(rows,columns,1.5);
		noalias(resultcm) = -2.0 * prod(arg1cm,arg2rm);
		checkMatrixMatrixMultiply(arg1cm,arg2rm,resultcm,-2.0,0);
	}
	{
		std::cout<<"ccr"<<std::endl;
		matrix<float,row_major, hip_tag> resultrm(rows,columns,1.5);
		noalias(resultrm) = -2.0 * prod(arg1cm,arg2cm);
		checkMatrixMatrixMultiply(arg1cm,arg2cm,resultrm,-2.0,0);
	}
	{
		std::cout<<"ccc"<<std::endl;
		matrix<float,column_major, hip_tag> resultcm(rows,columns,1.5);
		noalias(resultcm) = -2.0 * prod(arg1cm,arg2cm);
		checkMatrixMatrixMultiply(arg1cm,arg2cm,resultcm,-2.0,0);
	}
}
BOOST_AUTO_TEST_SUITE_END()
