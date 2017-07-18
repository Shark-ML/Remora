#define BOOST_TEST_MODULE Remora_Vector_vector_expression
#include <boost/test/unit_test.hpp>
#include <boost/test/floating_point_comparison.hpp>

#include <remora/vector_expression.hpp>
#include <remora/vector.hpp>
#include <remora/vector_sparse.hpp>
#include <remora/io.hpp>

#include <iostream>
using namespace remora;

//////////////////DENSE//////////////////////////////

template<class Operation, class Result>
void checkDenseExpressionEquality(
	vector_expression<Operation, cpu_tag> const& op, Result const& result
){
	BOOST_REQUIRE_EQUAL(op().size(), result.size());
	
	typename Operation::const_iterator pos = op().begin();
	for(std::size_t i = 0; i != op().size(); ++i,++pos){
		BOOST_REQUIRE(pos != op().end());
		BOOST_CHECK_EQUAL(pos.index(), i);
		BOOST_CHECK_SMALL(result(i) - op()(i),typename Result::value_type(1.e-10));
		BOOST_CHECK_SMALL(*pos - op()(i),typename Result::value_type(1.e-10));
	}
	BOOST_REQUIRE(pos == op().end());

	//test block eval
	remora::vector<typename Result::value_type> res1(result.size(),1.0);
	remora::vector<typename Result::value_type> res2(result.size(),1.0);
	op().assign_to(res1,2);
	op().plus_assign_to(res2,2);
	
	for(std::size_t i = 0; i != op().size(); ++i,++pos){
		BOOST_CHECK_SMALL(res1(i) - 2*op()(i),typename Result::value_type(1.e-7));
		BOOST_CHECK_SMALL(res2(i) - 2*op()(i) - 1,typename Result::value_type(1.e-7));
	}
	
}

const std::size_t Dimensions = 10;

/////////////////////////////////////////////////////////////
//////UNARY TRANSFORMATIONS///////
////////////////////////////////////////////////////////////
BOOST_AUTO_TEST_SUITE (Remora_vector_expression)

BOOST_AUTO_TEST_CASE( Remora_Vector_Unary_Minus )
{
	vector<double> x(Dimensions); 
	vector<double> result(Dimensions);
	
	for (size_t i = 0; i < Dimensions; i++)
	{
		x(i) = i-3.0;
		result(i)= -x(i);
	}
	checkDenseExpressionEquality(-x,result);
}
BOOST_AUTO_TEST_CASE( Remora_Vector_Scalar_Multiply )
{
	vector<double> x(Dimensions); 
	vector<double> result(Dimensions);
	
	for (size_t i = 0; i < Dimensions; i++)
	{
		x(i) = i-3.0;
		result(i)= 5.0*x(i);
	}
	checkDenseExpressionEquality(5.0*x,result);
	checkDenseExpressionEquality(x*5.0,result);
}
BOOST_AUTO_TEST_CASE( Remora_Vector_Scalar_Div )
{
	vector<double> x(Dimensions); 
	vector<double> result(Dimensions);
	
	for (size_t i = 0; i < Dimensions; i++)
	{
		x(i) = i-3.0;
		result(i)= x(i)/5.0;
	}
	checkDenseExpressionEquality(x/5.0,result);
}
BOOST_AUTO_TEST_CASE( Remora_Vector_Abs )
{
	vector<double> x(Dimensions); 
	vector<double> result(Dimensions);
	
	for (size_t i = 0; i < Dimensions; i++)
	{
		x(i) = 3.0-i;
		result(i)= std::abs(x(i));
	}
	checkDenseExpressionEquality(abs(x),result);
}
BOOST_AUTO_TEST_CASE( Remora_Vector_Sqr )
{
	vector<double> x(Dimensions); 
	vector<double> result(Dimensions);
	
	for (size_t i = 0; i < Dimensions; i++)
	{
		x(i) = 3.0-i;
		result(i)= x(i)*x(i);
	}
	checkDenseExpressionEquality(sqr(x),result);
}
BOOST_AUTO_TEST_CASE( Remora_Vector_Sqrt )
{
	vector<double> x(Dimensions); 
	vector<double> result(Dimensions);
	
	for (size_t i = 0; i < Dimensions; i++)
	{
		x(i) = i;
		result(i)= sqrt(x(i));
	}
	checkDenseExpressionEquality(sqrt(x),result);
}

BOOST_AUTO_TEST_CASE( Remora_Vector_Cbrt )
{
	vector<double> x(Dimensions); 
	vector<double> result(Dimensions);
	
	for (size_t i = 0; i < Dimensions; i++)
	{
		x(i) = i;
		result(i)= cbrt(x(i));
	}
	checkDenseExpressionEquality(cbrt(x),result);
}
BOOST_AUTO_TEST_CASE( Remora_Vector_Exp )
{
	vector<double> x(Dimensions); 
	vector<double> result(Dimensions);
	
	for (size_t i = 0; i < Dimensions; i++)
	{
		x(i) = i;
		result(i)=std::exp(x(i));
	}
	checkDenseExpressionEquality(exp(x),result);
}
BOOST_AUTO_TEST_CASE( Remora_Vector_Log )
{

	vector<double> x(Dimensions); 
	vector<double> result(Dimensions);
	
	for (size_t i = 0; i < Dimensions; i++)
	{
		x(i) = i+1;
		result(i)=std::log(x(i));
	}
	checkDenseExpressionEquality(log(x),result);
}

BOOST_AUTO_TEST_CASE( Remora_Vector_Sin )
{

	vector<double> x(Dimensions); 
	vector<double> result(Dimensions);
	
	for (size_t i = 0; i < Dimensions; i++)
	{
		x(i) = i+1;
		result(i)=std::sin(x(i));
	}
	checkDenseExpressionEquality(sin(x),result);
}

BOOST_AUTO_TEST_CASE( Remora_Vector_Cos )
{

	vector<double> x(Dimensions); 
	vector<double> result(Dimensions);
	
	for (size_t i = 0; i < Dimensions; i++)
	{
		x(i) = i+1;
		result(i)=std::cos(x(i));
	}
	checkDenseExpressionEquality(cos(x),result);
}

BOOST_AUTO_TEST_CASE( Remora_Vector_Tan )
{

	vector<double> x(Dimensions); 
	vector<double> result(Dimensions);
	
	for (size_t i = 0; i < Dimensions; i++)
	{
		x(i) = i+1;
		result(i)=std::tan(x(i));
	}
	checkDenseExpressionEquality(tan(x),result);
}

BOOST_AUTO_TEST_CASE( Remora_Vector_aSin )
{

	vector<double> x(Dimensions); 
	vector<double> result(Dimensions);
	
	for (size_t i = 0; i < Dimensions; i++)
	{
		x(i) = (2.0*(i+1))/Dimensions -1.0;
		result(i) = std::asin(x(i));
	}
	checkDenseExpressionEquality(asin(x),result);
}

BOOST_AUTO_TEST_CASE( Remora_Vector_aCos )
{

	vector<double> x(Dimensions); 
	vector<double> result(Dimensions);
	
	for (size_t i = 0; i < Dimensions; i++)
	{
		x(i) = (2.0*(i+1))/Dimensions -1.0;
		result(i) = std::acos(x(i));
	}
	checkDenseExpressionEquality(acos(x),result);
}

BOOST_AUTO_TEST_CASE( Remora_Vector_aTan )
{

	vector<double> x(Dimensions); 
	vector<double> result(Dimensions);
	
	for (size_t i = 0; i < Dimensions; i++)
	{
		x(i) = (2.0*(i+1))/Dimensions -1.0;
		result(i) = std::atan(x(i));
	}
	checkDenseExpressionEquality(atan(x),result);
}

BOOST_AUTO_TEST_CASE( Remora_Vector_erf )
{

	vector<double> x(Dimensions); 
	vector<double> result(Dimensions);
	
	for (size_t i = 0; i < Dimensions; i++)
	{
		x(i) = i+1;
		result(i)=std::erf(x(i));
	}
	checkDenseExpressionEquality(erf(x),result);
}
BOOST_AUTO_TEST_CASE( Remora_Vector_erfc )
{

	vector<double> x(Dimensions); 
	vector<double> result(Dimensions);
	
	for (size_t i = 0; i < Dimensions; i++)
	{
		x(i) = i+1;
		result(i)=std::erfc(x(i));
	}
	checkDenseExpressionEquality(erfc(x),result);
}

BOOST_AUTO_TEST_CASE( Remora_Vector_Tanh )
{
	vector<double> x(Dimensions); 
	vector<double> result(Dimensions);
	
	for (size_t i = 0; i < Dimensions; i++){
		x(i) = i;
		result(i)=std::tanh(x(i));
	}
	checkDenseExpressionEquality(tanh(x),result);
}
BOOST_AUTO_TEST_CASE( Remora_Vector_Sigmoid )
{
	vector<double> x(Dimensions); 
	vector<double> result(Dimensions);
	
	for (size_t i = 0; i < Dimensions; i++){
		x(i) = i;
		result(i) = 1.0/(1.0+std::exp(-x(i)));
	}
	checkDenseExpressionEquality(sigmoid(x),result);
}
BOOST_AUTO_TEST_CASE( Remora_Vector_SoftPlus )
{
	vector<double> x(Dimensions); 
	vector<double> result(Dimensions);
	
	for (size_t i = 0; i < Dimensions; i++)
	{
		x(i) = i;
		result(i) = std::log(1+std::exp(x(i)));
	}
	checkDenseExpressionEquality(softPlus(x),result);
}
BOOST_AUTO_TEST_CASE( Remora_Vector_Pow )
{
	vector<double> x(Dimensions); 
	vector<double> result(Dimensions);
	
	for (size_t i = 0; i < Dimensions; i++)
	{
		x(i) = i+1.0;
		result(i)= std::pow(x(i),3.2);
	}
	checkDenseExpressionEquality(pow(x,3.2),result);
}

/////////////////////////////////////////////////////
///////BINARY OPERATIONS//////////
/////////////////////////////////////////////////////

BOOST_AUTO_TEST_CASE( Remora_Vector_Binary_Plus)
{
	vector<double> x(Dimensions); 
	vector<double> y(Dimensions); 
	vector<double> result(Dimensions);
	
	for (size_t i = 0; i < Dimensions; i++)
	{
		x(i) = i;
		y(i) = i+Dimensions;
		result(i) = x(i)+y(i);
	}
	checkDenseExpressionEquality(x+y,result);
}
BOOST_AUTO_TEST_CASE( Remora_Vector_Binary_Minus)
{
	vector<double> x(Dimensions); 
	vector<double> y(Dimensions); 
	vector<double> result(Dimensions);
	
	for (size_t i = 0; i < Dimensions; i++)
	{
		x(i) = i;
		y(i) = -3.0*i+Dimensions;
		result(i) = x(i)-y(i);
	}
	checkDenseExpressionEquality(x-y,result);
}

BOOST_AUTO_TEST_CASE( Remora_Vector_Binary_Multiply)
{
	vector<double> x(Dimensions); 
	vector<double> y(Dimensions); 
	vector<double> result(Dimensions);
	
	for (size_t i = 0; i < Dimensions; i++)
	{
		x(i) = i;
		y(i) = -3.0*i+Dimensions;
		result(i) = x(i)*y(i);
	}
	checkDenseExpressionEquality(x*y,result);
	checkDenseExpressionEquality(element_prod(x,y),result);
}

BOOST_AUTO_TEST_CASE( Remora_Vector_Binary_Div)
{
	vector<double> x(Dimensions); 
	vector<double> y(Dimensions); 
	vector<double> result(Dimensions);
	
	for (size_t i = 0; i < Dimensions; i++)
	{
		x(i) = 3.0*i+3.0;
		y(i) = i+1;
		result(i) = x(i)/y(i);
	}
	checkDenseExpressionEquality(x/y,result);
	checkDenseExpressionEquality(element_div(x,y),result);
}

BOOST_AUTO_TEST_CASE( Remora_Vector_Safe_Div )
{
	vector<double> x(Dimensions); 
	vector<double> y(Dimensions); 
	vector<double> result(Dimensions);
	
	for (size_t i = 0; i < Dimensions; i++)
	{
		x(i) = i;
		y(i) = i % 3;
		result(i) = (i % 3 == 0)? 2.0: x(i)/y(i);
	}
	checkDenseExpressionEquality(safe_div(x,y,2.0),result);
}

/////////////////////////////////////////////////////
///////////Vector Concatenation///////////
/////////////////////////////////////////////////////
BOOST_AUTO_TEST_CASE( Remora_Vector_Scalar_Concat )
{
	vector<double> x(Dimensions); 
	double alpha = 2.0;
	vector<double> result(Dimensions+1);
	vector<double> result2(Dimensions+1,1.0);
	result(Dimensions) = 2*alpha;
	result2(Dimensions) += 2*alpha;
	
	for (size_t i = 0; i < Dimensions; i++){
		x(i) = exp(-(i-5.0)*(i-5.0));
		result(i) = 2*x(i);
		result2(i) += 2*x(i);
	}
	vector<double> test_assign(Dimensions+1,1.0);
	vector<double> test_plus_assign(Dimensions+1,1.0); 
	(x|alpha).assign_to(test_assign,2);
	(x|alpha).plus_assign_to(test_plus_assign,2);
	
	for (size_t i = 0; i < Dimensions; i++){
		BOOST_CHECK_SMALL(test_assign(i) - result(i),1.e-7);
		BOOST_CHECK_SMALL(test_plus_assign(i) - result2(i),1.e-7);
	}
}
BOOST_AUTO_TEST_CASE( Remora_Scalar_Vector_Concat )
{
	vector<double> x(Dimensions); 
	double alpha = 2.0;
	vector<double> result(Dimensions+1,1.0);
	vector<double> result2(Dimensions+1,1.0);
	result(0) = 2*alpha;
	result2(0) += 2*alpha;
	
	for (size_t i = 0; i < Dimensions; i++){
		x(i) = exp(-(i-5.0)*(i-5.0));
		result(i+1) = 2*x(i);
		result2(i+1) += 2*x(i);
	}
	vector<double> test_assign(Dimensions+1,1.0);
	vector<double> test_plus_assign(Dimensions+1,1.0); 
	(alpha|x).assign_to(test_assign,2);
	(alpha|x).plus_assign_to(test_plus_assign,2);
	
	for (size_t i = 0; i < Dimensions; i++){
		BOOST_CHECK_SMALL(test_assign(i) - result(i),1.e-7);
		BOOST_CHECK_SMALL(test_plus_assign(i) - result2(i),1.e-7);
	}
}
BOOST_AUTO_TEST_CASE( Remora_Vector_Vector_Concat )
{
	vector<double> x(Dimensions); 
	vector<double> y(Dimensions); 
	vector<double> result(Dimensions * 2,1.0);
	vector<double> result2(Dimensions * 2,1.0);
	
	for (size_t i = 0; i < Dimensions; i++){
		x(i) = exp(-(i-5.0)*(i-5.0));
		y(i) = exp(-(i-3.0)*(i-3.0));
		result(i) = 2*x(i);
		result(i+Dimensions) = 2*y(i);
		result2(i) += 2*x(i);
		result2(i+Dimensions) += 2*y(i);
	}
	vector<double> test_assign(Dimensions*2,1.0);
	vector<double> test_plus_assign(Dimensions*2,1.0); 
	(x|y).assign_to(test_assign,2);
	(x|y).plus_assign_to(test_plus_assign,2);
	
	for (size_t i = 0; i < Dimensions; i++){
		BOOST_CHECK_SMALL(test_assign(i) - result(i),1.e-7);
		BOOST_CHECK_SMALL(test_plus_assign(i) - result2(i),1.e-7);
	}
}

/////////////////////////////////////////////////////
///////////Vector Reductions///////////
/////////////////////////////////////////////////////

BOOST_AUTO_TEST_CASE( Remora_Vector_Max )
{
	vector<double> x(Dimensions); 
	double result = 1;
	
	for (size_t i = 0; i < Dimensions; i++){
		x(i) = exp(-(i-5.0)*(i-5.0));//max at i = 5
		result = std::max(result,x(i));
	}
	BOOST_CHECK_CLOSE(max(x),result,1.e-10);
}
BOOST_AUTO_TEST_CASE( Remora_Vector_Min )
{
	vector<double> x(Dimensions); 
	double result = -1;
	
	for (size_t i = 0; i < Dimensions; i++){
		x(i) = -std::exp(-(i-5.0)*(i-5.0));//min at i = 5
		result = std::min(result,x(i));
	}
	BOOST_CHECK_CLOSE(min(x),result,1.e-10);
}

BOOST_AUTO_TEST_CASE( Remora_Vector_Arg_Max )
{
	vector<double> x(Dimensions); 
	unsigned int result = 5;
	
	for (size_t i = 0; i < Dimensions; i++){
		x(i) = -abs(5.0-i);//max at i = 5
	}
	BOOST_CHECK_EQUAL(arg_max(x),result);
}

BOOST_AUTO_TEST_CASE( Remora_Vector_Arg_Min )
{
	vector<double> x(Dimensions); 
	unsigned int result = 5;
	
	for (size_t i = 0; i < Dimensions; i++){
		x(i) = -exp(-(i-5.0)*(i-5.0));//min at i = 5
	}
	BOOST_CHECK_EQUAL(arg_min(x),result);
}

BOOST_AUTO_TEST_CASE( Remora_Vector_Sum )
{
	vector<double> x(Dimensions); 
	double result = 0;
	
	for (size_t i = 0; i < Dimensions; i++){
		x(i) = 2*i-5;
		result +=x(i);
	}
	BOOST_CHECK_CLOSE(sum(x),result,1.e-10);
}

BOOST_AUTO_TEST_CASE( Remora_Vector_norm_1 )
{
	vector<double> x(Dimensions); 
	double result = 0;
	
	for (size_t i = 0; i < Dimensions; i++){
		x(i) = 2*i-5;
		result +=std::abs(x(i));
	}
	BOOST_CHECK_CLOSE(norm_1(x),result,1.e-10);
}

BOOST_AUTO_TEST_CASE( Remora_Vector_norm_sqr )
{
	vector<double> x(Dimensions); 
	double result = 0;
	
	for (size_t i = 0; i < Dimensions; i++){
		x(i) = 2.0*i-5;
		result +=x(i)*x(i);
	}
	BOOST_CHECK_CLOSE(norm_sqr(x),result,1.e-10);
}
BOOST_AUTO_TEST_CASE( Remora_Vector_norm_2 )
{
	vector<double> x(Dimensions); 
	double result = 0;
	
	for (size_t i = 0; i < Dimensions; i++){
		x(i) = 2.0*i-5;
		result +=x(i)*x(i);
	}
	result = std::sqrt(result);
	BOOST_CHECK_CLOSE(norm_2(x),result,1.e-10);
}
BOOST_AUTO_TEST_CASE( Remora_Vector_norm_inf )
{
	vector<double> x(Dimensions); 
	for (size_t i = 0; i < Dimensions; i++){
		x(i) = exp(-(i-5.0)*(i-5.0));
	}
	x(8)=-2;
	BOOST_CHECK_EQUAL(norm_inf(x),2.0);
}
BOOST_AUTO_TEST_CASE( Remora_Vector_index_norm_inf )
{
	vector<double> x(Dimensions); 
	
	for (size_t i = 0; i < Dimensions; i++){
		x(i) = exp(-(i-5.0)*(i-5.0));
	}
	x(8)=-2;
	BOOST_CHECK_EQUAL(index_norm_inf(x),8);
}

BOOST_AUTO_TEST_CASE( Remora_Vector_inner_prod )
{
	vector<double> x(Dimensions); 
	vector<double> y(Dimensions); 
	
	for (size_t i = 0; i < Dimensions; i++){
		x(i) = exp(-(i-5.0)*(i-5.0));
		y(i) = exp((i-5.0)*(i-5.0));
	}
	BOOST_CHECK_CLOSE(inner_prod(x,y),(double)Dimensions,1.e-5);
}

/////////////////////////////////////////COMPARISONS///////////////////////////

BOOST_AUTO_TEST_CASE( Remora_Vector_Equal ){
	vector<int> x(Dimensions); 
	vector<int> result(Dimensions);
	
	for (int i = 0; i < (int)Dimensions; i++){
		x(i) = -5 + 2*i;
		result(i) = x(i) == 0;
	}
	checkDenseExpressionEquality(x == 0,result);
}

BOOST_AUTO_TEST_CASE( Remora_Vector_Not_Equal ){
	vector<int> x(Dimensions); 
	vector<int> result(Dimensions);
	
	for (int i = 0; i < (int)Dimensions; i++){
		x(i) = -5 + 2*i;
		result(i) = x(i) != 0;
	}
	checkDenseExpressionEquality(x != 0,result);
}

BOOST_AUTO_TEST_CASE( Remora_Vector_Less ){
	vector<int> x(Dimensions); 
	vector<int> result(Dimensions);
	
	for (int i = 0; i < (int)Dimensions; i++){
		x(i) = -5 + 2*i;
		result(i) = x(i) < 0;
	}
	checkDenseExpressionEquality(x < 0,result);
}

BOOST_AUTO_TEST_CASE( Remora_Vector_Less_Equal ){
	vector<int> x(Dimensions); 
	vector<int> result(Dimensions);
	
	for (int i = 0; i < (int)Dimensions; i++){
		x(i) = -5 + 2*i;
		result(i) = x(i) <= 0;
	}
	checkDenseExpressionEquality(x <= 0,result);
}

BOOST_AUTO_TEST_CASE( Remora_Vector_Greater ){
	vector<int> x(Dimensions); 
	vector<int> result(Dimensions);
	
	for (int i = 0; i < (int)Dimensions; i++){
		x(i) = -5 + 2*i;
		result(i) = x(i) > 0;
	}
	checkDenseExpressionEquality(x > 0,result);
}

BOOST_AUTO_TEST_CASE( Remora_Vector_Greater_Equal ){
	vector<int> x(Dimensions); 
	vector<int> result(Dimensions);
	
	for (int i = 0; i < (int)Dimensions; i++){
		x(i) = -5 + 2*i;
		result(i) = x(i) >= 0;
	}
	checkDenseExpressionEquality(x >= 0,result);
}

//////////////////////////////SPARSE TESTS//////////////////////////////

//we only check the operations which make sense for sparseness, that is sparseness is preserved.


template<class Operation, class Result>
void checkSparseExpressionEquality(
	Operation op, Result const& result
){
	BOOST_CHECK_EQUAL(op.size(), result.size());
	
	typename Operation::const_iterator posOp = op.begin();
	typename Result::const_iterator posResult = result.begin();
	
	for(;posResult != result.end();++posOp,++posResult){
		BOOST_REQUIRE(posOp != op.end());
		BOOST_REQUIRE_EQUAL(posOp.index(), posResult.index());
		BOOST_CHECK_SMALL(*posOp-*posResult,1.e-3);
	}
}

/////////////////////////////////////////////////////////////
//////UNARY TRANSFORMATIONS///////
////////////////////////////////////////////////////////////


std::size_t SparseDimensions = 100;
std::size_t VectorNNZ = 10;

BOOST_AUTO_TEST_CASE( Remora_Sparse_Vector_Unary_Minus )
{
	compressed_vector<double> x(SparseDimensions); 
	compressed_vector<double> result(SparseDimensions);
	
	for (size_t i = 0; i < VectorNNZ; i++)
	{
		x(i*9+3) = i-3.0;
		result(i*9+3)= -x(i*9+3);
	}
	checkSparseExpressionEquality(-x,result);
}
BOOST_AUTO_TEST_CASE( Remora_Sparse_Vector_Scalar_Multiply )
{
	compressed_vector<double> x(SparseDimensions); 
	compressed_vector<double> result(SparseDimensions);
	
	for (size_t i = 0; i < VectorNNZ; i++)
	{
		x(i) = i-3.0;
		result(i)= 5.0*x(i);
	}
	checkSparseExpressionEquality(5.0*x,result);
	checkSparseExpressionEquality(x*5.0,result);
}
BOOST_AUTO_TEST_CASE( Remora_Sparse_Vector_Scalar_Div )
{
	compressed_vector<double> x(SparseDimensions); 
	compressed_vector<double> result(SparseDimensions);
	
	for (size_t i = 0; i < VectorNNZ; i++)
	{
		x(i+90) = i-3.0;
		result(i+90)= x(i+90)/5.0;
	}
	checkSparseExpressionEquality(x/5.0,result);
}
BOOST_AUTO_TEST_CASE( Remora_Sparse_Vector_Abs )
{
	compressed_vector<double> x(SparseDimensions); 
	compressed_vector<double> result(SparseDimensions);
	
	for (size_t i = 0; i < VectorNNZ; i++)
	{
		x(i) = -double(i)+3.0;
		result(i)= std::abs(x(i));
	}
	checkSparseExpressionEquality(abs(x),result);
}
BOOST_AUTO_TEST_CASE( Remora_Sparse_Vector_Sqr )
{
	compressed_vector<double> x(SparseDimensions); 
	compressed_vector<double> result(SparseDimensions);
	
	for (size_t i = 0; i < VectorNNZ; i++)
	{
		x(i) = -double(i)+3.0;
		result(i)= x(i)*x(i);
	}
	checkSparseExpressionEquality(sqr(x),result);
}
BOOST_AUTO_TEST_CASE( Remora_Sparse_Vector_Sqrt )
{
	compressed_vector<double> x(SparseDimensions); 
	compressed_vector<double> result(SparseDimensions);
	
	for (size_t i = 0; i < VectorNNZ; i++)
	{
		x(i) = i;
		result(i)= sqrt(x(i));
	}
	checkSparseExpressionEquality(sqrt(x),result);
}

//////////////////////////////////////////////////////////////
//////BINARY TRANSFORMATIONS///////
/////////////////////////////////////////////////////////////

BOOST_AUTO_TEST_CASE( Remora_Sparse_Vector_Binary_Plus)
{
	compressed_vector<double> x(SparseDimensions); 
	compressed_vector<double> y(SparseDimensions); 
	compressed_vector<double> result(SparseDimensions);
	
	for (size_t i = 0; i < VectorNNZ; i++)
	{
		x(10*i+1) = 0.5*i;
		x(10*i+2) = i;
		
		y(10*i) = 0.5*i;
		y(10*i+1) = 2*i;
		result(10*i) = y(10*i);
		result(10*i+1) = x(10*i+1)+y(10*i+1);
		result(10*i+2) = x(10*i+2);
	}
	checkSparseExpressionEquality(x+y,result);
}
BOOST_AUTO_TEST_CASE( Remora_Sparse_Vector_Binary_Minus)
{
	compressed_vector<double> x(SparseDimensions); 
	compressed_vector<double> y(SparseDimensions); 
	compressed_vector<double> result(SparseDimensions);
	
	for (size_t i = 0; i < VectorNNZ; i++)
	{
		x(10*i+1) = 0.5*i;
		x(10*i+2) = i;
		
		y(10*i) = 0.5*i;
		y(10*i+1) = 2*i;
		result(10*i) = -y(10*i);
		result(10*i+1) = x(10*i+1)-y(10*i+1);
		result(10*i+2) = x(10*i+2);
	}
	checkSparseExpressionEquality(x-y,result);
}

BOOST_AUTO_TEST_CASE( Remora_Sparse_Vector_Binary_Multiply)
{
	compressed_vector<double> x(SparseDimensions); 
	compressed_vector<double> y(SparseDimensions); 
	compressed_vector<double> result(SparseDimensions);
	
	for (size_t i = 0; i < VectorNNZ; i++)
	{
		y(10*i) = 0.5*i+1;
		y(10*i+1) = 2*i+1;

		x(10*i+1) = 0.5*i+1;
		x(10*i+2) = i+1;
		result(10*i+1) = x(10*i+1)*y(10*i+1);
	}
	checkSparseExpressionEquality(x*y,result);
}

//////////////////////////////DENSE-SPARSE TESTS//////////////////////////////

//////////////////////////////////////////////////////////////
//////BINARY TRANSFORMATIONS///////
/////////////////////////////////////////////////////////////

BOOST_AUTO_TEST_CASE( Remora_Dense_Sparse_Vector_Binary_Plus)
{
	vector<double> x(SparseDimensions); 
	compressed_vector<double> y(SparseDimensions); 
	vector<double> result(SparseDimensions);
	
	
	for (size_t i = 0; i < SparseDimensions; i++)
	{
		x(i) = 0.5*i+2;
		result(i) = x(i);
	}
	
	for (size_t i = 0; i < VectorNNZ; i++)
	{
		y(10*i+1) = 0.5*i;
		result(10*i+1) += y(10*i+1);
	}
	checkDenseExpressionEquality(x+y,result);
	checkDenseExpressionEquality(y+x,result);
}

BOOST_AUTO_TEST_CASE( Remora_Dense_Sparse_Vector_Binary_Multiply)
{
	vector<double> x(SparseDimensions); 
	compressed_vector<double> y(SparseDimensions); 
	compressed_vector<double> result(SparseDimensions);
	
	for (size_t i = 0; i < SparseDimensions; i++)
	{
		x(i) = 0.5*i+2;
	}
	
	for (size_t i = 0; i < VectorNNZ; i++)
	{
		y(10*i+1) = 0.5*i;
		result(10*i+1) = x(10*i+1)*y(10*i+1);
	}
	checkSparseExpressionEquality(x*y,result);
	checkSparseExpressionEquality(y*x,result);
}


BOOST_AUTO_TEST_SUITE_END()
