#define BOOST_TEST_MODULE Remora_GPU_vector_expression
#include <boost/test/unit_test.hpp>
#include <boost/test/floating_point_comparison.hpp>

#include <remora/vector_expression.hpp>
#include <remora/vector.hpp>
#include <remora/gpu/vector.hpp>
#include <remora/gpu/copy.hpp>

using namespace remora;

template<class Operation, class Result>
void checkDenseExpressionEquality(
	Operation op_gpu, Result const& result
){
	BOOST_REQUIRE_EQUAL(op_gpu.size(), result.size());
	
	//test copy to cpu, this tests the buffer
	vector<float> op = copy_to_cpu(op_gpu);
	for(std::size_t i = 0; i != op.size(); ++i){
		BOOST_CHECK_CLOSE(result(i), op(i),1.e-3);
	}
	
	//test iterators
	BOOST_REQUIRE_EQUAL(op_gpu.end() - op_gpu.begin(), op.size());
	vector<float, gpu_tag> opcopy_gpu(op.size());
	boost::compute::copy(op_gpu.begin(),op_gpu.end(),opcopy_gpu.begin());
	vector<float> opcopy = copy_to_cpu(opcopy_gpu);
	for(std::size_t i = 0; i != result.size(); ++i){
		BOOST_CHECK_CLOSE(result(i), opcopy(i),1.e-3);
	}
}

const std::size_t Dimensions = 1000;

/////////////////////////////////////////////////////////////
//////UNARY TRANSFORMATIONS///////
////////////////////////////////////////////////////////////
BOOST_AUTO_TEST_SUITE (Remora_vector_expression)

BOOST_AUTO_TEST_CASE( Remora_Vector_Unary_Minus )
{
	vector<float> x_cpu(Dimensions); 
	vector<float> result(Dimensions);
	
	for (size_t i = 0; i < Dimensions; i++)
	{
		x_cpu(i) = i-3.0;
		result(i)= -x_cpu(i);
	}
	vector<float, gpu_tag> x = copy_to_gpu(x_cpu);
	checkDenseExpressionEquality(-x,result);
}
BOOST_AUTO_TEST_CASE( Remora_Vector_Scalar_Add )
{
	vector<float> x_cpu(Dimensions); 
	vector<float> result(Dimensions);
	
	for (size_t i = 0; i < Dimensions; i++)
	{
		x_cpu(i) = i-3.0;
		result(i)= 5.0+x_cpu(i);
	}
	vector<float, gpu_tag> x = copy_to_gpu(x_cpu);
	checkDenseExpressionEquality(5.0 + x,result);
	checkDenseExpressionEquality(x + 5.0,result);
}
BOOST_AUTO_TEST_CASE( Remora_Vector_Scalar_Multiply )
{
	vector<float> x_cpu(Dimensions); 
	vector<float> result(Dimensions);
	
	for (size_t i = 0; i < Dimensions; i++)
	{
		x_cpu(i) = i-3.0;
		result(i)= 5.0*x_cpu(i);
	}
	vector<float, gpu_tag> x = copy_to_gpu(x_cpu);
	checkDenseExpressionEquality(5.0*x,result);
	checkDenseExpressionEquality(x*5.0,result);
}
BOOST_AUTO_TEST_CASE( Remora_Vector_Scalar_Div )
{
	vector<float> x_cpu(Dimensions); 
	vector<float> result(Dimensions);
	
	for (size_t i = 0; i < Dimensions; i++)
	{
		x_cpu(i) = 2*i+1.0;
		result(i)= x_cpu(i)/5.0;
	}
	vector<float, gpu_tag> x = copy_to_gpu(x_cpu);
	checkDenseExpressionEquality(x/5.0f,result);
}
BOOST_AUTO_TEST_CASE( Remora_Vector_Abs )
{
	vector<float> x_cpu(Dimensions); 
	vector<float> result(Dimensions);
	
	for (size_t i = 0; i < Dimensions; i++)
	{
		x_cpu(i) = 3.0-1;
		result(i)= std::abs(x_cpu(i));
	}
	vector<float, gpu_tag> x = copy_to_gpu(x_cpu);
	checkDenseExpressionEquality(abs(x),result);
}
BOOST_AUTO_TEST_CASE( Remora_Vector_Sqr )
{
	vector<float> x_cpu(Dimensions); 
	vector<float> result(Dimensions);
	
	for (size_t i = 0; i < Dimensions; i++)
	{
		x_cpu(i) = 3.0-i;
		result(i)= x_cpu(i)*x_cpu(i);
	}
	vector<float, gpu_tag> x = copy_to_gpu(x_cpu);
	checkDenseExpressionEquality(sqr(x),result);
}
BOOST_AUTO_TEST_CASE( Remora_Vector_Sqrt )
{
	vector<float> x_cpu(Dimensions); 
	vector<float> result(Dimensions);
	
	for (size_t i = 0; i < Dimensions; i++)
	{
		x_cpu(i) = i;
		result(i)= sqrt(x_cpu(i));
	}
	vector<float, gpu_tag> x = copy_to_gpu(x_cpu);
	checkDenseExpressionEquality(sqrt(x),result);
}
BOOST_AUTO_TEST_CASE( Remora_Vector_Cbrt )
{
	vector<float> x_cpu(Dimensions); 
	vector<float> result(Dimensions);
	
	for (size_t i = 0; i < Dimensions; i++)
	{
		x_cpu(i) = i;
		result(i)= cbrt(x_cpu(i));
	}
	vector<float, gpu_tag> x = copy_to_gpu(x_cpu);
	checkDenseExpressionEquality(cbrt(x),result);
}
BOOST_AUTO_TEST_CASE( Remora_Vector_Exp )
{
	vector<float> x_cpu(Dimensions); 
	vector<float> result(Dimensions);
	
	for (size_t i = 0; i < Dimensions; i++)
	{
		x_cpu(i) = 0.01*i;
		result(i)=std::exp(x_cpu(i));
	}
	vector<float, gpu_tag> x = copy_to_gpu(x_cpu);
	checkDenseExpressionEquality(exp(x),result);
}
BOOST_AUTO_TEST_CASE( Remora_Vector_Log )
{

	vector<float> x_cpu(Dimensions); 
	vector<float> result(Dimensions);
	
	for (size_t i = 0; i < Dimensions; i++)
	{
		x_cpu(i) = i+1;
		result(i)=std::log(x_cpu(i));
	}
	vector<float, gpu_tag> x = copy_to_gpu(x_cpu);
	checkDenseExpressionEquality(log(x),result);
}

BOOST_AUTO_TEST_CASE( Remora_Vector_sin )
{

	vector<float> x_cpu(Dimensions); 
	vector<float> result(Dimensions);
	
	for (size_t i = 0; i < Dimensions; i++)
	{
		x_cpu(i) = i+1;
		result(i)=std::sin(x_cpu(i));
	}
	vector<float, gpu_tag> x = copy_to_gpu(x_cpu);
	checkDenseExpressionEquality(sin(x),result);
}
BOOST_AUTO_TEST_CASE( Remora_Vector_cos )
{

	vector<float> x_cpu(Dimensions); 
	vector<float> result(Dimensions);
	
	for (size_t i = 0; i < Dimensions; i++)
	{
		x_cpu(i) = i+1;
		result(i)=std::cos(x_cpu(i));
	}
	vector<float, gpu_tag> x = copy_to_gpu(x_cpu);
	checkDenseExpressionEquality(cos(x),result);
}
BOOST_AUTO_TEST_CASE( Remora_Vector_tan )
{

	vector<float> x_cpu(Dimensions); 
	vector<float> result(Dimensions);
	
	for (size_t i = 0; i < Dimensions; i++)
	{
		x_cpu(i) = i+1;
		result(i)=std::tan(x_cpu(i));
	}
	vector<float, gpu_tag> x = copy_to_gpu(x_cpu);
	checkDenseExpressionEquality(tan(x),result);
}
BOOST_AUTO_TEST_CASE( Remora_Vector_asin )
{

	vector<float> x_cpu(Dimensions); 
	vector<float> result(Dimensions);
	
	for (size_t i = 0; i < Dimensions; i++)
	{
		x_cpu(i) = (2.0*(i+1))/Dimensions -1.0;
		result(i)=std::asin(x_cpu(i));
	}
	vector<float, gpu_tag> x = copy_to_gpu(x_cpu);
	checkDenseExpressionEquality(asin(x),result);
}
BOOST_AUTO_TEST_CASE( Remora_Vector_acos )
{

	vector<float> x_cpu(Dimensions); 
	vector<float> result(Dimensions);
	
	for (size_t i = 0; i < Dimensions; i++)
	{
		x_cpu(i) = (2.0*(i+1))/Dimensions -1.0;
		result(i)=std::acos(x_cpu(i));
	}
	vector<float, gpu_tag> x = copy_to_gpu(x_cpu);
	checkDenseExpressionEquality(acos(x),result);
}
BOOST_AUTO_TEST_CASE( Remora_Vector_atan ){
	vector<float> x_cpu(Dimensions); 
	vector<float> result(Dimensions);
	
	for (size_t i = 0; i < Dimensions; i++)
	{
		x_cpu(i) = (2.0*(i+1))/Dimensions -1.0;
		result(i)=std::atan(x_cpu(i));
	}
	vector<float, gpu_tag> x = copy_to_gpu(x_cpu);
	checkDenseExpressionEquality(atan(x),result);
}

BOOST_AUTO_TEST_CASE( Remora_Vector_erf ){
	vector<float> x_cpu(Dimensions); 
	vector<float> result(Dimensions);
	
	for (size_t i = 0; i < Dimensions; i++)
	{
		x_cpu(i) = (2.0*(i+1))/Dimensions -1.0;
		result(i)=std::erf(x_cpu(i));
	}
	vector<float, gpu_tag> x = copy_to_gpu(x_cpu);
	checkDenseExpressionEquality(erf(x),result);
}
BOOST_AUTO_TEST_CASE( Remora_Vector_erfc ){
	vector<float> x_cpu(Dimensions); 
	vector<float> result(Dimensions);
	
	for (size_t i = 0; i < Dimensions; i++)
	{
		x_cpu(i) = (2.0*(i+1))/Dimensions -1.0;
		result(i)=std::erfc(x_cpu(i));
	}
	vector<float, gpu_tag> x = copy_to_gpu(x_cpu);
	checkDenseExpressionEquality(erfc(x),result);
}

BOOST_AUTO_TEST_CASE( Remora_Vector_Tanh )
{
	vector<float> x_cpu(Dimensions); 
	vector<float> result(Dimensions);
	
	for (size_t i = 0; i < Dimensions; i++){
		x_cpu(i) = i;
		result(i)=std::tanh(x_cpu(i));
	}
	vector<float, gpu_tag> x = copy_to_gpu(x_cpu);
	checkDenseExpressionEquality(tanh(x),result);
}
BOOST_AUTO_TEST_CASE( Remora_Vector_Sigmoid )
{
	vector<float> x_cpu(Dimensions); 
	vector<float> result(Dimensions);
	
	for (size_t i = 0; i < Dimensions; i++){
		x_cpu(i) = i;
		result(i) = 1.0/(1.0+std::exp(-x_cpu(i)));
	}
	vector<float, gpu_tag> x = copy_to_gpu(x_cpu);
	checkDenseExpressionEquality(sigmoid(x),result);
}
BOOST_AUTO_TEST_CASE( Remora_Vector_SoftPlus )
{
	vector<float> x_cpu(Dimensions); 
	vector<float> result(Dimensions);
	
	for (size_t i = 0; i < Dimensions; i++)
	{
		x_cpu(i) = 0.02*i;
		result(i) =std::log(1+std::exp(x_cpu(i)));
	}
	vector<float, gpu_tag> x = copy_to_gpu(x_cpu);
	checkDenseExpressionEquality(softPlus(x),result);
}
BOOST_AUTO_TEST_CASE( Remora_Vector_Pow )
{
	vector<float> x_cpu(Dimensions); 
	vector<float> result(Dimensions);
	
	for (size_t i = 0; i < Dimensions; i++)
	{
		x_cpu(i) = i+1.0;
		result(i)= std::pow(x_cpu(i),3.2);
	}
	vector<float, gpu_tag> x = copy_to_gpu(x_cpu);
	checkDenseExpressionEquality(pow(x,3.2),result);
}

/////////////////////////////////////////////////////
///////BINARY OPERATIONS//////////
/////////////////////////////////////////////////////

BOOST_AUTO_TEST_CASE( Remora_Vector_Binary_Plus)
{
	vector<float> x_cpu(Dimensions); 
	vector<float> y_cpu(Dimensions); 
	vector<float> result(Dimensions);
	
	for (size_t i = 0; i < Dimensions; i++)
	{
		x_cpu(i) = i;
		y_cpu(i) = i+Dimensions;
		result(i) = x_cpu(i)+y_cpu(i);
	}
	vector<float, gpu_tag> x = copy_to_gpu(x_cpu);
	vector<float, gpu_tag> y = copy_to_gpu(y_cpu);
	checkDenseExpressionEquality(x+y,result);
}
BOOST_AUTO_TEST_CASE( Remora_Vector_Binary_Minus)
{
	vector<float> x_cpu(Dimensions); 
	vector<float> y_cpu(Dimensions); 
	vector<float> result(Dimensions);
	
	for (size_t i = 0; i < Dimensions; i++)
	{
		x_cpu(i) = i;
		y_cpu(i) = -3.0*i+Dimensions;
		result(i) = x_cpu(i)-y_cpu(i);
	}
	vector<float, gpu_tag> x = copy_to_gpu(x_cpu);
	vector<float, gpu_tag> y = copy_to_gpu(y_cpu);
	checkDenseExpressionEquality(x-y,result);
}

BOOST_AUTO_TEST_CASE( Remora_Vector_Binary_Multiply)
{
	vector<float> x_cpu(Dimensions); 
	vector<float> y_cpu(Dimensions); 
	vector<float> result(Dimensions);
	
	for (size_t i = 0; i < Dimensions; i++)
	{
		x_cpu(i) = i;
		y_cpu(i) = -3.0*i+Dimensions;
		result(i) = x_cpu(i)*y_cpu(i);
	}
	vector<float, gpu_tag> x = copy_to_gpu(x_cpu);
	vector<float, gpu_tag> y = copy_to_gpu(y_cpu);
	checkDenseExpressionEquality(x*y,result);
	checkDenseExpressionEquality(element_prod(x,y),result);
}

BOOST_AUTO_TEST_CASE( Remora_Vector_Binary_Div)
{
	vector<float> x_cpu(Dimensions); 
	vector<float> y_cpu(Dimensions); 
	vector<float> result(Dimensions);
	
	for (size_t i = 0; i < Dimensions; i++)
	{
		x_cpu(i) = 3.0*i+3.0;
		y_cpu(i) = i+1;
		result(i) = x_cpu(i)/y_cpu(i);
	}
	vector<float, gpu_tag> x = copy_to_gpu(x_cpu);
	vector<float, gpu_tag> y = copy_to_gpu(y_cpu);
	checkDenseExpressionEquality(x/y,result);
	checkDenseExpressionEquality(element_div(x,y),result);
}

BOOST_AUTO_TEST_CASE( Remora_Vector_Safe_Div )
{
	vector<float> x_cpu(Dimensions); 
	vector<float> y_cpu(Dimensions); 
	vector<float> result(Dimensions);
	
	for (size_t i = 0; i < Dimensions; i++)
	{
		x_cpu(i) = i;
		y_cpu(i) = i % 3;
		result(i) = (i % 3 == 0)? 2.0: x_cpu(i)/y_cpu(i);
	}
	vector<float, gpu_tag> x = copy_to_gpu(x_cpu);
	vector<float, gpu_tag> y = copy_to_gpu(y_cpu);
	checkDenseExpressionEquality(safe_div(x,y,2.0),result);
}

/////////////////////////////////////////////////////
///////////Vector Reductions///////////
/////////////////////////////////////////////////////

BOOST_AUTO_TEST_CASE( Remora_Vector_Max )
{
	vector<float> x_cpu(Dimensions); 
	float result = 1;
	
	for (size_t i = 0; i < Dimensions; i++){
		x_cpu(i) = exp(-(i-5.0)*(i-5.0));//max at i = 5
		result = std::max(result,x_cpu(i));
	}
	vector<float, gpu_tag> x = copy_to_gpu(x_cpu);
	BOOST_CHECK_CLOSE(max(x),result,1.e-10);
}
BOOST_AUTO_TEST_CASE( Remora_Vector_Min )
{
	vector<float> x_cpu(Dimensions); 
	float result = -1;
	
	for (size_t i = 0; i < Dimensions; i++){
		x_cpu(i) = -std::exp(-(i-5.0)*(i-5.0));//min at i = 5
		result = std::min(result,x_cpu(i));
	}
	vector<float, gpu_tag> x = copy_to_gpu(x_cpu);
	BOOST_CHECK_CLOSE(min(x),result,1.e-10);
}

BOOST_AUTO_TEST_CASE( Remora_Vector_Arg_Max )
{
	vector<float> x_cpu(Dimensions); 
	unsigned int result = 5;
	
	for (size_t i = 0; i < Dimensions; i++){
		x_cpu(i) = exp(-(i-5.0)*(i-5.0));//max at i = 5
	}
	vector<float, gpu_tag> x = copy_to_gpu(x_cpu);
	BOOST_CHECK_EQUAL(arg_max(x),result);
}

BOOST_AUTO_TEST_CASE( Remora_Vector_Arg_Min )
{
	vector<float> x_cpu(Dimensions); 
	unsigned int result = 5;
	
	for (size_t i = 0; i < Dimensions; i++){
		x_cpu(i) = -exp(-(i-5.0)*(i-5.0));//min at i = 5
	}
	vector<float, gpu_tag> x = copy_to_gpu(x_cpu);
	BOOST_CHECK_EQUAL(arg_min(x),result);
}

BOOST_AUTO_TEST_CASE( Remora_Vector_Sum )
{
	vector<float> x_cpu(Dimensions); 
	float result = 0;
	
	for (size_t i = 0; i < Dimensions; i++){
		x_cpu(i) = 2*i-5.0;
		result +=x_cpu(i);
	}
	vector<float, gpu_tag> x = copy_to_gpu(x_cpu);
	BOOST_CHECK_CLOSE(sum(x),result,1.e-10);
}

BOOST_AUTO_TEST_CASE( Remora_Vector_norm_1 )
{
	vector<float> x_cpu(Dimensions); 
	float result = 0;
	
	for (size_t i = 0; i < Dimensions; i++){
		x_cpu(i) = 2*i-5.0;
		result +=std::abs(x_cpu(i));
	}
	vector<float, gpu_tag> x = copy_to_gpu(x_cpu);
	BOOST_CHECK_CLOSE(norm_1(x),result,1.e-10);
}

BOOST_AUTO_TEST_CASE( Remora_Vector_norm_sqr )
{
	vector<float> x_cpu(Dimensions); 
	float result = 0;
	
	for (size_t i = 0; i < Dimensions; i++){
		x_cpu(i) = 0.1*(2*i-5.0);
		result +=x_cpu(i)*x_cpu(i);
	}
	vector<float, gpu_tag> x = copy_to_gpu(x_cpu);
	BOOST_CHECK_CLOSE(norm_sqr(x),result,1.e-10);
}
BOOST_AUTO_TEST_CASE( Remora_Vector_norm_2 )
{
	vector<float> x_cpu(Dimensions); 
	float result = 0;
	for (size_t i = 0; i < Dimensions; i++){
		x_cpu(i) = 0.1*(2*i-5.0);
		result += x_cpu(i)*x_cpu(i);
	}
	result = std::sqrt(result);
	vector<float, gpu_tag> x = copy_to_gpu(x_cpu);
	BOOST_CHECK_CLOSE(norm_2(x),result,1.e-10);
}
BOOST_AUTO_TEST_CASE( Remora_Vector_norm_inf )
{
	vector<float> x_cpu(Dimensions); 
	for (size_t i = 0; i < Dimensions; i++){
		x_cpu(i) = exp(-(i-5.0)*(i-5.0));
	}
	x_cpu(8)=-2;
	vector<float, gpu_tag> x = copy_to_gpu(x_cpu);
	BOOST_CHECK_EQUAL(norm_inf(x),2.0);
}
BOOST_AUTO_TEST_CASE( Remora_Vector_index_norm_inf )
{
	vector<float> x_cpu(Dimensions); 
	
	for (size_t i = 0; i < Dimensions; i++){
		x_cpu(i) = exp(-(i-5.0)*(i-5.0));
	}
	x_cpu(8)=-2;
	vector<float, gpu_tag> x = copy_to_gpu(x_cpu);
	BOOST_CHECK_EQUAL(index_norm_inf(x),8);
}

BOOST_AUTO_TEST_CASE( Remora_Vector_inner_prod )
{
	vector<float> x_cpu(Dimensions); 
	vector<float> y_cpu(Dimensions); 
	double result = 0;
	for (size_t i = 0; i < Dimensions; i++){
		x_cpu(i) = 0.1*i+3.0;
		y_cpu(i) = i+1.0;
		result += x_cpu(i) * y_cpu(i);
	}
	vector<float, gpu_tag> x = copy_to_gpu(x_cpu);
	vector<float, gpu_tag> y = copy_to_gpu(y_cpu);
	BOOST_CHECK_CLOSE(inner_prod(x,y),result,1.e-4);
}


BOOST_AUTO_TEST_SUITE_END()
