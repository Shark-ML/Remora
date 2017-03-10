#define BOOST_TEST_MODULE Remora_GPU_Matrix_Expression
#define BOOST_COMPUTE_DEBUG_KERNEL_COMPILATION
#include <boost/test/unit_test.hpp>
#include <boost/test/floating_point_comparison.hpp>

#include <remora/matrix_expression.hpp>
#include <remora/vector.hpp>
#include <remora/matrix.hpp>
#include <remora/gpu/vector.hpp>
#include <remora/gpu/matrix.hpp>
#include <remora/gpu/copy.hpp>

using namespace remora;


template<class Operation, class Result>
void checkDenseExpressionEquality(
	matrix_expression<Operation, gpu_tag> const& op_gpu, Result const& result
){
	BOOST_REQUIRE_EQUAL(op_gpu().size1(), result.size1());
	BOOST_REQUIRE_EQUAL(op_gpu().size2(), result.size2());
	
	//check that matrix assignment using op() works(implicit test)
	matrix<float> op = copy_to_cpu(op_gpu());
	for(std::size_t i = 0; i != op.size1(); ++i){
		for(std::size_t j = 0; j != op.size2(); ++j){
			BOOST_CHECK_CLOSE(result(i,j), op(i,j),1.e-4);
		}
	}
	gpu::vector<float> op_row_gpu(op_gpu().size2());
	gpu::vector<float> op_col_gpu(op_gpu().size1());
	vector<float> op_row(op_gpu().size2());
	vector<float> op_col(op_gpu().size2());
	//~ //check that row iterator work
	//~ for(std::size_t i = 0; i != op.size1(); ++i){
		
		//~ boost::compute::copy(op_gpu().row_begin(i), op_gpu().row_end(i), op_row_gpu.begin());
		//~ boost::compute::copy(op_row_gpu.begin(), op_row_gpu.end(), op_row.begin());
		
		//~ for(std::size_t j = 0; j != op.size2(); ++j){
			//~ BOOST_CHECK_CLOSE(result(i,j), op_row(j),1.e-8);
		//~ }
	//~ }
	
	//~ //check that column iterator work
	//~ for(std::size_t j = 0; j != op.size2(); ++j){
		//~ boost::compute::copy(op_gpu().column_begin(j), op_gpu().column_end(j), op_col_gpu.begin());
		//~ boost::compute::copy(op_col_gpu.begin(), op_col_gpu.end(), op_col.begin());
		//~ for(std::size_t i = 0; i != op.size1(); ++i){
			//~ BOOST_CHECK_CLOSE(result(i,j), op_col(i),1.e-8);
		//~ }
	//~ }
}


template<class Operation, class Result>
void checkDenseExpressionEquality(
	vector_expression<Operation, gpu_tag> const& op_gpu, Result const& result
){
	BOOST_REQUIRE_EQUAL(op_gpu().size(), result.size());
	
	vector<float> op = copy_to_cpu(op_gpu());
	for(std::size_t i = 0; i != op.size(); ++i){
		BOOST_CHECK_CLOSE(result(i), op(i),1.e-8);
	}
}



std::size_t Dimension1 = 50;
std::size_t Dimension2 = 100;



BOOST_AUTO_TEST_SUITE (Remora_GPU_matrix_expression)

/////////////////////////////////////////////////////////////
//////Vector->Matrix expansions///////
////////////////////////////////////////////////////////////

BOOST_AUTO_TEST_CASE( Remora_matrix_Outer_Prod ){
	vector<float> x_cpu(Dimension1); 
	vector<float> y_cpu(Dimension2); 
	matrix<float> result(Dimension1, Dimension2);
	
	for (size_t i = 0; i < Dimension1; i++)
		x_cpu(i) = i-3.0;
	for (size_t j = 0; j < Dimension2; j++)
		y_cpu(j) = 2*j;
	
	for (size_t i = 0; i < Dimension1; i++){
		for (size_t j = 0; j < Dimension2; j++){
			result(i,j)= x_cpu(i)*y_cpu(j);
		}
	}
	gpu::vector<float> x = gpu::copy_to_gpu(x_cpu);
	gpu::vector<float> y = gpu::copy_to_gpu(y_cpu);
	checkDenseExpressionEquality(outer_prod(x,y),result);
}

BOOST_AUTO_TEST_CASE( Remora_matrix_Vector_Repeater){
	vector<float> x_cpu(Dimension2); 
	matrix<float> result(Dimension1, Dimension2);
	
	for (size_t i = 0; i < Dimension2; i++)
		x_cpu(i) = i-3.0;
	
	for (size_t i = 0; i < Dimension1; i++){
		for (size_t j = 0; j < Dimension2; j++){
			result(i,j)= x_cpu(j);
		}
	}
	gpu::vector<float> x = gpu::copy_to_gpu(x_cpu);
	checkDenseExpressionEquality(repeat(x,Dimension1),result);
}

/////////////////////////////////////////////////////////////
//////UNARY TRANSFORMATIONS///////
////////////////////////////////////////////////////////////
BOOST_AUTO_TEST_CASE( Remora_matrix_Unary_Minus )
{
	matrix<float> x_cpu(Dimension1, Dimension2); 
	matrix<float> result(Dimension1, Dimension2);
	
	for (size_t i = 0; i < Dimension1; i++){
		for (size_t j = 0; j < Dimension2; j++){
			x_cpu(i,j) = -3.0+i+j;
			result(i,j)= -x_cpu(i,j);
		}
	}
	gpu::matrix<float> x = gpu::copy_to_gpu(x_cpu);
	checkDenseExpressionEquality(-x,result);
}
BOOST_AUTO_TEST_CASE( Remora_matrix_Scalar_Multiply )
{
	matrix<float> x_cpu(Dimension1, Dimension2); 
	matrix<float> result(Dimension1, Dimension2);
	
	for (size_t i = 0; i < Dimension1; i++){
		for (size_t j = 0; j < Dimension2; j++){
			x_cpu(i,j) = -3.0+i+j;
			result(i,j)= 5.0* x_cpu(i,j);
		}
	}
	gpu::matrix<float> x = gpu::copy_to_gpu(x_cpu);
	checkDenseExpressionEquality(5.0*x,result);
	checkDenseExpressionEquality(x*5.0,result);
}
BOOST_AUTO_TEST_CASE( Remora_matrix_Scalar_Add )
{
	matrix<float> x_cpu(Dimension1, Dimension2); 
	matrix<float> result(Dimension1, Dimension2);
	
	for (size_t i = 0; i < Dimension1; i++){
		for (size_t j = 0; j < Dimension2; j++){
			x_cpu(i,j) = i-3.0+j;
			result(i,j)= 5.0 + x_cpu(i,j);
		}
	}
	gpu::matrix<float> x = gpu::copy_to_gpu(x_cpu);
	checkDenseExpressionEquality(5.0+x,result);
	checkDenseExpressionEquality(x+5.0,result);
}
BOOST_AUTO_TEST_CASE( Remora_matrix_Scalar_Subtract )
{
	matrix<float> x_cpu(Dimension1, Dimension2); 
	matrix<float> result1(Dimension1, Dimension2);
	matrix<float> result2(Dimension1, Dimension2);
	
	for (size_t i = 0; i < Dimension1; i++){
		for (size_t j = 0; j < Dimension2; j++){
			x_cpu(i,j) = i-3.0+j;
			result1(i,j)= 5.0 - x_cpu(i,j);
			result2(i,j)= x_cpu(i,j) - 5.0;
		}
	}
	gpu::matrix<float> x = gpu::copy_to_gpu(x_cpu);
	checkDenseExpressionEquality(5.0- x,result1);
	checkDenseExpressionEquality(x - 5.0,result2);
}
BOOST_AUTO_TEST_CASE( Remora_matrix_Scalar_Div )
{
	matrix<float> x_cpu(Dimension1, Dimension2); 
	matrix<float> result(Dimension1, Dimension2);
	
	for (size_t i = 0; i < Dimension1; i++){
		for (size_t j = 0; j < Dimension2; j++){
			x_cpu(i,j) = i-3.0+j;
			result(i,j)= x_cpu(i,j)/5.0;
		}
	}
	gpu::matrix<float> x = gpu::copy_to_gpu(x_cpu);
	checkDenseExpressionEquality(x/5.0f,result);
}
BOOST_AUTO_TEST_CASE( Remora_matrix_Scalar_elem_inv)
{
	matrix<float> x_cpu(Dimension1, Dimension2); 
	matrix<float> result(Dimension1, Dimension2);
	
	for (size_t i = 0; i < Dimension1; i++){
		for (size_t j = 0; j < Dimension2; j++){
			x_cpu(i,j) = i+j+3.0;
			result(i,j)= 1.0/x_cpu(i,j);
		}
	}
	gpu::matrix<float> x = gpu::copy_to_gpu(x_cpu);
	checkDenseExpressionEquality(elem_inv(x),result);
}
BOOST_AUTO_TEST_CASE( Remora_matrix_Abs )
{
	matrix<float> x_cpu(Dimension1, Dimension2); 
	matrix<float> result(Dimension1, Dimension2);
	
	for (size_t i = 0; i < Dimension1; i++){
		for (size_t j = 0; j < Dimension2; j++){
			x_cpu(i,j) = -3.0+i-j;
			result(i,j)= std::abs(x_cpu(i,j));
		}
	}
	gpu::matrix<float> x = gpu::copy_to_gpu(x_cpu);
	checkDenseExpressionEquality(abs(x),result);
}
BOOST_AUTO_TEST_CASE( Remora_matrix_Sqr )
{
	matrix<float> x_cpu(Dimension1, Dimension2); 
	matrix<float> result(Dimension1, Dimension2);
	for (size_t i = 0; i < Dimension1; i++){
		for (size_t j = 0; j < Dimension2; j++){
			x_cpu(i,j) = -3.0+i-j;
			result(i,j)= x_cpu(i,j) * x_cpu(i,j);
		}
	}
	gpu::matrix<float> x = gpu::copy_to_gpu(x_cpu);
	checkDenseExpressionEquality(sqr(x),result);
}
BOOST_AUTO_TEST_CASE( Remora_matrix_Sqrt )
{
	matrix<float> x_cpu(Dimension1, Dimension2); 
	matrix<float> result(Dimension1, Dimension2);
	
	for (size_t i = 0; i < Dimension1; i++){
		for (size_t j = 0; j < Dimension2; j++){
			x_cpu(i,j) = i+j+3.0;
			result(i,j)= std::sqrt(x_cpu(i,j));
		}
	}
	gpu::matrix<float> x = gpu::copy_to_gpu(x_cpu);
	checkDenseExpressionEquality(sqrt(x),result);
}
BOOST_AUTO_TEST_CASE( Remora_matrix_Cbrt)
{
	matrix<float> x_cpu(Dimension1, Dimension2); 
	matrix<float> result(Dimension1, Dimension2);
	
	for (size_t i = 0; i < Dimension1; i++){
		for (size_t j = 0; j < Dimension2; j++){
			x_cpu(i,j) = i+j+3.0;
			result(i,j)= std::cbrt(x_cpu(i,j));
		}
	}
	gpu::matrix<float> x = gpu::copy_to_gpu(x_cpu);
	checkDenseExpressionEquality(cbrt(x),result);
}

BOOST_AUTO_TEST_CASE( Remora_matrix_Exp )
{
	matrix<float> x_cpu(Dimension1, Dimension2); 
	matrix<float> result(Dimension1, Dimension2);
	
	for (size_t i = 0; i < Dimension1; i++){
		for (size_t j = 0; j < Dimension2; j++){
			x_cpu(i,j) = -3.0+i-j;
			result(i,j)= std::exp(x_cpu(i,j));
		}
	}
	gpu::matrix<float> x = gpu::copy_to_gpu(x_cpu);
	checkDenseExpressionEquality(exp(x),result);
}
BOOST_AUTO_TEST_CASE( Remora_matrix_Log )
{

	matrix<float> x_cpu(Dimension1, Dimension2); 
	matrix<float> result(Dimension1, Dimension2);
	
	for (size_t i = 0; i < Dimension1; i++){
		for (size_t j = 0; j < Dimension2; j++){
			x_cpu(i,j) = i+j+3.0;
			result(i,j)= std::log(x_cpu(i,j));
		}
	}
	gpu::matrix<float> x = gpu::copy_to_gpu(x_cpu);
	checkDenseExpressionEquality(log(x),result);
}
BOOST_AUTO_TEST_CASE( Remora_matrix_sin )
{
	matrix<float> x_cpu(Dimension1, Dimension2); 
	matrix<float> result(Dimension1, Dimension2);
	
	for (size_t i = 0; i < Dimension1; i++){
		for (size_t j = 0; j < Dimension2; j++){
			x_cpu(i,j) = -3.0+double(i)/Dimension1 - double(j)/Dimension2;
			result(i,j)= std::sin(x_cpu(i,j));
		}
	}
	gpu::matrix<float> x = gpu::copy_to_gpu(x_cpu);
	checkDenseExpressionEquality(sin(x),result);
}
BOOST_AUTO_TEST_CASE( Remora_matrix_cos )
{
	matrix<float> x_cpu(Dimension1, Dimension2); 
	matrix<float> result(Dimension1, Dimension2);
	
	for (size_t i = 0; i < Dimension1; i++){
		for (size_t j = 0; j < Dimension2; j++){
			x_cpu(i,j) = -3.0+double(i)/Dimension1 - double(j)/Dimension2;
			result(i,j)= std::cos(x_cpu(i,j));
		}
	}
	gpu::matrix<float> x = gpu::copy_to_gpu(x_cpu);
	checkDenseExpressionEquality(cos(x),result);
}
BOOST_AUTO_TEST_CASE( Remora_matrix_tan )
{
	matrix<float> x_cpu(Dimension1, Dimension2); 
	matrix<float> result(Dimension1, Dimension2);
	
	for (size_t i = 0; i < Dimension1; i++){
		for (size_t j = 0; j < Dimension2; j++){
			x_cpu(i,j) = -3.0+double(i)/Dimension1 - double(j)/Dimension2;
			result(i,j)= std::tan(x_cpu(i,j));
		}
	}
	gpu::matrix<float> x = gpu::copy_to_gpu(x_cpu);
	checkDenseExpressionEquality(tan(x),result);
}

BOOST_AUTO_TEST_CASE( Remora_matrix_asin )
{
	matrix<float> x_cpu(Dimension1, Dimension2); 
	matrix<float> result(Dimension1, Dimension2);
	
	for (size_t i = 0; i < Dimension1; i++){
		for (size_t j = 0; j < Dimension2; j++){
			x_cpu(i,j) = double(i)/Dimension1 - double(j)/Dimension2;
			result(i,j)= std::asin(x_cpu(i,j));
		}
	}
	gpu::matrix<float> x = gpu::copy_to_gpu(x_cpu);
	checkDenseExpressionEquality(asin(x),result);
}
BOOST_AUTO_TEST_CASE( Remora_matrix_acos )
{
	matrix<float> x_cpu(Dimension1, Dimension2); 
	matrix<float> result(Dimension1, Dimension2);
	
	for (size_t i = 0; i < Dimension1; i++){
		for (size_t j = 0; j < Dimension2; j++){
			x_cpu(i,j) = double(i)/Dimension1 - double(j)/Dimension2;
			result(i,j)= std::acos(x_cpu(i,j));
		}
	}
	gpu::matrix<float> x = gpu::copy_to_gpu(x_cpu);
	checkDenseExpressionEquality(acos(x),result);
}
BOOST_AUTO_TEST_CASE( Remora_matrix_atan )
{
	matrix<float> x_cpu(Dimension1, Dimension2); 
	matrix<float> result(Dimension1, Dimension2);
	
	for (size_t i = 0; i < Dimension1; i++){
		for (size_t j = 0; j < Dimension2; j++){
			x_cpu(i,j) = -3.0+double(i)/Dimension1 - double(j)/Dimension2;
			result(i,j)= std::atan(x_cpu(i,j));
		}
	}
	gpu::matrix<float> x = gpu::copy_to_gpu(x_cpu);
	checkDenseExpressionEquality(atan(x),result);
}

BOOST_AUTO_TEST_CASE( Remora_matrix_erf )
{
	matrix<float> x_cpu(Dimension1, Dimension2); 
	matrix<float> result(Dimension1, Dimension2);
	
	for (size_t i = 0; i < Dimension1; i++){
		for (size_t j = 0; j < Dimension2; j++){
			x_cpu(i,j) = -3.0+double(i)/Dimension1 - double(j)/Dimension2;
			result(i,j)= std::erf(x_cpu(i,j));
		}
	}
	gpu::matrix<float> x = gpu::copy_to_gpu(x_cpu);
	checkDenseExpressionEquality(erf(x),result);
}
BOOST_AUTO_TEST_CASE( Remora_matrix_erfc )
{
	matrix<float> x_cpu(Dimension1, Dimension2); 
	matrix<float> result(Dimension1, Dimension2);
	
	for (size_t i = 0; i < Dimension1; i++){
		for (size_t j = 0; j < Dimension2; j++){
			x_cpu(i,j) = -3.0+double(i)/Dimension1 - double(j)/Dimension2;
			result(i,j)= std::erfc(x_cpu(i,j));
		}
	}
	gpu::matrix<float> x = gpu::copy_to_gpu(x_cpu);
	checkDenseExpressionEquality(erfc(x),result);
}

BOOST_AUTO_TEST_CASE( Remora_matrix_Tanh )
{
	matrix<float> x_cpu(Dimension1, Dimension2); 
	matrix<float> result(Dimension1, Dimension2);
	
	for (size_t i = 0; i < Dimension1; i++){
		for (size_t j = 0; j < Dimension2; j++){
			x_cpu(i,j) = -3.0+i-j;
			result(i,j)= std::tanh(x_cpu(i,j));
		}
	}
	gpu::matrix<float> x = gpu::copy_to_gpu(x_cpu);
	checkDenseExpressionEquality(tanh(x),result);
}
BOOST_AUTO_TEST_CASE( Remora_matrix_Sigmoid )
{
	matrix<float> x_cpu(Dimension1, Dimension2); 
	matrix<float> result(Dimension1, Dimension2);
	
	for (size_t i = 0; i < Dimension1; i++){
		for (size_t j = 0; j < Dimension2; j++){
			x_cpu(i,j) = 0.01*(-3.0+i-j);
			result(i,j)= 1.0/(1.0+std::exp(-x_cpu(i,j)));
		}
	}
	gpu::matrix<float> x = gpu::copy_to_gpu(x_cpu);
	checkDenseExpressionEquality(sigmoid(x),result);
}
BOOST_AUTO_TEST_CASE( Remora_matrix_SoftPlus )
{
	matrix<float> x_cpu(Dimension1, Dimension2); 
	matrix<float> result(Dimension1, Dimension2);
	
	for (size_t i = 0; i < Dimension1; i++){
		for (size_t j = 0; j < Dimension2; j++){
			x_cpu(i,j) = -3.0+i-j;
			result(i,j)= std::log(1+std::exp(x_cpu(i,j)));
		}
	}
	gpu::matrix<float> x = gpu::copy_to_gpu(x_cpu);
	checkDenseExpressionEquality(softPlus(x),result);
}
BOOST_AUTO_TEST_CASE( Remora_matrix_Pow )
{
	matrix<float> x_cpu(Dimension1, Dimension2); 
	matrix<float> result(Dimension1, Dimension2);
	
	for (size_t i = 0; i < Dimension1; i++){
		for (size_t j = 0; j < Dimension2; j++){
			x_cpu(i,j) = 0.001*(i+j+2.0);;
			result(i,j)= std::pow(x_cpu(i,j),3.2);
		}
	}
	gpu::matrix<float> x = gpu::copy_to_gpu(x_cpu);
	checkDenseExpressionEquality(pow(x,3.2),result);
}

BOOST_AUTO_TEST_CASE( Remora_matrix_Unary_Min)
{
	matrix<float> x_cpu(Dimension1, Dimension2); 
	matrix<float> result(Dimension1, Dimension2);
	
	for (size_t i = 0; i < Dimension1; i++){
		for (size_t j = 0; j < Dimension2; j++){
			x_cpu(i,j) = i-1.0*j;
			result(i,j)= std::min(x_cpu(i,j),5.0f);
		}
	}
	gpu::matrix<float> x = gpu::copy_to_gpu(x_cpu);
	checkDenseExpressionEquality(min(x,5.0f),result);
	checkDenseExpressionEquality(min(5.0f,x),result);
}
BOOST_AUTO_TEST_CASE( Remora_matrix_Unary_Max)
{
	matrix<float> x_cpu(Dimension1, Dimension2); 
	matrix<float> result(Dimension1, Dimension2);
	
	for (size_t i = 0; i < Dimension1; i++){
		for (size_t j = 0; j < Dimension2; j++){
			x_cpu(i,j) = i-1.0*j;
			result(i,j)= std::max(x_cpu(i,j),5.0f);
		}
	}
	gpu::matrix<float> x = gpu::copy_to_gpu(x_cpu);
	checkDenseExpressionEquality(max(x,5.0f),result);
	checkDenseExpressionEquality(max(5.0f,x),result);
}

/////////////////////////////////////////////////////
///////BINARY OPERATIONS//////////
/////////////////////////////////////////////////////

BOOST_AUTO_TEST_CASE( Remora_matrix_Binary_Plus)
{
	matrix<float> x_cpu(Dimension1, Dimension2); 
	matrix<float> y_cpu(Dimension1, Dimension2); 
	matrix<float> result(Dimension1, Dimension2);
	
	for (size_t i = 0; i < Dimension1; i++){
		for (size_t j = 0; j < Dimension2; j++){
			x_cpu(i,j) = -3.0+i-j;
			y_cpu(i,j) = i+j+Dimension1;
			result(i,j)= x_cpu(i,j)+y_cpu(i,j);
		}
	}
	gpu::matrix<float> x = gpu::copy_to_gpu(x_cpu);
	gpu::matrix<float> y = gpu::copy_to_gpu(y_cpu);
	checkDenseExpressionEquality(x+y,result);
}
BOOST_AUTO_TEST_CASE( Remora_matrix_Binary_Minus)
{
	matrix<float> x_cpu(Dimension1, Dimension2); 
	matrix<float> y_cpu(Dimension1, Dimension2); 
	matrix<float> result(Dimension1, Dimension2);
	
	for (size_t i = 0; i < Dimension1; i++){
		for (size_t j = 0; j < Dimension2; j++){
			x_cpu(i,j) = -3.0+i-j;
			y_cpu(i,j) = i+j+Dimension1;
			result(i,j)= x_cpu(i,j)-y_cpu(i,j);
		}
	}
	gpu::matrix<float> x = gpu::copy_to_gpu(x_cpu);
	gpu::matrix<float> y = gpu::copy_to_gpu(y_cpu);
	checkDenseExpressionEquality(x-y,result);
}

BOOST_AUTO_TEST_CASE( Remora_matrix_Binary_Multiply)
{
	matrix<float> x_cpu(Dimension1, Dimension2); 
	matrix<float> y_cpu(Dimension1, Dimension2); 
	matrix<float> result(Dimension1, Dimension2);
	
	for (size_t i = 0; i < Dimension1; i++){
		for (size_t j = 0; j < Dimension2; j++){
			x_cpu(i,j) = -3.0+i-j;
			y_cpu(i,j) = i+j+Dimension1;
			result(i,j)= x_cpu(i,j)*y_cpu(i,j);
		}
	}
	gpu::matrix<float> x = gpu::copy_to_gpu(x_cpu);
	gpu::matrix<float> y = gpu::copy_to_gpu(y_cpu);
	checkDenseExpressionEquality(x*y,result);
	checkDenseExpressionEquality(element_prod(x,y),result);
}

BOOST_AUTO_TEST_CASE( Remora_matrix_Binary_Div)
{
	matrix<float> x_cpu(Dimension1, Dimension2); 
	matrix<float> y_cpu(Dimension1, Dimension2); 
	matrix<float> result(Dimension1, Dimension2);
	
	for (size_t i = 0; i < Dimension1; i++){
		for (size_t j = 0; j < Dimension2; j++){
			x_cpu(i,j) = -3.0+i-j;
			y_cpu(i,j) = i+j+1;
			result(i,j)= x_cpu(i,j)/y_cpu(i,j);
		}
	}
	gpu::matrix<float> x = gpu::copy_to_gpu(x_cpu);
	gpu::matrix<float> y = gpu::copy_to_gpu(y_cpu);
	checkDenseExpressionEquality(x/y,result);
	checkDenseExpressionEquality(element_div(x,y),result);
}

BOOST_AUTO_TEST_CASE( Remora_matrix_Safe_Div )
{
	matrix<float> x_cpu(Dimension1, Dimension2); 
	matrix<float> y_cpu(Dimension1, Dimension2); 
	matrix<float> result(Dimension1, Dimension2);
	for (size_t i = 0; i < Dimension1; i++){
		for (size_t j = 0; j < Dimension2; j++){
			x_cpu(i,j) = i+j;
			y_cpu(i,j) = (i+j)%3;
			result(i,j)= ((i+j) % 3 == 0)? 2.0: x_cpu(i,j)/y_cpu(i,j);
		}
	}
	gpu::matrix<float> x = gpu::copy_to_gpu(x_cpu);
	gpu::matrix<float> y = gpu::copy_to_gpu(y_cpu);
	checkDenseExpressionEquality(safe_div(x,y,2.0),result);
}

BOOST_AUTO_TEST_CASE( Remora_matrix_Binary_Pow)
{
	matrix<float> x_cpu(Dimension1, Dimension2); 
	matrix<float> y_cpu(Dimension1, Dimension2); 
	matrix<float> result(Dimension1, Dimension2);
	
	for (size_t i = 0; i < Dimension1; i++){
		for (size_t j = 0; j < Dimension2; j++){
			x_cpu(i,j) = 0.1*(i+j+1);
			y_cpu(i,j) = 0.0001*(i+j-3);
			result(i,j)= std::pow(x_cpu(i,j),y_cpu(i,j));
		}
	}
	gpu::matrix<float> x = gpu::copy_to_gpu(x_cpu);
	gpu::matrix<float> y = gpu::copy_to_gpu(y_cpu);
	checkDenseExpressionEquality(pow(x,y),result);
}

BOOST_AUTO_TEST_CASE( Remora_matrix_Binary_Max)
{
	matrix<float> x_cpu(Dimension1, Dimension2); 
	matrix<float> y_cpu(Dimension1, Dimension2); 
	matrix<float> result(Dimension1, Dimension2);
	
	for (size_t i = 0; i < Dimension1; i++){
		for (size_t j = 0; j < Dimension2; j++){
			x_cpu(i,j) = 50.0+i-j;
			y_cpu(i,j) = i+j+1;
			result(i,j)= std::max(x_cpu(i,j),y_cpu(i,j));
		}
	}
	gpu::matrix<float> x = gpu::copy_to_gpu(x_cpu);
	gpu::matrix<float> y = gpu::copy_to_gpu(y_cpu);
	checkDenseExpressionEquality(max(x,y),result);
}
BOOST_AUTO_TEST_CASE( Remora_matrix_Binary_Min)
{
	matrix<float> x_cpu(Dimension1, Dimension2); 
	matrix<float> y_cpu(Dimension1, Dimension2); 
	matrix<float> result(Dimension1, Dimension2);
	
	for (size_t i = 0; i < Dimension1; i++){
		for (size_t j = 0; j < Dimension2; j++){
			x_cpu(i,j) = 50.0+i-j;
			y_cpu(i,j) = i+j+1;
			result(i,j)= std::min(x_cpu(i,j),y_cpu(i,j));
		}
	}
	gpu::matrix<float> x = gpu::copy_to_gpu(x_cpu);
	gpu::matrix<float> y = gpu::copy_to_gpu(y_cpu);
	checkDenseExpressionEquality(min(x,y),result);
}


////////////////////////////////////////////////////////////////////////
////////////ROW-WISE REDUCTIONS
////////////////////////////////////////////////////////////////////////
BOOST_AUTO_TEST_CASE( Remora_sum_rows){
	matrix<float> x_cpu(Dimension1, Dimension2,0.0); 
	
	for (size_t i = 0; i < Dimension1; i++){
		for (size_t j = 0; j < Dimension2; j++){
			x_cpu(i,j) = -3.0+i-j;
		}
	}
	vector<float> result = sum_rows(x_cpu);
	gpu::matrix<float,row_major> x_row = gpu::copy_to_gpu(x_cpu);
	gpu::matrix<float,column_major> x_col = gpu::copy_to_gpu(x_cpu);
	checkDenseExpressionEquality(sum_rows(x_row),result);
	checkDenseExpressionEquality(sum_rows(x_col),result);
}
BOOST_AUTO_TEST_CASE( Remora_sum_columns){
	matrix<float> x_cpu(Dimension1, Dimension2,0.0); 
	
	for (size_t i = 0; i < Dimension1; i++){
		for (size_t j = 0; j < Dimension2; j++){
			x_cpu(i,j) = -3.0+i-j;
		}
	}
	vector<float> result = sum_columns(x_cpu);
	gpu::matrix<float,row_major> x_row = gpu::copy_to_gpu(x_cpu);
	gpu::matrix<float,column_major> x_col = gpu::copy_to_gpu(x_cpu);
	checkDenseExpressionEquality(sum_columns(x_row),result);
	checkDenseExpressionEquality(sum_columns(x_col),result);
}

////////////////////////////////////////////////////////////////////////
////////////REDUCTIONS
////////////////////////////////////////////////////////////////////////

BOOST_AUTO_TEST_CASE( Remora_trace){
	matrix<float> x_cpu(Dimension1, Dimension1); 
	float result = 0.0f;
	for (size_t i = 0; i < Dimension1; i++){
		for (size_t j = 0; j < Dimension1; j++){
			x_cpu(i,j) = -3.0+2.0*i-j;
		}
		result += x_cpu(i,i);
	}
	gpu::matrix<float,row_major> x_row = gpu::copy_to_gpu(x_cpu);
	gpu::matrix<float,column_major> x_col = gpu::copy_to_gpu(x_cpu);
	BOOST_CHECK_CLOSE(trace(x_row),result, 1.e-6);
	BOOST_CHECK_CLOSE(trace(x_col),result, 1.e-6);
}

BOOST_AUTO_TEST_CASE( Remora_norm_1){
	matrix<float> x_cpu(Dimension1, Dimension1); 
	for (size_t i = 0; i < Dimension1; i++){
		for (size_t j = 0; j < Dimension1; j++){
			x_cpu(i,j) = -3.0+2.0*i-j;
		}
	}
	float result = norm_1(x_cpu);
	
	gpu::matrix<float,row_major> x_row = gpu::copy_to_gpu(x_cpu);
	gpu::matrix<float,column_major> x_col = gpu::copy_to_gpu(x_cpu);
	BOOST_CHECK_CLOSE(norm_1(x_row),result, 1.e-6);
	BOOST_CHECK_CLOSE(norm_1(x_col),result, 1.e-6);
}
BOOST_AUTO_TEST_CASE( Remora_norm_inf){
	matrix<float> x_cpu(Dimension1, Dimension1); 
	for (size_t i = 0; i < Dimension1; i++){
		for (size_t j = 0; j < Dimension1; j++){
			x_cpu(i,j) = -3.0+2.0*i-j;
		}
	}
	float result = norm_inf(x_cpu);
	
	gpu::matrix<float,row_major> x_row = gpu::copy_to_gpu(x_cpu);
	gpu::matrix<float,column_major> x_col = gpu::copy_to_gpu(x_cpu);
	BOOST_CHECK_CLOSE(norm_inf(x_row),result, 1.e-6);
	BOOST_CHECK_CLOSE(norm_inf(x_col),result, 1.e-6);
}

BOOST_AUTO_TEST_CASE( Remora_norm_Frobenius){
	matrix<float> x_cpu(Dimension1, Dimension1); 
	for (size_t i = 0; i < Dimension1; i++){
		for (size_t j = 0; j < Dimension1; j++){
			x_cpu(i,j) = -3.0+2.0*i-j;
		}
	}
	float result = norm_frobenius(x_cpu);
	
	gpu::matrix<float,row_major> x_row = gpu::copy_to_gpu(x_cpu);
	gpu::matrix<float,column_major> x_col = gpu::copy_to_gpu(x_cpu);
	BOOST_CHECK_CLOSE(norm_frobenius(x_row),result, 1.e-6);
	BOOST_CHECK_CLOSE(norm_frobenius(x_col),result, 1.e-6);
}

BOOST_AUTO_TEST_CASE( Remora_sum){
	matrix<float> x_cpu(Dimension1, Dimension1); 
	for (size_t i = 0; i < Dimension1; i++){
		for (size_t j = 0; j < Dimension1; j++){
			x_cpu(i,j) = -3.0+2.0*i-j;
		}
	}
	float result = sum(x_cpu);
	
	gpu::matrix<float,row_major> x_row = gpu::copy_to_gpu(x_cpu);
	gpu::matrix<float,column_major> x_col = gpu::copy_to_gpu(x_cpu);
	BOOST_CHECK_CLOSE(sum(x_row),result, 1.e-6);
	BOOST_CHECK_CLOSE(sum(x_col),result, 1.e-6);
}

BOOST_AUTO_TEST_CASE( Remora_max){
	matrix<float> x_cpu(Dimension1, Dimension1); 
	for (size_t i = 0; i < Dimension1; i++){
		for (size_t j = 0; j < Dimension1; j++){
			x_cpu(i,j) = -3.0+2.0*i-j;
		}
	}
	float result = max(x_cpu);
	
	gpu::matrix<float,row_major> x_row = gpu::copy_to_gpu(x_cpu);
	gpu::matrix<float,column_major> x_col = gpu::copy_to_gpu(x_cpu);
	BOOST_CHECK_CLOSE(max(x_row),result, 1.e-6);
	BOOST_CHECK_CLOSE(max(x_col),result, 1.e-6);
}

BOOST_AUTO_TEST_CASE( Remora_min){
	matrix<float> x_cpu(Dimension1, Dimension1); 
	for (size_t i = 0; i < Dimension1; i++){
		for (size_t j = 0; j < Dimension1; j++){
			x_cpu(i,j) = -3.0+2.0*i-j;
		}
	}
	float result = min(x_cpu);
	
	gpu::matrix<float,row_major> x_row = gpu::copy_to_gpu(x_cpu);
	gpu::matrix<float,column_major> x_col = gpu::copy_to_gpu(x_cpu);
	BOOST_CHECK_CLOSE(min(x_row),result, 1.e-6);
	BOOST_CHECK_CLOSE(min(x_col),result, 1.e-6);
}

BOOST_AUTO_TEST_CASE( Remora_frobenius_prod){
	matrix<float> x_cpu(Dimension1, Dimension2); 
	matrix<float> y_cpu(Dimension1, Dimension2); 
	for (size_t i = 0; i < Dimension1; i++){
		for (size_t j = 0; j < Dimension1; j++){
			x_cpu(i,j) = -3.0+2.0*i-j;
			y_cpu(i,j) = i+j+1;
		}
	}
	float result = frobenius_prod(x_cpu,y_cpu);
	
	gpu::matrix<float,row_major> x_row = gpu::copy_to_gpu(x_cpu);
	gpu::matrix<float,column_major> x_col = gpu::copy_to_gpu(x_cpu);
	gpu::matrix<float,row_major> y_row = gpu::copy_to_gpu(y_cpu);
	gpu::matrix<float,column_major> y_col = gpu::copy_to_gpu(y_cpu);
	BOOST_CHECK_CLOSE(frobenius_prod(x_row,y_row),result, 1.e-6);
	BOOST_CHECK_CLOSE(frobenius_prod(x_row,y_col),result, 1.e-6);
	BOOST_CHECK_CLOSE(frobenius_prod(x_col,y_row),result, 1.e-6);
	BOOST_CHECK_CLOSE(frobenius_prod(x_col,y_col),result, 1.e-6);
}

BOOST_AUTO_TEST_SUITE_END()
