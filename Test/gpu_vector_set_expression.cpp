#define BOOST_TEST_MODULE Remora_GPU_Vector_Set_Expression
#define BOOST_COMPUTE_DEBUG_KERNEL_COMPILATION
#include <boost/test/unit_test.hpp>
#include <boost/test/floating_point_comparison.hpp>

#include <remora/matrix_expression.hpp>
#include <remora/dense.hpp>
#include <remora/device_copy.hpp>

#include <boost/mpl/list.hpp>

using namespace remora;


template<class Operation, class Result>
void checkDenseExpressionEquality(
	vector_expression<Operation, gpu_tag> const& op_gpu, Result const& result
){
	BOOST_REQUIRE_EQUAL(op_gpu().size(), result.size());
	
	vector<float> op = copy_to_cpu(op_gpu());
	for(std::size_t i = 0; i != op.size(); ++i){
		BOOST_CHECK_CLOSE(result(i), op(i),1.e-4);
	}
}
template<class Operation, class Result>
void checkDenseExpressionEquality(
	vector_set_expression<Operation, gpu_tag> const& op_gpu, Result const& result
){
	matrix<float, row_major, gpu_tag> mat_gpu = op_gpu;
	matrix<float> op = copy_to_cpu(mat_gpu);
	BOOST_REQUIRE_EQUAL(op.size1(), result.size1());
	BOOST_REQUIRE_EQUAL(op.size2(), result.size2());
	for(std::size_t i = 0; i != op.size1(); ++i){
		for(std::size_t j = 0; j != op.size2(); ++j){
			BOOST_CHECK_CLOSE(result(i,j), op(i,j),1.e-4);
		}
	}
}

std::size_t Dimension1 = 50;
std::size_t Dimension2 = 100;

BOOST_AUTO_TEST_SUITE (Remora_vector_set_expression)

typedef boost::mpl::list<row_major,column_major> result_orientations;
////////////////////////////////////////////////////////////////////////
////////////ROW-WISE REDUCTIONS
////////////////////////////////////////////////////////////////////////
BOOST_AUTO_TEST_CASE_TEMPLATE( Remora_sum_set, Orientation, result_orientations )
{
	matrix<float, Orientation> x_cpu(Dimension1, Dimension2); 
	vector<float> resultColumns(Dimension2,0.0);
	vector<float> resultRows(Dimension1,0.0);
	
	for (size_t i = 0; i < Dimension1; i++){
		for (size_t j = 0; j < Dimension2; j++){
			x_cpu(i,j) = i-3.0-j;
			resultColumns(j) += x_cpu(i,j);
			resultRows(i) += x_cpu(i,j);
		}
	}
	
	matrix<float, Orientation, gpu_tag> x = copy_to_device(x_cpu, gpu_tag());
	
	checkDenseExpressionEquality(sum(as_rows(x)),resultRows);
	checkDenseExpressionEquality(sum(as_columns(x)),resultColumns);
}
BOOST_AUTO_TEST_CASE_TEMPLATE( Remora_max_set, Orientation, result_orientations )
{
	matrix<float, Orientation> x_cpu(Dimension1, Dimension2); 
	vector<float> resultColumns(Dimension2,0.0);
	vector<float> resultRows(Dimension1,0.0);
	
	for (size_t i = 0; i < Dimension1; i++){
		for (size_t j = 0; j < Dimension2; j++){
			x_cpu(i,j) = i-3.0-j;
			if(i == 0)
				resultColumns(j) = x_cpu(i,j);
			else
				resultColumns(j) = std::max(resultColumns(j), x_cpu(i,j));
			
			if(j == 0)
				resultRows(i) = x_cpu(i,j);
			else
				resultRows(i) = std::max(resultRows(i), x_cpu(i,j));
		}
	}
	
	matrix<float, Orientation, gpu_tag> x = copy_to_device(x_cpu, gpu_tag());
	
	checkDenseExpressionEquality(max(as_rows(x)),resultRows);
	checkDenseExpressionEquality(max(as_columns(x)),resultColumns);
}
BOOST_AUTO_TEST_CASE_TEMPLATE( Remora_min_set, Orientation, result_orientations )
{
	matrix<float, Orientation> x_cpu(Dimension1, Dimension2); 
	vector<float> resultColumns(Dimension2,0.0);
	vector<float> resultRows(Dimension1,0.0);
	
	for (size_t i = 0; i < Dimension1; i++){
		for (size_t j = 0; j < Dimension2; j++){
			x_cpu(i,j) = i-3.0-j;
			if(i == 0)
				resultColumns(j) = x_cpu(i,j);
			else
				resultColumns(j) = std::min(resultColumns(j), x_cpu(i,j));
			
			if(j == 0)
				resultRows(i) = x_cpu(i,j);
			else
				resultRows(i) = std::min(resultRows(i), x_cpu(i,j));
		}
	}
	
	matrix<float, Orientation, gpu_tag> x = copy_to_device(x_cpu, gpu_tag());
	
	checkDenseExpressionEquality(min(as_rows(x)),resultRows);
	checkDenseExpressionEquality(min(as_columns(x)),resultColumns);
}


BOOST_AUTO_TEST_CASE_TEMPLATE( Remora_norm_1_set, Orientation, result_orientations )
{
	matrix<float, Orientation> x_cpu(Dimension1, Dimension2); 
	vector<float> resultColumns(Dimension2,0.0);
	vector<float> resultRows(Dimension1,0.0);
	
	for (size_t i = 0; i < Dimension1; i++){
		for (size_t j = 0; j < Dimension2; j++){
			x_cpu(i,j) = i-3.0-j;
			resultColumns(j) += std::abs(x_cpu(i,j));
			resultRows(i) += std::abs(x_cpu(i,j));
		}
	}
	
	matrix<float, Orientation, gpu_tag> x = copy_to_device(x_cpu, gpu_tag());
	
	checkDenseExpressionEquality(norm_1(as_rows(x)),resultRows);
	checkDenseExpressionEquality(norm_1(as_columns(x)),resultColumns);
}

BOOST_AUTO_TEST_CASE_TEMPLATE( Remora_norm_sqr_set, Orientation, result_orientations )
{
	matrix<float, Orientation> x_cpu(Dimension1, Dimension2); 
	vector<float> resultColumns(Dimension2,0.0);
	vector<float> resultRows(Dimension1,0.0);
	
	for (size_t i = 0; i < Dimension1; i++){
		for (size_t j = 0; j < Dimension2; j++){
			x_cpu(i,j) = i-3.0-j;
			resultColumns(j) += x_cpu(i,j) * x_cpu(i,j);
			resultRows(i) += x_cpu(i,j) * x_cpu(i,j);
		}
	}
	
	matrix<float, Orientation, gpu_tag> x = copy_to_device(x_cpu, gpu_tag());
	
	checkDenseExpressionEquality(norm_sqr(as_rows(x)),resultRows);
	checkDenseExpressionEquality(norm_sqr(as_columns(x)),resultColumns);
}

BOOST_AUTO_TEST_CASE_TEMPLATE( Remora_norm_2_set, Orientation, result_orientations )
{
	matrix<float, Orientation> x_cpu(Dimension1, Dimension2); 
	vector<float> resultColumns(Dimension2,0.0);
	vector<float> resultRows(Dimension1,0.0);
	
	for (size_t i = 0; i < Dimension1; i++){
		for (size_t j = 0; j < Dimension2; j++){
			x_cpu(i,j) = i-3.0-j;
			resultColumns(j) += x_cpu(i,j) * x_cpu(i,j);
			resultRows(i) += x_cpu(i,j) * x_cpu(i,j);
		}
	}
	
	resultRows = sqrt(resultRows);
	resultColumns = sqrt(resultColumns);
	
	matrix<float, Orientation, gpu_tag> x = copy_to_device(x_cpu, gpu_tag());
	
	checkDenseExpressionEquality(norm_2(as_rows(x)),resultRows);
	checkDenseExpressionEquality(norm_2(as_columns(x)),resultColumns);
}

BOOST_AUTO_TEST_CASE_TEMPLATE( Remora_norm_inf_set, Orientation, result_orientations )
{
	matrix<float, Orientation> x_cpu(Dimension1, Dimension2); 
	vector<float> resultColumns(Dimension2,0.0);
	vector<float> resultRows(Dimension1,0.0);
	
	for (size_t i = 0; i < Dimension1; i++){
		for (size_t j = 0; j < Dimension2; j++){
			x_cpu(i,j) = i-3.0-j;
			if(i == 0)
				resultColumns(j) = std::abs(x_cpu(i,j));
			else
				resultColumns(j) = std::max(resultColumns(j), std::abs(x_cpu(i,j)));
			
			if(j == 0)
				resultRows(i) = std::abs(x_cpu(i,j));
			else
				resultRows(i) = std::max(resultRows(i), std::abs(x_cpu(i,j)));
		}
	}
	
	matrix<float, Orientation, gpu_tag> x = copy_to_device(x_cpu, gpu_tag());
	
	checkDenseExpressionEquality(norm_inf(as_rows(x)),resultRows);
	checkDenseExpressionEquality(norm_inf(as_columns(x)),resultColumns);
}


/////////////////////////////////////////////////////////////
//////UNARY TRANSFORMATIONS///////
////////////////////////////////////////////////////////////
BOOST_AUTO_TEST_CASE_TEMPLATE( Remora_vector_set_Unary_Minus, Orientation, result_orientations )
{
	matrix<float, Orientation> x_cpu(Dimension1, Dimension2);
	matrix<float> result(Dimension1, Dimension2);
	
	for (size_t i = 0; i < Dimension1; i++){
		for (size_t j = 0; j < Dimension2; j++){
			x_cpu(i,j) = i-3.0+j;
			result(i,j)= -x_cpu(i,j);
		}
	}
	matrix<float, Orientation, gpu_tag> x = copy_to_device(x_cpu, gpu_tag());
	checkDenseExpressionEquality(-as_rows(x),result);
	checkDenseExpressionEquality(-as_columns(x),result);
}
BOOST_AUTO_TEST_CASE_TEMPLATE( Remora_vector_set_Scalar_elem_inv, Orientation, result_orientations )
{
	matrix<float, Orientation> x_cpu(Dimension1, Dimension2); 
	matrix<float> result(Dimension1, Dimension2);
	
	for (size_t i = 0; i < Dimension1; i++){
		for (size_t j = 0; j < Dimension2; j++){
			x_cpu(i,j) = i+j+3.0;
			result(i,j)= 1.0/x_cpu(i,j);
		}
	}
	matrix<float, Orientation, gpu_tag> x = copy_to_device(x_cpu, gpu_tag());
	checkDenseExpressionEquality(elem_inv(as_rows(x)),result);
	checkDenseExpressionEquality(elem_inv(as_columns(x)),result);
}
BOOST_AUTO_TEST_CASE_TEMPLATE( Remora_vector_set_Abs, Orientation, result_orientations )
{
	matrix<float, Orientation> x_cpu(Dimension1, Dimension2); 
	matrix<float> result(Dimension1, Dimension2);
	
	for (size_t i = 0; i < Dimension1; i++){
		for (size_t j = 0; j < Dimension2; j++){
			x_cpu(i,j) = i-3.0-j;
			result(i,j)= std::abs(x_cpu(i,j));
		}
	}
	matrix<float, Orientation, gpu_tag> x = copy_to_device(x_cpu, gpu_tag());
	checkDenseExpressionEquality(abs(as_rows(x)),result);
	checkDenseExpressionEquality(abs(as_columns(x)),result);
}
BOOST_AUTO_TEST_CASE_TEMPLATE( Remora_vector_set_Sqr, Orientation, result_orientations )
{
	matrix<float, Orientation> x_cpu(Dimension1, Dimension2); 
	matrix<float> result(Dimension1, Dimension2);
	for (size_t i = 0; i < Dimension1; i++){
		for (size_t j = 0; j < Dimension2; j++){
			x_cpu(i,j) = i-3.0-j;
			result(i,j)= x_cpu(i,j) * x_cpu(i,j);
		}
	}
	matrix<float, Orientation, gpu_tag> x = copy_to_device(x_cpu, gpu_tag());
	checkDenseExpressionEquality(sqr(as_rows(x)),result);
	checkDenseExpressionEquality(sqr(as_columns(x)),result);
}
BOOST_AUTO_TEST_CASE_TEMPLATE( Remora_vector_set_Sqrt, Orientation, result_orientations )
{
	matrix<float, Orientation> x_cpu(Dimension1, Dimension2); 
	matrix<float> result(Dimension1, Dimension2);
	
	for (size_t i = 0; i < Dimension1; i++){
		for (size_t j = 0; j < Dimension2; j++){
			x_cpu(i,j) = i+j+3.0;
			result(i,j)= std::sqrt(x_cpu(i,j));
		}
	}
	matrix<float, Orientation, gpu_tag> x = copy_to_device(x_cpu, gpu_tag());
	checkDenseExpressionEquality(sqrt(as_rows(x)),result);
	checkDenseExpressionEquality(sqrt(as_columns(x)),result);
}
BOOST_AUTO_TEST_CASE_TEMPLATE( Remora_vector_set_Cbrt, Orientation, result_orientations )
{
	matrix<float, Orientation> x_cpu(Dimension1, Dimension2); 
	matrix<float> result(Dimension1, Dimension2);
	
	for (size_t i = 0; i < Dimension1; i++){
		for (size_t j = 0; j < Dimension2; j++){
			x_cpu(i,j) = i+j+3.0;
			result(i,j)= std::cbrt(x_cpu(i,j));
		}
	}
	matrix<float, Orientation, gpu_tag> x = copy_to_device(x_cpu, gpu_tag());
	checkDenseExpressionEquality(cbrt(as_rows(x)),result);
	checkDenseExpressionEquality(cbrt(as_columns(x)),result);
}
BOOST_AUTO_TEST_CASE_TEMPLATE( Remora_vector_set_Exp, Orientation, result_orientations )
{
	matrix<float, Orientation> x_cpu(Dimension1, Dimension2); 
	matrix<float> result(Dimension1, Dimension2);
	
	for (size_t i = 0; i < Dimension1; i++){
		for (size_t j = 0; j < Dimension2; j++){
			x_cpu(i,j) = 0.01*(i-3.0-float(j));
			result(i,j)= std::exp(x_cpu(i,j));
		}
	}
	matrix<float, Orientation, gpu_tag> x = copy_to_device(x_cpu, gpu_tag());
	checkDenseExpressionEquality(exp(as_rows(x)),result);
	checkDenseExpressionEquality(exp(as_columns(x)),result);
}
BOOST_AUTO_TEST_CASE_TEMPLATE( Remora_vector_set_Log, Orientation, result_orientations )
{
	matrix<float, Orientation> x_cpu(Dimension1, Dimension2); 
	matrix<float> result(Dimension1, Dimension2);
	
	for (size_t i = 0; i < Dimension1; i++){
		for (size_t j = 0; j < Dimension2; j++){
			x_cpu(i,j) = i+j+3.0;
			result(i,j)= std::log(x_cpu(i,j));
		}
	}
	matrix<float, Orientation, gpu_tag> x = copy_to_device(x_cpu, gpu_tag());
	checkDenseExpressionEquality(log(as_rows(x)),result);
	checkDenseExpressionEquality(log(as_columns(x)),result);
}
BOOST_AUTO_TEST_CASE_TEMPLATE( Remora_vector_set_sin, Orientation, result_orientations )
{
	matrix<float, Orientation> x_cpu(Dimension1, Dimension2); 
	matrix<float> result(Dimension1, Dimension2);
	
	for (size_t i = 0; i < Dimension1; i++){
		for (size_t j = 0; j < Dimension2; j++){
			x_cpu(i,j) = float(i)/Dimension1-float(j)/Dimension2;
			result(i,j)= std::sin(x_cpu(i,j));
		}
	}
	matrix<float, Orientation, gpu_tag> x = copy_to_device(x_cpu, gpu_tag());
	checkDenseExpressionEquality(sin(as_rows(x)),result);
	checkDenseExpressionEquality(sin(as_columns(x)),result);
}

BOOST_AUTO_TEST_CASE_TEMPLATE( Remora_vector_set_cos, Orientation, result_orientations )
{
	matrix<float, Orientation> x_cpu(Dimension1, Dimension2); 
	matrix<float> result(Dimension1, Dimension2);
	
	for (size_t i = 0; i < Dimension1; i++){
		for (size_t j = 0; j < Dimension2; j++){
			x_cpu(i,j) = float(i)/Dimension1-float(j)/Dimension2;
			result(i,j)= std::cos(x_cpu(i,j));
		}
	}
	matrix<float, Orientation, gpu_tag> x = copy_to_device(x_cpu, gpu_tag());
	checkDenseExpressionEquality(cos(as_rows(x)),result);
	checkDenseExpressionEquality(cos(as_columns(x)),result);
}

BOOST_AUTO_TEST_CASE_TEMPLATE( Remora_vector_set_tan, Orientation, result_orientations )
{
	matrix<float, Orientation> x_cpu(Dimension1, Dimension2); 
	matrix<float> result(Dimension1, Dimension2);
	
	for (size_t i = 0; i < Dimension1; i++){
		for (size_t j = 0; j < Dimension2; j++){
			x_cpu(i,j) = float(i)/Dimension1-float(j)/Dimension2;
			result(i,j)= std::tan(x_cpu(i,j));
		}
	}
	matrix<float, Orientation, gpu_tag> x = copy_to_device(x_cpu, gpu_tag());
	checkDenseExpressionEquality(tan(as_rows(x)),result);
	checkDenseExpressionEquality(tan(as_columns(x)),result);
}

BOOST_AUTO_TEST_CASE_TEMPLATE( Remora_vector_set_asin, Orientation, result_orientations )
{
	matrix<float, Orientation> x_cpu(Dimension1, Dimension2); 
	matrix<float> result(Dimension1, Dimension2);
	
	for (size_t i = 0; i < Dimension1; i++){
		for (size_t j = 0; j < Dimension2; j++){
			x_cpu(i,j) = float(i)/Dimension1-float(j)/Dimension2;
			result(i,j)= std::asin(x_cpu(i,j));
		}
	}
	matrix<float, Orientation, gpu_tag> x = copy_to_device(x_cpu, gpu_tag());
	checkDenseExpressionEquality(asin(as_rows(x)),result);
	checkDenseExpressionEquality(asin(as_columns(x)),result);
}

BOOST_AUTO_TEST_CASE_TEMPLATE( Remora_vector_set_acos, Orientation, result_orientations )
{
	matrix<float, Orientation> x_cpu(Dimension1, Dimension2); 
	matrix<float> result(Dimension1, Dimension2);
	
	for (size_t i = 0; i < Dimension1; i++){
		for (size_t j = 0; j < Dimension2; j++){
			x_cpu(i,j) = float(i)/Dimension1-float(j)/Dimension2;
			result(i,j)= std::acos(x_cpu(i,j));
		}
	}
	matrix<float, Orientation, gpu_tag> x = copy_to_device(x_cpu, gpu_tag());
	checkDenseExpressionEquality(acos(as_rows(x)),result);
	checkDenseExpressionEquality(acos(as_columns(x)),result);
}

BOOST_AUTO_TEST_CASE_TEMPLATE( Remora_vector_set_atan, Orientation, result_orientations )
{
	matrix<float, Orientation> x_cpu(Dimension1, Dimension2); 
	matrix<float> result(Dimension1, Dimension2);
	
	for (size_t i = 0; i < Dimension1; i++){
		for (size_t j = 0; j < Dimension2; j++){
			x_cpu(i,j) = float(i)/Dimension1-float(j)/Dimension2;
			result(i,j)= std::atan(x_cpu(i,j));
		}
	}
	matrix<float, Orientation, gpu_tag> x = copy_to_device(x_cpu, gpu_tag());
	checkDenseExpressionEquality(atan(as_rows(x)),result);
	checkDenseExpressionEquality(atan(as_columns(x)),result);
}

BOOST_AUTO_TEST_CASE_TEMPLATE( Remora_vector_set_erf, Orientation, result_orientations )
{
	matrix<float, Orientation> x_cpu(Dimension1, Dimension2); 
	matrix<float> result(Dimension1, Dimension2);
	
	for (size_t i = 0; i < Dimension1; i++){
		for (size_t j = 0; j < Dimension2; j++){
			x_cpu(i,j) = float(i)/Dimension1-float(j)/Dimension2;
			result(i,j)= std::erf(x_cpu(i,j));
		}
	}
	matrix<float, Orientation, gpu_tag> x = copy_to_device(x_cpu, gpu_tag());
	checkDenseExpressionEquality(erf(as_rows(x)),result);
	checkDenseExpressionEquality(erf(as_columns(x)),result);
}

BOOST_AUTO_TEST_CASE_TEMPLATE( Remora_vector_set_erfc, Orientation, result_orientations )
{
	matrix<float, Orientation> x_cpu(Dimension1, Dimension2); 
	matrix<float> result(Dimension1, Dimension2);
	
	for (size_t i = 0; i < Dimension1; i++){
		for (size_t j = 0; j < Dimension2; j++){
			x_cpu(i,j) = float(i)/Dimension1-float(j)/Dimension2;
			result(i,j)= std::erfc(x_cpu(i,j));
		}
	}
	matrix<float, Orientation, gpu_tag> x = copy_to_device(x_cpu, gpu_tag());
	checkDenseExpressionEquality(erfc(as_rows(x)),result);
	checkDenseExpressionEquality(erfc(as_columns(x)),result);
}

BOOST_AUTO_TEST_CASE_TEMPLATE( Remora_vector_set_Tanh, Orientation, result_orientations )
{
	matrix<float, Orientation> x_cpu(Dimension1, Dimension2); 
	matrix<float> result(Dimension1, Dimension2);
	
	for (size_t i = 0; i < Dimension1; i++){
		for (size_t j = 0; j < Dimension2; j++){
			x_cpu(i,j) = i-3.0-j;
			result(i,j)= std::tanh(x_cpu(i,j));
		}
	}
	matrix<float, Orientation, gpu_tag> x = copy_to_device(x_cpu, gpu_tag());
	checkDenseExpressionEquality(tanh(as_rows(x)),result);
	checkDenseExpressionEquality(tanh(as_columns(x)),result);
}
BOOST_AUTO_TEST_CASE_TEMPLATE( Remora_vector_set_Sigmoid, Orientation, result_orientations )
{
	matrix<float, Orientation> x_cpu(Dimension1, Dimension2); 
	matrix<float> result(Dimension1, Dimension2);
	
	for (size_t i = 0; i < Dimension1; i++){
		for (size_t j = 0; j < Dimension2; j++){
			x_cpu(i,j) = 0.01*(i-3.0-j);
			result(i,j)= 1.0/(1.0+std::exp(-x_cpu(i,j)));
		}
	}
	matrix<float, Orientation, gpu_tag> x = copy_to_device(x_cpu, gpu_tag());
	checkDenseExpressionEquality(sigmoid(as_rows(x)),result);
	checkDenseExpressionEquality(sigmoid(as_columns(x)),result);
}
BOOST_AUTO_TEST_CASE_TEMPLATE( Remora_vector_set_SoftPlus, Orientation, result_orientations )
{
	matrix<float, Orientation> x_cpu(Dimension1, Dimension2); 
	matrix<float> result(Dimension1, Dimension2);
	
	for (size_t i = 0; i < Dimension1; i++){
		for (size_t j = 0; j < Dimension2; j++){
			x_cpu(i,j) = i-3.0-j;
			result(i,j)= std::log(1+std::exp(x_cpu(i,j)));
		}
	}
	matrix<float, Orientation, gpu_tag> x = copy_to_device(x_cpu, gpu_tag());
	checkDenseExpressionEquality(softPlus(as_rows(x)),result);
	checkDenseExpressionEquality(softPlus(as_columns(x)),result);
}


//~ /////////////////////////////////////////////////////////////
//~ //////SCALAR TRANSFORMATIONS///////
//~ ////////////////////////////////////////////////////////////
//~ BOOST_AUTO_TEST_CASE_TEMPLATE( Remora_vector_set_Scalar_Multiply, Orientation, result_orientations )
//~ {
	//~ matrix<float, Orientation> x(Dimension1, Dimension2); 
	//~ matrix<float> result(Dimension1, Dimension2);
	
	//~ for (size_t i = 0; i < Dimension1; i++){
		//~ for (size_t j = 0; j < Dimension2; j++){
			//~ x(i,j) = i-3.0+j;
			//~ result(i,j)= 5.0* x(i,j);
		//~ }
	//~ }
	//~ checkDenseExpressionEquality(5.0*x,result);
	//~ checkDenseExpressionEquality(x*5.0,result);
//~ }
//~ BOOST_AUTO_TEST_CASE_TEMPLATE( Remora_vector_set_Scalar_Add, Orientation, result_orientations )
//~ {
	//~ matrix<float, Orientation> x(Dimension1, Dimension2); 
	//~ matrix<float> result(Dimension1, Dimension2);
	
	//~ for (size_t i = 0; i < Dimension1; i++){
		//~ for (size_t j = 0; j < Dimension2; j++){
			//~ x(i,j) = i-3.0+j;
			//~ result(i,j)= 5.0 + x(i,j);
		//~ }
	//~ }
	//~ checkDenseExpressionEquality(5.0+x,result);
	//~ checkDenseExpressionEquality(x+5.0,result);
//~ }
//~ BOOST_AUTO_TEST_CASE_TEMPLATE( Remora_vector_set_Scalar_Subtract, Orientation, result_orientations )
//~ {
	//~ matrix<float, Orientation> x(Dimension1, Dimension2); 
	//~ matrix<float> result1(Dimension1, Dimension2);
	//~ matrix<float> result2(Dimension1, Dimension2);
	
	//~ for (size_t i = 0; i < Dimension1; i++){
		//~ for (size_t j = 0; j < Dimension2; j++){
			//~ x(i,j) = i-3.0+j;
			//~ result1(i,j)= 5.0 - x(i,j);
			//~ result2(i,j)= x(i,j) - 5.0;
		//~ }
	//~ }
	//~ checkDenseExpressionEquality(5.0- x,result1);
	//~ checkDenseExpressionEquality(x - 5.0,result2);
//~ }
//~ BOOST_AUTO_TEST_CASE_TEMPLATE( Remora_vector_set_Scalar_Div, Orientation, result_orientations )
//~ {
	//~ matrix<float, Orientation> x(Dimension1, Dimension2); 
	//~ matrix<float> result(Dimension1, Dimension2);
	
	//~ for (size_t i = 0; i < Dimension1; i++){
		//~ for (size_t j = 0; j < Dimension2; j++){
			//~ x(i,j) = i-3.0+j;
			//~ result(i,j)= x(i,j)/5.0;
		//~ }
	//~ }
	//~ checkDenseExpressionEquality(x/5.0f,result);
//~ }

//~ BOOST_AUTO_TEST_CASE_TEMPLATE( Remora_vector_set_Pow, Orientation, result_orientations )
//~ {
	//~ matrix<float, Orientation> x(Dimension1, Dimension2); 
	//~ matrix<float> result(Dimension1, Dimension2);
	
	//~ for (size_t i = 0; i < Dimension1; i++){
		//~ for (size_t j = 0; j < Dimension2; j++){
			//~ x(i,j) = 0.001*(i+j+2.0);
			//~ result(i,j)= std::pow(x(i,j),3.2);
		//~ }
	//~ }
	//~ checkDenseExpressionEquality(pow(x,3.2),result);
//~ }

//~ BOOST_AUTO_TEST_CASE_TEMPLATE( Remora_vector_set_Unary_Min, Orientation, result_orientations )
//~ {
	//~ matrix<float, Orientation> x(Dimension1, Dimension2); 
	//~ matrix<float> result(Dimension1, Dimension2);
	
	//~ for (size_t i = 0; i < Dimension1; i++){
		//~ for (size_t j = 0; j < Dimension2; j++){
			//~ x(i,j) = i-1.0*j;
			//~ result(i,j)= std::min(x(i,j),5.0);
		//~ }
	//~ }
	//~ checkDenseExpressionEquality(min(x,5.0),result);
	//~ checkDenseExpressionEquality(min(5.0,x),result);
//~ }
//~ BOOST_AUTO_TEST_CASE_TEMPLATE( Remora_vector_set_Unary_Max, Orientation, result_orientations )
//~ {
	//~ matrix<float, Orientation> x(Dimension1, Dimension2); 
	//~ matrix<float> result(Dimension1, Dimension2);
	
	//~ for (size_t i = 0; i < Dimension1; i++){
		//~ for (size_t j = 0; j < Dimension2; j++){
			//~ x(i,j) = i-1.0*j;
			//~ result(i,j)= std::max(x(i,j),5.0);
		//~ }
	//~ }
	//~ checkDenseExpressionEquality(max(x,5.0),result);
	//~ checkDenseExpressionEquality(max(5.0,x),result);
//~ }


BOOST_AUTO_TEST_SUITE_END()
