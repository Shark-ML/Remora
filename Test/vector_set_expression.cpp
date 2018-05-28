#define BOOST_TEST_MODULE Remora_Vector_Set_Expression
#include <boost/test/unit_test.hpp>
#include <boost/test/floating_point_comparison.hpp>


#include <remora/matrix_expression.hpp>
#include <remora/dense.hpp>

#include <boost/mpl/list.hpp>

using namespace remora;


template<class Operation, class Result>
void checkDenseBlockAssign(
	vector_expression<Operation, cpu_tag> const& op, Result const& result
){
	BOOST_REQUIRE_EQUAL(op().size(), result.size());
	remora::vector<typename Result::value_type> res1(result.size(),1.0);
	remora::vector<typename Result::value_type> res2(result.size(),1.0);
	op().assign_to(res1);
	op().plus_assign_to(res2);
	
	for(std::size_t i = 0; i != op().size(); ++i){
		BOOST_CHECK_SMALL(res1(i) - result(i),typename Result::value_type(1.e-7));
		BOOST_CHECK_SMALL(res2(i) - result(i) - 1,typename Result::value_type(1.e-7));
	}
}

template<class Operation, class Result>
void checkDenseBlockAssign(
	matrix_expression<Operation, cpu_tag> const& op, Result const& result
){
	remora::matrix<typename Result::value_type> res1(result.size1(),result.size2(),1.0);
	remora::matrix<typename Result::value_type> res2(result.size1(),result.size2(),1.0);
	op().assign_to(res1);
	op().plus_assign_to(res2);
	
	for(std::size_t i = 0; i != op().size1(); ++i){
		for(std::size_t j = 0; j != op().size2(); ++j){
			BOOST_CHECK_SMALL(res1(i,j) - result(i,j),typename Result::value_type(1.e-7));
			BOOST_CHECK_SMALL(res2(i,j) - result(i,j) - 1,typename Result::value_type(1.e-7));
		}
	}
}
template<class M1>
double get(M1 const& m, std::size_t i, std::size_t j, row_major){
	return m(i,j);
}
template<class M1>
double get(M1 const& m, std::size_t i, std::size_t j, column_major){
	return m(j,i);
}
template<class Operation, class Result>
void checkDenseExpressionEquality(
	matrix_expression<Operation, cpu_tag> const& op, Result const& result
){
	BOOST_REQUIRE_EQUAL(op().size1(), result.size1());
	BOOST_REQUIRE_EQUAL(op().size2(), result.size2());
	
	//check that op(i,j) works
	for(std::size_t i = 0; i != op().size1(); ++i){
		for(std::size_t j = 0; j != op().size2(); ++j){
			BOOST_CHECK_CLOSE(result(i,j), op()(i,j),1.e-5);
		}
	}
	//check that iterators work
	for(std::size_t i = 0; i != major_size(op); ++i){
		auto pos = op().major_begin(i);
		for(std::size_t j = 0; j != minor_size(op); ++j,++pos){
			BOOST_CHECK_EQUAL(j, pos.index());
			BOOST_CHECK_CLOSE(get(result,i,j, typename Operation::orientation()), *pos,1.e-5);
		}
	}
	checkDenseBlockAssign(op,result);
}

template<class Operation, class Result>
void checkDenseExpressionEquality(
	vector_set_expression<Operation, cpu_tag> const& op, Result const& result
){
	checkDenseExpressionEquality(op().expression(), result);
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
	matrix<double, Orientation> x(Dimension1, Dimension2); 
	vector<double> resultColumns(Dimension2,0.0);
	vector<double> resultRows(Dimension1,0.0);
	
	for (size_t i = 0; i < Dimension1; i++){
		for (size_t j = 0; j < Dimension2; j++){
			x(i,j) = i-3.0-j;
			resultColumns(j) += x(i,j);
			resultRows(i) += x(i,j);
		}
	}
	
	checkDenseBlockAssign(sum(as_rows(x)),resultRows);
	checkDenseBlockAssign(sum(as_columns(x)),resultColumns);
}
BOOST_AUTO_TEST_CASE_TEMPLATE( Remora_max_set, Orientation, result_orientations )
{
	matrix<double, Orientation> x(Dimension1, Dimension2); 
	vector<double> resultColumns(Dimension2,0.0);
	vector<double> resultRows(Dimension1,0.0);
	
	for (size_t i = 0; i < Dimension1; i++){
		for (size_t j = 0; j < Dimension2; j++){
			x(i,j) = i-3.0-j;
			if(i == 0)
				resultColumns(j) = x(i,j);
			else
				resultColumns(j) = std::max(resultColumns(j), x(i,j));
			
			if(j == 0)
				resultRows(i) = x(i,j);
			else
				resultRows(i) = std::max(resultRows(i), x(i,j));
		}
	}
	
	checkDenseBlockAssign(max(as_rows(x)),resultRows);
	checkDenseBlockAssign(max(as_columns(x)),resultColumns);
}
BOOST_AUTO_TEST_CASE_TEMPLATE( Remora_min_set, Orientation, result_orientations )
{
	matrix<double, Orientation> x(Dimension1, Dimension2); 
	vector<double> resultColumns(Dimension2,0.0);
	vector<double> resultRows(Dimension1,0.0);
	
	for (size_t i = 0; i < Dimension1; i++){
		for (size_t j = 0; j < Dimension2; j++){
			x(i,j) = i-3.0-j;
			if(i == 0)
				resultColumns(j) = x(i,j);
			else
				resultColumns(j) = std::min(resultColumns(j), x(i,j));
			
			if(j == 0)
				resultRows(i) = x(i,j);
			else
				resultRows(i) = std::min(resultRows(i), x(i,j));
		}
	}
	
	checkDenseBlockAssign(min(as_rows(x)),resultRows);
	checkDenseBlockAssign(min(as_columns(x)),resultColumns);
}


BOOST_AUTO_TEST_CASE_TEMPLATE( Remora_norm_1_set, Orientation, result_orientations )
{
	matrix<double, Orientation> x(Dimension1, Dimension2); 
	vector<double> resultColumns(Dimension2,0.0);
	vector<double> resultRows(Dimension1,0.0);
	
	for (size_t i = 0; i < Dimension1; i++){
		for (size_t j = 0; j < Dimension2; j++){
			x(i,j) = i-3.0-j;
			resultColumns(j) += std::abs(x(i,j));
			resultRows(i) += std::abs(x(i,j));
		}
	}
	
	checkDenseBlockAssign(norm_1(as_rows(x)),resultRows);
	checkDenseBlockAssign(norm_1(as_columns(x)),resultColumns);
}

BOOST_AUTO_TEST_CASE_TEMPLATE( Remora_norm_sqr_set, Orientation, result_orientations )
{
	matrix<double, Orientation> x(Dimension1, Dimension2); 
	vector<double> resultColumns(Dimension2,0.0);
	vector<double> resultRows(Dimension1,0.0);
	
	for (size_t i = 0; i < Dimension1; i++){
		for (size_t j = 0; j < Dimension2; j++){
			x(i,j) = i-3.0-j;
			resultColumns(j) += x(i,j) * x(i,j);
			resultRows(i) += x(i,j) * x(i,j);
		}
	}
	
	checkDenseBlockAssign(norm_sqr(as_rows(x)),resultRows);
	checkDenseBlockAssign(norm_sqr(as_columns(x)),resultColumns);
}

BOOST_AUTO_TEST_CASE_TEMPLATE( Remora_norm_2_set, Orientation, result_orientations )
{
	matrix<double, Orientation> x(Dimension1, Dimension2); 
	vector<double> resultColumns(Dimension2,0.0);
	vector<double> resultRows(Dimension1,0.0);
	
	for (size_t i = 0; i < Dimension1; i++){
		for (size_t j = 0; j < Dimension2; j++){
			x(i,j) = i-3.0-j;
			resultColumns(j) += x(i,j) * x(i,j);
			resultRows(i) += x(i,j) * x(i,j);
		}
	}
	
	resultRows = sqrt(resultRows);
	resultColumns = sqrt(resultColumns);
	
	checkDenseBlockAssign(norm_2(as_rows(x)),resultRows);
	checkDenseBlockAssign(norm_2(as_columns(x)),resultColumns);
}

BOOST_AUTO_TEST_CASE_TEMPLATE( Remora_norm_inf_set, Orientation, result_orientations )
{
	matrix<double, Orientation> x(Dimension1, Dimension2); 
	vector<double> resultColumns(Dimension2,0.0);
	vector<double> resultRows(Dimension1,0.0);
	
	for (size_t i = 0; i < Dimension1; i++){
		for (size_t j = 0; j < Dimension2; j++){
			x(i,j) = i-3.0-j;
			if(i == 0)
				resultColumns(j) = std::abs(x(i,j));
			else
				resultColumns(j) = std::max(resultColumns(j), std::abs(x(i,j)));
			
			if(j == 0)
				resultRows(i) = std::abs(x(i,j));
			else
				resultRows(i) = std::max(resultRows(i), std::abs(x(i,j)));
		}
	}
	
	checkDenseBlockAssign(norm_inf(as_rows(x)),resultRows);
	checkDenseBlockAssign(norm_inf(as_columns(x)),resultColumns);
}


/////////////////////////////////////////////////////////////
//////UNARY TRANSFORMATIONS///////
////////////////////////////////////////////////////////////
BOOST_AUTO_TEST_CASE_TEMPLATE( Remora_vector_set_Unary_Minus, Orientation, result_orientations )
{
	matrix<double, Orientation> x(Dimension1, Dimension2);
	matrix<double> result(Dimension1, Dimension2);
	
	for (size_t i = 0; i < Dimension1; i++){
		for (size_t j = 0; j < Dimension2; j++){
			x(i,j) = i-3.0+j;
			result(i,j)= -x(i,j);
		}
	}
	checkDenseExpressionEquality(-as_rows(x),result);
	checkDenseExpressionEquality(-as_columns(x),result);
}
BOOST_AUTO_TEST_CASE_TEMPLATE( Remora_vector_set_Scalar_elem_inv, Orientation, result_orientations )
{
	matrix<double, Orientation> x(Dimension1, Dimension2); 
	matrix<double> result(Dimension1, Dimension2);
	
	for (size_t i = 0; i < Dimension1; i++){
		for (size_t j = 0; j < Dimension2; j++){
			x(i,j) = i+j+3.0;
			result(i,j)= 1.0/x(i,j);
		}
	}
	checkDenseExpressionEquality(elem_inv(as_rows(x)),result);
	checkDenseExpressionEquality(elem_inv(as_columns(x)),result);
}
BOOST_AUTO_TEST_CASE_TEMPLATE( Remora_vector_set_Abs, Orientation, result_orientations )
{
	matrix<double, Orientation> x(Dimension1, Dimension2); 
	matrix<double> result(Dimension1, Dimension2);
	
	for (size_t i = 0; i < Dimension1; i++){
		for (size_t j = 0; j < Dimension2; j++){
			x(i,j) = i-3.0-j;
			result(i,j)= std::abs(x(i,j));
		}
	}
	checkDenseExpressionEquality(abs(as_rows(x)),result);
	checkDenseExpressionEquality(abs(as_columns(x)),result);
}
BOOST_AUTO_TEST_CASE_TEMPLATE( Remora_vector_set_Sqr, Orientation, result_orientations )
{
	matrix<double, Orientation> x(Dimension1, Dimension2); 
	matrix<double> result(Dimension1, Dimension2);
	for (size_t i = 0; i < Dimension1; i++){
		for (size_t j = 0; j < Dimension2; j++){
			x(i,j) = i-3.0-j;
			result(i,j)= x(i,j) * x(i,j);
		}
	}
	checkDenseExpressionEquality(sqr(as_rows(x)),result);
	checkDenseExpressionEquality(sqr(as_columns(x)),result);
}
BOOST_AUTO_TEST_CASE_TEMPLATE( Remora_vector_set_Sqrt, Orientation, result_orientations )
{
	matrix<double, Orientation> x(Dimension1, Dimension2); 
	matrix<double> result(Dimension1, Dimension2);
	
	for (size_t i = 0; i < Dimension1; i++){
		for (size_t j = 0; j < Dimension2; j++){
			x(i,j) = i+j+3.0;
			result(i,j)= std::sqrt(x(i,j));
		}
	}
	checkDenseExpressionEquality(sqrt(as_rows(x)),result);
	checkDenseExpressionEquality(sqrt(as_columns(x)),result);
}
BOOST_AUTO_TEST_CASE_TEMPLATE( Remora_vector_set_Cbrt, Orientation, result_orientations )
{
	matrix<double, Orientation> x(Dimension1, Dimension2); 
	matrix<double> result(Dimension1, Dimension2);
	
	for (size_t i = 0; i < Dimension1; i++){
		for (size_t j = 0; j < Dimension2; j++){
			x(i,j) = i+j+3.0;
			result(i,j)= std::cbrt(x(i,j));
		}
	}
	checkDenseExpressionEquality(cbrt(as_rows(x)),result);
	checkDenseExpressionEquality(cbrt(as_columns(x)),result);
}
BOOST_AUTO_TEST_CASE_TEMPLATE( Remora_vector_set_Exp, Orientation, result_orientations )
{
	matrix<double, Orientation> x(Dimension1, Dimension2); 
	matrix<double> result(Dimension1, Dimension2);
	
	for (size_t i = 0; i < Dimension1; i++){
		for (size_t j = 0; j < Dimension2; j++){
			x(i,j) = 0.01*(i-3.0-double(j));
			result(i,j)= std::exp(x(i,j));
		}
	}
	checkDenseExpressionEquality(exp(as_rows(x)),result);
	checkDenseExpressionEquality(exp(as_columns(x)),result);
}
BOOST_AUTO_TEST_CASE_TEMPLATE( Remora_vector_set_Log, Orientation, result_orientations )
{
	matrix<double, Orientation> x(Dimension1, Dimension2); 
	matrix<double> result(Dimension1, Dimension2);
	
	for (size_t i = 0; i < Dimension1; i++){
		for (size_t j = 0; j < Dimension2; j++){
			x(i,j) = i+j+3.0;
			result(i,j)= std::log(x(i,j));
		}
	}
	checkDenseExpressionEquality(log(as_rows(x)),result);
	checkDenseExpressionEquality(log(as_columns(x)),result);
}
BOOST_AUTO_TEST_CASE_TEMPLATE( Remora_vector_set_sin, Orientation, result_orientations )
{
	matrix<double, Orientation> x(Dimension1, Dimension2); 
	matrix<double> result(Dimension1, Dimension2);
	
	for (size_t i = 0; i < Dimension1; i++){
		for (size_t j = 0; j < Dimension2; j++){
			x(i,j) = double(i)/Dimension1-double(j)/Dimension2;
			result(i,j)= std::sin(x(i,j));
		}
	}
	checkDenseExpressionEquality(sin(as_rows(x)),result);
	checkDenseExpressionEquality(sin(as_columns(x)),result);
}

BOOST_AUTO_TEST_CASE_TEMPLATE( Remora_vector_set_cos, Orientation, result_orientations )
{
	matrix<double, Orientation> x(Dimension1, Dimension2); 
	matrix<double> result(Dimension1, Dimension2);
	
	for (size_t i = 0; i < Dimension1; i++){
		for (size_t j = 0; j < Dimension2; j++){
			x(i,j) = double(i)/Dimension1-double(j)/Dimension2;
			result(i,j)= std::cos(x(i,j));
		}
	}
	checkDenseExpressionEquality(cos(as_rows(x)),result);
	checkDenseExpressionEquality(cos(as_columns(x)),result);
}

BOOST_AUTO_TEST_CASE_TEMPLATE( Remora_vector_set_tan, Orientation, result_orientations )
{
	matrix<double, Orientation> x(Dimension1, Dimension2); 
	matrix<double> result(Dimension1, Dimension2);
	
	for (size_t i = 0; i < Dimension1; i++){
		for (size_t j = 0; j < Dimension2; j++){
			x(i,j) = double(i)/Dimension1-double(j)/Dimension2;
			result(i,j)= std::tan(x(i,j));
		}
	}
	checkDenseExpressionEquality(tan(as_rows(x)),result);
	checkDenseExpressionEquality(tan(as_columns(x)),result);
}

BOOST_AUTO_TEST_CASE_TEMPLATE( Remora_vector_set_asin, Orientation, result_orientations )
{
	matrix<double, Orientation> x(Dimension1, Dimension2); 
	matrix<double> result(Dimension1, Dimension2);
	
	for (size_t i = 0; i < Dimension1; i++){
		for (size_t j = 0; j < Dimension2; j++){
			x(i,j) = double(i)/Dimension1-double(j)/Dimension2;
			result(i,j)= std::asin(x(i,j));
		}
	}
	checkDenseExpressionEquality(asin(as_rows(x)),result);
	checkDenseExpressionEquality(asin(as_columns(x)),result);
}

BOOST_AUTO_TEST_CASE_TEMPLATE( Remora_vector_set_acos, Orientation, result_orientations )
{
	matrix<double, Orientation> x(Dimension1, Dimension2); 
	matrix<double> result(Dimension1, Dimension2);
	
	for (size_t i = 0; i < Dimension1; i++){
		for (size_t j = 0; j < Dimension2; j++){
			x(i,j) = double(i)/Dimension1-double(j)/Dimension2;
			result(i,j)= std::acos(x(i,j));
		}
	}
	checkDenseExpressionEquality(acos(as_rows(x)),result);
	checkDenseExpressionEquality(acos(as_columns(x)),result);
}

BOOST_AUTO_TEST_CASE_TEMPLATE( Remora_vector_set_atan, Orientation, result_orientations )
{
	matrix<double, Orientation> x(Dimension1, Dimension2); 
	matrix<double> result(Dimension1, Dimension2);
	
	for (size_t i = 0; i < Dimension1; i++){
		for (size_t j = 0; j < Dimension2; j++){
			x(i,j) = double(i)/Dimension1-double(j)/Dimension2;
			result(i,j)= std::atan(x(i,j));
		}
	}
	checkDenseExpressionEquality(atan(as_rows(x)),result);
	checkDenseExpressionEquality(atan(as_columns(x)),result);
}

BOOST_AUTO_TEST_CASE_TEMPLATE( Remora_vector_set_erf, Orientation, result_orientations )
{
	matrix<double, Orientation> x(Dimension1, Dimension2); 
	matrix<double> result(Dimension1, Dimension2);
	
	for (size_t i = 0; i < Dimension1; i++){
		for (size_t j = 0; j < Dimension2; j++){
			x(i,j) = double(i)/Dimension1-double(j)/Dimension2;
			result(i,j)= std::erf(x(i,j));
		}
	}
	checkDenseExpressionEquality(erf(as_rows(x)),result);
	checkDenseExpressionEquality(erf(as_columns(x)),result);
}

BOOST_AUTO_TEST_CASE_TEMPLATE( Remora_vector_set_erfc, Orientation, result_orientations )
{
	matrix<double, Orientation> x(Dimension1, Dimension2); 
	matrix<double> result(Dimension1, Dimension2);
	
	for (size_t i = 0; i < Dimension1; i++){
		for (size_t j = 0; j < Dimension2; j++){
			x(i,j) = double(i)/Dimension1-double(j)/Dimension2;
			result(i,j)= std::erfc(x(i,j));
		}
	}
	checkDenseExpressionEquality(erfc(as_rows(x)),result);
	checkDenseExpressionEquality(erfc(as_columns(x)),result);
}

BOOST_AUTO_TEST_CASE_TEMPLATE( Remora_vector_set_Tanh, Orientation, result_orientations )
{
	matrix<double, Orientation> x(Dimension1, Dimension2); 
	matrix<double> result(Dimension1, Dimension2);
	
	for (size_t i = 0; i < Dimension1; i++){
		for (size_t j = 0; j < Dimension2; j++){
			x(i,j) = i-3.0-j;
			result(i,j)= std::tanh(x(i,j));
		}
	}
	checkDenseExpressionEquality(tanh(as_rows(x)),result);
	checkDenseExpressionEquality(tanh(as_columns(x)),result);
}
BOOST_AUTO_TEST_CASE_TEMPLATE( Remora_vector_set_Sigmoid, Orientation, result_orientations )
{
	matrix<double, Orientation> x(Dimension1, Dimension2); 
	matrix<double> result(Dimension1, Dimension2);
	
	for (size_t i = 0; i < Dimension1; i++){
		for (size_t j = 0; j < Dimension2; j++){
			x(i,j) = 0.01*(i-3.0-j);
			result(i,j)= 1.0/(1.0+std::exp(-x(i,j)));
		}
	}
	checkDenseExpressionEquality(sigmoid(as_rows(x)),result);
	checkDenseExpressionEquality(sigmoid(as_columns(x)),result);
}
BOOST_AUTO_TEST_CASE_TEMPLATE( Remora_vector_set_SoftPlus, Orientation, result_orientations )
{
	matrix<double, Orientation> x(Dimension1, Dimension2); 
	matrix<double> result(Dimension1, Dimension2);
	
	for (size_t i = 0; i < Dimension1; i++){
		for (size_t j = 0; j < Dimension2; j++){
			x(i,j) = i-3.0-j;
			result(i,j)= std::log(1+std::exp(x(i,j)));
		}
	}
	checkDenseExpressionEquality(softPlus(as_rows(x)),result);
	checkDenseExpressionEquality(softPlus(as_columns(x)),result);
}


/////////////////////////////////////////////////////////////
//////SCALAR TRANSFORMATIONS///////
////////////////////////////////////////////////////////////
//~ BOOST_AUTO_TEST_CASE_TEMPLATE( Remora_vector_set_Scalar_Multiply, Orientation, result_orientations )
//~ {
	//~ matrix<double, Orientation> x(Dimension1, Dimension2); 
	//~ matrix<double> result(Dimension1, Dimension2);
	
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
	//~ matrix<double, Orientation> x(Dimension1, Dimension2); 
	//~ matrix<double> result(Dimension1, Dimension2);
	
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
	//~ matrix<double, Orientation> x(Dimension1, Dimension2); 
	//~ matrix<double> result1(Dimension1, Dimension2);
	//~ matrix<double> result2(Dimension1, Dimension2);
	
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
	//~ matrix<double, Orientation> x(Dimension1, Dimension2); 
	//~ matrix<double> result(Dimension1, Dimension2);
	
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
	//~ matrix<double, Orientation> x(Dimension1, Dimension2); 
	//~ matrix<double> result(Dimension1, Dimension2);
	
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
	//~ matrix<double, Orientation> x(Dimension1, Dimension2); 
	//~ matrix<double> result(Dimension1, Dimension2);
	
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
	//~ matrix<double, Orientation> x(Dimension1, Dimension2); 
	//~ matrix<double> result(Dimension1, Dimension2);
	
	//~ for (size_t i = 0; i < Dimension1; i++){
		//~ for (size_t j = 0; j < Dimension2; j++){
			//~ x(i,j) = i-1.0*j;
			//~ result(i,j)= std::max(x(i,j),5.0);
		//~ }
	//~ }
	//~ checkDenseExpressionEquality(max(x,5.0),result);
	//~ checkDenseExpressionEquality(max(5.0,x),result);
//~ }

/////////////////////////////////////////////////////
///////BINARY OPERATIONS//////////
/////////////////////////////////////////////////////

//~ BOOST_AUTO_TEST_CASE( Remora_vector_set_Binary_Plus)
//~ {
	//~ matrix<double> x(Dimension1, Dimension2); 
	//~ matrix<double> y(Dimension1, Dimension2); 
	//~ matrix<double> result(Dimension1, Dimension2);
	
	//~ for (size_t i = 0; i < Dimension1; i++){
		//~ for (size_t j = 0; j < Dimension2; j++){
			//~ x(i,j) = i-3.0-j;
			//~ y(i,j) = i+j+Dimension1;
			//~ result(i,j)= x(i,j)+y(i,j);
		//~ }
	//~ }
	//~ checkDenseExpressionEquality(x+y,result);
//~ }
//~ BOOST_AUTO_TEST_CASE_TEMPLATE( Remora_vector_set_Binary_Minus, Orientation, result_orientations )
//~ {
	//~ matrix<double, Orientation> x(Dimension1, Dimension2); 
	//~ matrix<double, Orientation> y(Dimension1, Dimension2); 
	//~ matrix<double> result(Dimension1, Dimension2);
	
	//~ for (size_t i = 0; i < Dimension1; i++){
		//~ for (size_t j = 0; j < Dimension2; j++){
			//~ x(i,j) = i-3.0-j;
			//~ y(i,j) = i+j+Dimension1;
			//~ result(i,j)= x(i,j)-y(i,j);
		//~ }
	//~ }
	//~ checkDenseExpressionEquality(x-y,result);
//~ }

//~ BOOST_AUTO_TEST_CASE_TEMPLATE( Remora_vector_set_Binary_Multiply, Orientation, result_orientations )
//~ {
	//~ matrix<double, Orientation> x(Dimension1, Dimension2); 
	//~ matrix<double, Orientation> y(Dimension1, Dimension2); 
	//~ matrix<double> result(Dimension1, Dimension2);
	
	//~ for (size_t i = 0; i < Dimension1; i++){
		//~ for (size_t j = 0; j < Dimension2; j++){
			//~ x(i,j) = i-3.0-j;
			//~ y(i,j) = i+j+Dimension1;
			//~ result(i,j)= x(i,j)*y(i,j);
		//~ }
	//~ }
	//~ checkDenseExpressionEquality(x*y,result);
	//~ checkDenseExpressionEquality(element_prod(x,y),result);
//~ }

//~ BOOST_AUTO_TEST_CASE_TEMPLATE( Remora_vector_set_Binary_Div, Orientation, result_orientations )
//~ {
	//~ matrix<double, Orientation> x(Dimension1, Dimension2); 
	//~ matrix<double, Orientation> y(Dimension1, Dimension2); 
	//~ matrix<double> result(Dimension1, Dimension2);
	
	//~ for (size_t i = 0; i < Dimension1; i++){
		//~ for (size_t j = 0; j < Dimension2; j++){
			//~ x(i,j) = i-3.0-j;
			//~ y(i,j) = i+j+1;
			//~ result(i,j)= x(i,j)/y(i,j);
		//~ }
	//~ }
	//~ checkDenseExpressionEquality(x/y,result);
	//~ checkDenseExpressionEquality(element_div(x,y),result);
//~ }

//~ BOOST_AUTO_TEST_CASE_TEMPLATE( Remora_vector_set_Safe_Div, Orientation, result_orientations )
//~ {
	//~ matrix<double, Orientation> x(Dimension1, Dimension2); 
	//~ matrix<double, Orientation> y(Dimension1, Dimension2); 
	//~ matrix<double> result(Dimension1, Dimension2);
	//~ for (size_t i = 0; i < Dimension1; i++){
		//~ for (size_t j = 0; j < Dimension2; j++){
			//~ x(i,j) = i+j;
			//~ y(i,j) = (i+j)%3;
			//~ result(i,j)= ((i+j) % 3 == 0)? 2.0: x(i,j)/y(i,j);
		//~ }
	//~ }
	//~ checkDenseExpressionEquality(safe_div(x,y,2.0),result);
//~ }

//~ BOOST_AUTO_TEST_CASE_TEMPLATE( Remora_vector_set_Binary_Pow, Orientation, result_orientations )
//~ {
	//~ matrix<double, Orientation> x(Dimension1, Dimension2); 
	//~ matrix<double, Orientation> y(Dimension1, Dimension2); 
	//~ matrix<double> result(Dimension1, Dimension2);
	
	//~ for (size_t i = 0; i < Dimension1; i++){
		//~ for (size_t j = 0; j < Dimension2; j++){
			//~ x(i,j) = 0.1*(i+j+1);
			//~ y(i,j) = 0.0001*(i+j-3);
			//~ result(i,j)= std::pow(x(i,j),y(i,j));
		//~ }
	//~ }
	//~ checkDenseExpressionEquality(pow(x,y),result);
//~ }

//~ BOOST_AUTO_TEST_CASE_TEMPLATE( Remora_vector_set_Binary_Max, Orientation, result_orientations )
//~ {
	//~ matrix<double, Orientation> x(Dimension1, Dimension2); 
	//~ matrix<double, Orientation> y(Dimension1, Dimension2); 
	//~ matrix<double> result(Dimension1, Dimension2);
	
	//~ for (size_t i = 0; i < Dimension1; i++){
		//~ for (size_t j = 0; j < Dimension2; j++){
			//~ x(i,j) = 50.0+i-j;
			//~ y(i,j) = i+j+1;
			//~ result(i,j)= std::max(x(i,j),y(i,j));
		//~ }
	//~ }
	//~ checkDenseExpressionEquality(max(x,y),result);
//~ }
//~ BOOST_AUTO_TEST_CASE_TEMPLATE( Remora_vector_set_Binary_Min, Orientation, result_orientations )
//~ {
	//~ matrix<double, Orientation> x(Dimension1, Dimension2); 
	//~ matrix<double, Orientation> y(Dimension1, Dimension2); 
	//~ matrix<double> result(Dimension1, Dimension2);
	
	//~ for (size_t i = 0; i < Dimension1; i++){
		//~ for (size_t j = 0; j < Dimension2; j++){
			//~ x(i,j) = 50.0+i-j;
			//~ y(i,j) = i+j+1;
			//~ result(i,j)= std::min(x(i,j),y(i,j));
		//~ }
	//~ }
	//~ checkDenseExpressionEquality(min(x,y),result);
//~ }


BOOST_AUTO_TEST_SUITE_END()
