#define BOOST_TEST_MODULE Remora_Matrix_Expression
#include <boost/test/unit_test.hpp>
#include <boost/test/floating_point_comparison.hpp>


#include <remora/matrix_expression.hpp>
#include <remora/dense.hpp>
#include <remora/sparse.hpp>
#include <boost/mpl/list.hpp>

using namespace remora;


template<class Operation, class Result>
void checkDenseBlockAssign(
	vector_expression<Operation, cpu_tag> const& op, Result const& result
){
	remora::vector<typename Result::value_type> res1(result.size(),1.0);
	remora::vector<typename Result::value_type> res2(result.size(),1.0);
	op().assign_to(res1, op);
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
	
	//check that elements() works
	auto op_elem = op().elements();
	for(std::size_t i = 0; i != op().size1(); ++i){
		for(std::size_t j = 0; j != op().size2(); ++j){
			BOOST_CHECK_CLOSE(result(i,j), op_elem(i,j),1.e-5);
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
	vector_expression<Operation, cpu_tag> const& op, Result const& result
){
	BOOST_REQUIRE_EQUAL(op().size(), result.size());
	
	typename Operation::const_iterator pos = op().begin();
	auto op_elem = op().elements();
	for(std::size_t i = 0; i != op().size(); ++i,++pos){
		BOOST_REQUIRE(pos != op().end());
		BOOST_CHECK_EQUAL(pos.index(), i);
		BOOST_CHECK_SMALL(result(i) - op_elem(i),typename Result::value_type(1.e-10));
		BOOST_CHECK_SMALL(*pos - op_elem(i),typename Result::value_type(1.e-10));
	}
	BOOST_REQUIRE(pos == op().end());

	checkDenseBlockAssign(op,result);
	
}

template<class M, class D>
void checkDiagonalMatrix(M const& diagonal, D const& diagonalElements){
	BOOST_REQUIRE_EQUAL(diagonal.size1(),diagonalElements.size());
	BOOST_REQUIRE_EQUAL(diagonal.size2(),diagonalElements.size());
	auto diag_elem = diagonal.elements();
	for(std::size_t i = 0; i != diagonalElements.size(); ++i){
		for(std::size_t j = 0; j != diagonalElements.size(); ++j){
			if(i != j)
				BOOST_CHECK_EQUAL(diag_elem(i,j),0);
			else
				BOOST_CHECK_EQUAL(diag_elem(i,i),diagonalElements(i));
		}
		auto major_begin = diagonal.major_begin(i);
		BOOST_CHECK_EQUAL(major_begin.index(),i);
		BOOST_CHECK_EQUAL(std::distance(major_begin, diagonal.major_end(i)),1);
		BOOST_CHECK_EQUAL(*major_begin,diagonalElements(i));
	}
	
}

std::size_t Dimension1 = 50;
std::size_t Dimension2 = 100;

BOOST_AUTO_TEST_SUITE (Remora_matrix_expression)

/////////////////////////////////////////////////////////////
//////Vector->Matrix expansions///////
////////////////////////////////////////////////////////////

BOOST_AUTO_TEST_CASE( Remora_matrix_Outer_Prod ){
	std::size_t Dimension2 = 50;
	vector<double> x(Dimension1); 
	vector<double> y(Dimension2); 
	matrix<double> result(Dimension1, Dimension2);
	matrix<double> result2(Dimension1, Dimension2);
	
	
	for (size_t i = 0; i < Dimension1; i++)
		x(i) = i-3.0;
	for (size_t j = 0; j < Dimension2; j++)
		y(j) = 2*j;
	
	for (size_t i = 0; i < Dimension1; i++){
		for (size_t j = 0; j < Dimension2; j++){
			result(i,j)= x(i)*y(j);
			result2(i,j)= 2.0*x(i)*y(j) +6*result(i,j);
		}
	}

	checkDenseExpressionEquality(outer_prod(x,y),result);
	checkDenseExpressionEquality(2.0 * ( outer_prod( x, y ) + 3.0 * result),result2);
}

BOOST_AUTO_TEST_CASE( Remora_matrix_Vector_Repeater){
	vector<double> x(Dimension2); 
	matrix<double> result(Dimension1, Dimension2);
	
	for (size_t i = 0; i < Dimension2; i++)
		x(i) = i-3.0;
	
	for (size_t i = 0; i < Dimension1; i++){
		for (size_t j = 0; j < Dimension2; j++){
			result(i,j)= x(j);
		}
	}
	checkDenseExpressionEquality(repeat(x,Dimension1),result);
}

BOOST_AUTO_TEST_CASE( Remora_Diagonal_Matrix ){
	vector<double> diagonalElements(Dimension1);
	matrix<double> result(Dimension1,Dimension2,0.0);
	for(std::size_t i = 0; i != Dimension1; ++i){
		diagonalElements(i) = i;
		result(i,i) = i;
	}
	
	diagonal_matrix<vector<double> > diagonal(diagonalElements);
	checkDiagonalMatrix(diagonal,diagonalElements);
}

BOOST_AUTO_TEST_CASE( Remora_Identity_Matrix ){
	vector<double> diagonalElements(Dimension1);
	for(std::size_t i = 0; i != Dimension1; ++i)
		diagonalElements(i) = 1;
	
	identity_matrix<double > diagonal(Dimension1);
	checkDiagonalMatrix(diagonal,diagonalElements);
}

/////////////////////////////////////////////////////////////
//////UNARY TRANSFORMATIONS///////
////////////////////////////////////////////////////////////
typedef boost::mpl::list<row_major,column_major> result_orientations;
BOOST_AUTO_TEST_CASE_TEMPLATE( Remora_matrix_Unary_Minus, Orientation, result_orientations )
{
	matrix<double, Orientation> x(Dimension1, Dimension2); 
	matrix<double> result(Dimension1, Dimension2);
	
	for (size_t i = 0; i < Dimension1; i++){
		for (size_t j = 0; j < Dimension2; j++){
			x(i,j) = i-3.0+j;
			result(i,j)= -x(i,j);
		}
	}
	checkDenseExpressionEquality(-x,result);
}
BOOST_AUTO_TEST_CASE_TEMPLATE( Remora_matrix_Scalar_Multiply, Orientation, result_orientations )
{
	matrix<double, Orientation> x(Dimension1, Dimension2); 
	matrix<double> result(Dimension1, Dimension2);
	
	for (size_t i = 0; i < Dimension1; i++){
		for (size_t j = 0; j < Dimension2; j++){
			x(i,j) = i-3.0+j;
			result(i,j)= 5.0* x(i,j);
		}
	}
	checkDenseExpressionEquality(5.0*x,result);
	checkDenseExpressionEquality(x*5.0,result);
}
BOOST_AUTO_TEST_CASE_TEMPLATE( Remora_matrix_Scalar_Add, Orientation, result_orientations )
{
	matrix<double, Orientation> x(Dimension1, Dimension2); 
	matrix<double> result(Dimension1, Dimension2);
	
	for (size_t i = 0; i < Dimension1; i++){
		for (size_t j = 0; j < Dimension2; j++){
			x(i,j) = i-3.0+j;
			result(i,j)= 5.0 + x(i,j);
		}
	}
	checkDenseExpressionEquality(5.0+x,result);
	checkDenseExpressionEquality(x+5.0,result);
}
BOOST_AUTO_TEST_CASE_TEMPLATE( Remora_matrix_Scalar_Subtract, Orientation, result_orientations )
{
	matrix<double, Orientation> x(Dimension1, Dimension2); 
	matrix<double> result1(Dimension1, Dimension2);
	matrix<double> result2(Dimension1, Dimension2);
	
	for (size_t i = 0; i < Dimension1; i++){
		for (size_t j = 0; j < Dimension2; j++){
			x(i,j) = i-3.0+j;
			result1(i,j)= 5.0 - x(i,j);
			result2(i,j)= x(i,j) - 5.0;
		}
	}
	checkDenseExpressionEquality(5.0- x,result1);
	checkDenseExpressionEquality(x - 5.0,result2);
}
BOOST_AUTO_TEST_CASE_TEMPLATE( Remora_matrix_Scalar_Div, Orientation, result_orientations )
{
	matrix<double, Orientation> x(Dimension1, Dimension2); 
	matrix<double> result(Dimension1, Dimension2);
	
	for (size_t i = 0; i < Dimension1; i++){
		for (size_t j = 0; j < Dimension2; j++){
			x(i,j) = i-3.0+j;
			result(i,j)= x(i,j)/5.0;
		}
	}
	checkDenseExpressionEquality(x/5.0f,result);
}
BOOST_AUTO_TEST_CASE_TEMPLATE( Remora_matrix_Scalar_elem_inv, Orientation, result_orientations )
{
	matrix<double, Orientation> x(Dimension1, Dimension2); 
	matrix<double> result(Dimension1, Dimension2);
	
	for (size_t i = 0; i < Dimension1; i++){
		for (size_t j = 0; j < Dimension2; j++){
			x(i,j) = i+j+3.0;
			result(i,j)= 1.0/x(i,j);
		}
	}
	checkDenseExpressionEquality(elem_inv(x),result);
}
BOOST_AUTO_TEST_CASE_TEMPLATE( Remora_matrix_Abs, Orientation, result_orientations )
{
	matrix<double, Orientation> x(Dimension1, Dimension2); 
	matrix<double> result(Dimension1, Dimension2);
	
	for (size_t i = 0; i < Dimension1; i++){
		for (size_t j = 0; j < Dimension2; j++){
			x(i,j) = i-3.0-j;
			result(i,j)= std::abs(x(i,j));
		}
	}
	checkDenseExpressionEquality(abs(x),result);
}
BOOST_AUTO_TEST_CASE_TEMPLATE( Remora_matrix_Sqr, Orientation, result_orientations )
{
	matrix<double, Orientation> x(Dimension1, Dimension2); 
	matrix<double> result(Dimension1, Dimension2);
	for (size_t i = 0; i < Dimension1; i++){
		for (size_t j = 0; j < Dimension2; j++){
			x(i,j) = i-3.0-j;
			result(i,j)= x(i,j) * x(i,j);
		}
	}
	checkDenseExpressionEquality(sqr(x),result);
}
BOOST_AUTO_TEST_CASE_TEMPLATE( Remora_matrix_Sqrt, Orientation, result_orientations )
{
	matrix<double, Orientation> x(Dimension1, Dimension2); 
	matrix<double> result(Dimension1, Dimension2);
	
	for (size_t i = 0; i < Dimension1; i++){
		for (size_t j = 0; j < Dimension2; j++){
			x(i,j) = i+j+3.0;
			result(i,j)= std::sqrt(x(i,j));
		}
	}
	checkDenseExpressionEquality(sqrt(x),result);
}
BOOST_AUTO_TEST_CASE_TEMPLATE( Remora_matrix_Cbrt, Orientation, result_orientations )
{
	matrix<double, Orientation> x(Dimension1, Dimension2); 
	matrix<double> result(Dimension1, Dimension2);
	
	for (size_t i = 0; i < Dimension1; i++){
		for (size_t j = 0; j < Dimension2; j++){
			x(i,j) = i+j+3.0;
			result(i,j)= std::cbrt(x(i,j));
		}
	}
	checkDenseExpressionEquality(cbrt(x),result);
}
BOOST_AUTO_TEST_CASE_TEMPLATE( Remora_matrix_Exp, Orientation, result_orientations )
{
	matrix<double, Orientation> x(Dimension1, Dimension2); 
	matrix<double> result(Dimension1, Dimension2);
	
	for (size_t i = 0; i < Dimension1; i++){
		for (size_t j = 0; j < Dimension2; j++){
			x(i,j) = 0.01*(i-3.0-double(j));
			result(i,j)= std::exp(x(i,j));
		}
	}
	checkDenseExpressionEquality(exp(x),result);
}
BOOST_AUTO_TEST_CASE_TEMPLATE( Remora_matrix_Log, Orientation, result_orientations )
{
	matrix<double, Orientation> x(Dimension1, Dimension2); 
	matrix<double> result(Dimension1, Dimension2);
	
	for (size_t i = 0; i < Dimension1; i++){
		for (size_t j = 0; j < Dimension2; j++){
			x(i,j) = i+j+3.0;
			result(i,j)= std::log(x(i,j));
		}
	}
	checkDenseExpressionEquality(log(x),result);
}
BOOST_AUTO_TEST_CASE_TEMPLATE( Remora_matrix_sin, Orientation, result_orientations )
{
	matrix<double, Orientation> x(Dimension1, Dimension2); 
	matrix<double> result(Dimension1, Dimension2);
	
	for (size_t i = 0; i < Dimension1; i++){
		for (size_t j = 0; j < Dimension2; j++){
			x(i,j) = double(i)/Dimension1-double(j)/Dimension2;
			result(i,j)= std::sin(x(i,j));
		}
	}
	checkDenseExpressionEquality(sin(x),result);
}

BOOST_AUTO_TEST_CASE_TEMPLATE( Remora_matrix_cos, Orientation, result_orientations )
{
	matrix<double, Orientation> x(Dimension1, Dimension2); 
	matrix<double> result(Dimension1, Dimension2);
	
	for (size_t i = 0; i < Dimension1; i++){
		for (size_t j = 0; j < Dimension2; j++){
			x(i,j) = double(i)/Dimension1-double(j)/Dimension2;
			result(i,j)= std::cos(x(i,j));
		}
	}
	checkDenseExpressionEquality(cos(x),result);
}

BOOST_AUTO_TEST_CASE_TEMPLATE( Remora_matrix_tan, Orientation, result_orientations )
{
	matrix<double, Orientation> x(Dimension1, Dimension2); 
	matrix<double> result(Dimension1, Dimension2);
	
	for (size_t i = 0; i < Dimension1; i++){
		for (size_t j = 0; j < Dimension2; j++){
			x(i,j) = double(i)/Dimension1-double(j)/Dimension2;
			result(i,j)= std::tan(x(i,j));
		}
	}
	checkDenseExpressionEquality(tan(x),result);
}

BOOST_AUTO_TEST_CASE_TEMPLATE( Remora_matrix_asin, Orientation, result_orientations )
{
	matrix<double, Orientation> x(Dimension1, Dimension2); 
	matrix<double> result(Dimension1, Dimension2);
	
	for (size_t i = 0; i < Dimension1; i++){
		for (size_t j = 0; j < Dimension2; j++){
			x(i,j) = double(i)/Dimension1-double(j)/Dimension2;
			result(i,j)= std::asin(x(i,j));
		}
	}
	checkDenseExpressionEquality(asin(x),result);
}

BOOST_AUTO_TEST_CASE_TEMPLATE( Remora_matrix_acos, Orientation, result_orientations )
{
	matrix<double, Orientation> x(Dimension1, Dimension2); 
	matrix<double> result(Dimension1, Dimension2);
	
	for (size_t i = 0; i < Dimension1; i++){
		for (size_t j = 0; j < Dimension2; j++){
			x(i,j) = double(i)/Dimension1-double(j)/Dimension2;
			result(i,j)= std::acos(x(i,j));
		}
	}
	checkDenseExpressionEquality(acos(x),result);
}

BOOST_AUTO_TEST_CASE_TEMPLATE( Remora_matrix_atan, Orientation, result_orientations )
{
	matrix<double, Orientation> x(Dimension1, Dimension2); 
	matrix<double> result(Dimension1, Dimension2);
	
	for (size_t i = 0; i < Dimension1; i++){
		for (size_t j = 0; j < Dimension2; j++){
			x(i,j) = double(i)/Dimension1-double(j)/Dimension2;
			result(i,j)= std::atan(x(i,j));
		}
	}
	checkDenseExpressionEquality(atan(x),result);
}

BOOST_AUTO_TEST_CASE_TEMPLATE( Remora_matrix_erf, Orientation, result_orientations )
{
	matrix<double, Orientation> x(Dimension1, Dimension2); 
	matrix<double> result(Dimension1, Dimension2);
	
	for (size_t i = 0; i < Dimension1; i++){
		for (size_t j = 0; j < Dimension2; j++){
			x(i,j) = double(i)/Dimension1-double(j)/Dimension2;
			result(i,j)= std::erf(x(i,j));
		}
	}
	checkDenseExpressionEquality(erf(x),result);
}

BOOST_AUTO_TEST_CASE_TEMPLATE( Remora_matrix_erfc, Orientation, result_orientations )
{
	matrix<double, Orientation> x(Dimension1, Dimension2); 
	matrix<double> result(Dimension1, Dimension2);
	
	for (size_t i = 0; i < Dimension1; i++){
		for (size_t j = 0; j < Dimension2; j++){
			x(i,j) = double(i)/Dimension1-double(j)/Dimension2;
			result(i,j)= std::erfc(x(i,j));
		}
	}
	checkDenseExpressionEquality(erfc(x),result);
}

BOOST_AUTO_TEST_CASE_TEMPLATE( Remora_matrix_Tanh, Orientation, result_orientations )
{
	matrix<double, Orientation> x(Dimension1, Dimension2); 
	matrix<double> result(Dimension1, Dimension2);
	
	for (size_t i = 0; i < Dimension1; i++){
		for (size_t j = 0; j < Dimension2; j++){
			x(i,j) = i-3.0-j;
			result(i,j)= std::tanh(x(i,j));
		}
	}
	checkDenseExpressionEquality(tanh(x),result);
}
BOOST_AUTO_TEST_CASE_TEMPLATE( Remora_matrix_Sigmoid, Orientation, result_orientations )
{
	matrix<double, Orientation> x(Dimension1, Dimension2); 
	matrix<double> result(Dimension1, Dimension2);
	
	for (size_t i = 0; i < Dimension1; i++){
		for (size_t j = 0; j < Dimension2; j++){
			x(i,j) = 0.01*(i-3.0-j);
			result(i,j)= 1.0/(1.0+std::exp(-x(i,j)));
		}
	}
	checkDenseExpressionEquality(sigmoid(x),result);
}
BOOST_AUTO_TEST_CASE_TEMPLATE( Remora_matrix_SoftPlus, Orientation, result_orientations )
{
	matrix<double, Orientation> x(Dimension1, Dimension2); 
	matrix<double> result(Dimension1, Dimension2);
	
	for (size_t i = 0; i < Dimension1; i++){
		for (size_t j = 0; j < Dimension2; j++){
			x(i,j) = i-3.0-j;
			result(i,j)= std::log(1+std::exp(x(i,j)));
		}
	}
	checkDenseExpressionEquality(softPlus(x),result);
}
BOOST_AUTO_TEST_CASE_TEMPLATE( Remora_matrix_Pow, Orientation, result_orientations )
{
	matrix<double, Orientation> x(Dimension1, Dimension2); 
	matrix<double> result(Dimension1, Dimension2);
	
	for (size_t i = 0; i < Dimension1; i++){
		for (size_t j = 0; j < Dimension2; j++){
			x(i,j) = 0.001*(i+j+2.0);
			result(i,j)= std::pow(x(i,j),3.2);
		}
	}
	checkDenseExpressionEquality(pow(x,3.2),result);
}

BOOST_AUTO_TEST_CASE_TEMPLATE( Remora_matrix_Unary_Min, Orientation, result_orientations )
{
	matrix<double, Orientation> x(Dimension1, Dimension2); 
	matrix<double> result(Dimension1, Dimension2);
	
	for (size_t i = 0; i < Dimension1; i++){
		for (size_t j = 0; j < Dimension2; j++){
			x(i,j) = i-1.0*j;
			result(i,j)= std::min(x(i,j),5.0);
		}
	}
	checkDenseExpressionEquality(min(x,5.0),result);
	checkDenseExpressionEquality(min(5.0,x),result);
}
BOOST_AUTO_TEST_CASE_TEMPLATE( Remora_matrix_Unary_Max, Orientation, result_orientations )
{
	matrix<double, Orientation> x(Dimension1, Dimension2); 
	matrix<double> result(Dimension1, Dimension2);
	
	for (size_t i = 0; i < Dimension1; i++){
		for (size_t j = 0; j < Dimension2; j++){
			x(i,j) = i-1.0*j;
			result(i,j)= std::max(x(i,j),5.0);
		}
	}
	checkDenseExpressionEquality(max(x,5.0),result);
	checkDenseExpressionEquality(max(5.0,x),result);
}

/////////////////////////////////////////////////////
///////BINARY OPERATIONS//////////
/////////////////////////////////////////////////////

BOOST_AUTO_TEST_CASE( Remora_matrix_Binary_Plus)
{
	matrix<double> x(Dimension1, Dimension2); 
	matrix<double> y(Dimension1, Dimension2); 
	matrix<double> result(Dimension1, Dimension2);
	
	for (size_t i = 0; i < Dimension1; i++){
		for (size_t j = 0; j < Dimension2; j++){
			x(i,j) = i-3.0-j;
			y(i,j) = i+j+Dimension1;
			result(i,j)= x(i,j)+y(i,j);
		}
	}
	checkDenseExpressionEquality(x+y,result);
}
BOOST_AUTO_TEST_CASE_TEMPLATE( Remora_matrix_Binary_Minus, Orientation, result_orientations )
{
	matrix<double, Orientation> x(Dimension1, Dimension2); 
	matrix<double, Orientation> y(Dimension1, Dimension2); 
	matrix<double> result(Dimension1, Dimension2);
	
	for (size_t i = 0; i < Dimension1; i++){
		for (size_t j = 0; j < Dimension2; j++){
			x(i,j) = i-3.0-j;
			y(i,j) = i+j+Dimension1;
			result(i,j)= x(i,j)-y(i,j);
		}
	}
	checkDenseExpressionEquality(x-y,result);
}

BOOST_AUTO_TEST_CASE_TEMPLATE( Remora_matrix_Binary_Multiply, Orientation, result_orientations )
{
	matrix<double, Orientation> x(Dimension1, Dimension2); 
	matrix<double, Orientation> y(Dimension1, Dimension2); 
	matrix<double> result(Dimension1, Dimension2);
	
	for (size_t i = 0; i < Dimension1; i++){
		for (size_t j = 0; j < Dimension2; j++){
			x(i,j) = i-3.0-j;
			y(i,j) = i+j+Dimension1;
			result(i,j)= x(i,j)*y(i,j);
		}
	}
	checkDenseExpressionEquality(x*y,result);
	checkDenseExpressionEquality(element_prod(x,y),result);
}

BOOST_AUTO_TEST_CASE_TEMPLATE( Remora_matrix_Binary_Div, Orientation, result_orientations )
{
	matrix<double, Orientation> x(Dimension1, Dimension2); 
	matrix<double, Orientation> y(Dimension1, Dimension2); 
	matrix<double> result(Dimension1, Dimension2);
	
	for (size_t i = 0; i < Dimension1; i++){
		for (size_t j = 0; j < Dimension2; j++){
			x(i,j) = i-3.0-j;
			y(i,j) = i+j+1;
			result(i,j)= x(i,j)/y(i,j);
		}
	}
	checkDenseExpressionEquality(x/y,result);
	checkDenseExpressionEquality(element_div(x,y),result);
}

BOOST_AUTO_TEST_CASE_TEMPLATE( Remora_matrix_Safe_Div, Orientation, result_orientations )
{
	matrix<double, Orientation> x(Dimension1, Dimension2); 
	matrix<double, Orientation> y(Dimension1, Dimension2); 
	matrix<double> result(Dimension1, Dimension2);
	for (size_t i = 0; i < Dimension1; i++){
		for (size_t j = 0; j < Dimension2; j++){
			x(i,j) = i+j;
			y(i,j) = (i+j)%3;
			result(i,j)= ((i+j) % 3 == 0)? 2.0: x(i,j)/y(i,j);
		}
	}
	checkDenseExpressionEquality(safe_div(x,y,2.0),result);
}

BOOST_AUTO_TEST_CASE_TEMPLATE( Remora_matrix_Binary_Pow, Orientation, result_orientations )
{
	matrix<double, Orientation> x(Dimension1, Dimension2); 
	matrix<double, Orientation> y(Dimension1, Dimension2); 
	matrix<double> result(Dimension1, Dimension2);
	
	for (size_t i = 0; i < Dimension1; i++){
		for (size_t j = 0; j < Dimension2; j++){
			x(i,j) = 0.1*(i+j+1);
			y(i,j) = 0.0001*(i+j-3);
			result(i,j)= std::pow(x(i,j),y(i,j));
		}
	}
	checkDenseExpressionEquality(pow(x,y),result);
}

BOOST_AUTO_TEST_CASE_TEMPLATE( Remora_matrix_Binary_Max, Orientation, result_orientations )
{
	matrix<double, Orientation> x(Dimension1, Dimension2); 
	matrix<double, Orientation> y(Dimension1, Dimension2); 
	matrix<double> result(Dimension1, Dimension2);
	
	for (size_t i = 0; i < Dimension1; i++){
		for (size_t j = 0; j < Dimension2; j++){
			x(i,j) = 50.0+i-j;
			y(i,j) = i+j+1;
			result(i,j)= std::max(x(i,j),y(i,j));
		}
	}
	checkDenseExpressionEquality(max(x,y),result);
}
BOOST_AUTO_TEST_CASE_TEMPLATE( Remora_matrix_Binary_Min, Orientation, result_orientations )
{
	matrix<double, Orientation> x(Dimension1, Dimension2); 
	matrix<double, Orientation> y(Dimension1, Dimension2); 
	matrix<double> result(Dimension1, Dimension2);
	
	for (size_t i = 0; i < Dimension1; i++){
		for (size_t j = 0; j < Dimension2; j++){
			x(i,j) = 50.0+i-j;
			y(i,j) = i+j+1;
			result(i,j)= std::min(x(i,j),y(i,j));
		}
	}
	checkDenseExpressionEquality(min(x,y),result);
}


/////////////////////////////////////////////////////////////
//////MATRIX CONCATENATION
/////////////////////////////////////////////////////////////

BOOST_AUTO_TEST_CASE_TEMPLATE( Remora_Concat_Matrix_Matrix_Right, Orientation, result_orientations )
{
	matrix<double> x(Dimension1, Dimension2); 
	matrix<double, Orientation> y(Dimension1,2 * Dimension2); 
	matrix<double> result(Dimension1, 3 * Dimension2);
	matrix<double> result2(Dimension1, 3 * Dimension2);
	
	for (size_t i = 0; i < Dimension1; i++){
		for (size_t j = 0; j < Dimension2; j++){
			x(i,j) = 50.0+i-j;
			y(i,j) = i+j+1;
			y(i,j+Dimension2) = i+j+2;
			result(i,j)= x(i,j);
			result(i,j+Dimension2)= y(i,j);
			result(i,j+2*Dimension2)= y(i,j+Dimension2);
			result2(i,j+2*Dimension2)= x(i,j);
			result2(i,j)= y(i,j);
			result2(i,j+Dimension2)= y(i,j+Dimension2);
		}
	}
	
	checkDenseBlockAssign(x|y,result);
	checkDenseBlockAssign(y|x,result2);
}

BOOST_AUTO_TEST_CASE_TEMPLATE( Remora_Concat_Matrix_Matrix_Bottom, Orientation, result_orientations )
{
	matrix<double, Orientation> x(Dimension1, Dimension2); 
	matrix<double> y(2 * Dimension1,Dimension2); 
	matrix<double> result(3 *Dimension1, Dimension2);
	matrix<double> result2(3 *Dimension1, Dimension2);
	
	for (size_t i = 0; i < Dimension1; i++){
		for (size_t j = 0; j < Dimension2; j++){
			x(i,j) = 50.0+i-j;
			y(i,j) = i+j+1;
			y(i+Dimension1, j) = i+j+2;
			result(i,j)= x(i,j);
			result(i + Dimension1, j)= y(i,j);
			result(i + 2 * Dimension1, j)= y(i + Dimension1,j);
			result2(i  + 2 * Dimension1,j)= x(i,j);
			result2(i, j)= y(i,j);
			result2(i + Dimension1, j)= y(i + Dimension1,j);
		}
	}
	checkDenseBlockAssign(x & y,result);
	checkDenseBlockAssign(y & x,result2);
}

BOOST_AUTO_TEST_CASE_TEMPLATE( Remora_Concat_Matrix_Vector_Right, Orientation, result_orientations )
{
	matrix<double, Orientation> x(Dimension1, Dimension2); 
	vector<double> y(Dimension1); 
	matrix<double> result(Dimension1, Dimension2+1);
	matrix<double> result2(Dimension1, Dimension2+1);
	
	for (size_t i = 0; i < Dimension1; i++){
		for (size_t j = 0; j < Dimension2; j++){
			x(i,j) = 50.0+i-j;
			result(i,j)= x(i,j);
			result2(i,j+1)= x(i,j);
		}
		y(i) = i;
		result(i,Dimension2) = y(i);
		result2(i,0) = y(i);
	}
	checkDenseBlockAssign(x | y,result);
	checkDenseBlockAssign(y | x,result2);
}

BOOST_AUTO_TEST_CASE_TEMPLATE( Remora_Concat_Matrix_Vector_Bottom, Orientation, result_orientations )
{
	matrix<double, Orientation> x(Dimension1, Dimension2); 
	vector<double> y(Dimension2); 
	matrix<double> result(Dimension1 + 1, Dimension2);
	matrix<double> result2(Dimension1 + 1, Dimension2);
	
	for (size_t i = 0; i < Dimension1; i++){
		for (size_t j = 0; j < Dimension2; j++){
			x(i,j) = 50.0+i-j;
			result(i,j)= x(i,j);
			result2(i+1,j)= x(i,j);
		}
	}
	for (size_t i = 0; i < Dimension2; i++){
		y(i) = i;
		result(Dimension1,i) = y(i);
		result2(0,i) = y(i);
	}
	checkDenseBlockAssign(x & y,result);
	checkDenseBlockAssign(y & x,result2);
}

BOOST_AUTO_TEST_CASE_TEMPLATE( Remora_Concat_Matrix_Scalar_Left, Orientation, result_orientations )
{
	matrix<double, Orientation> x(Dimension1, Dimension2); 
	double t = 2.0;
	matrix<double> result(Dimension1, Dimension2 + 1);
	matrix<double> result2(Dimension1, Dimension2 + 1);
	
	for (size_t i = 0; i < Dimension1; i++){
		for (size_t j = 0; j < Dimension2; j++){
			x(i,j) = 50.0+i-j;
			result(i,j+1)= x(i,j);
			result2(i,j)= x(i,j);
			result(i,0) = t;
			result2(i,Dimension2) = t;
		}
	}
	checkDenseBlockAssign(t | x,result);
	checkDenseBlockAssign(x | t,result2);
}

BOOST_AUTO_TEST_CASE_TEMPLATE( Remora_Concat_Matrix_Scalar_Top, Orientation, result_orientations )
{
	matrix<double, Orientation> x(Dimension1, Dimension2); 
	double t = 2.0;
	matrix<double> result(Dimension1 + 1, Dimension2);
	matrix<double> result2(Dimension1 + 1, Dimension2);
	
	for (size_t i = 0; i < Dimension1; i++){
		for (size_t j = 0; j < Dimension2; j++){
			x(i,j) = 50.0+i-j;
			result(i + 1,j)= x(i,j);
			result2(i,j)= x(i,j);
			result(0, j) = t;
			result2(Dimension1, j) = t;
		}
	}
	checkDenseBlockAssign(t & x,result);
	checkDenseBlockAssign(x & t,result2);
}



////////////////////////////////////////////////////////////////////////
////////////REDUCTIONS
////////////////////////////////////////////////////////////////////////

BOOST_AUTO_TEST_CASE_TEMPLATE( Remora_trace, Orientation, result_orientations )
{
	matrix<double, Orientation> x(Dimension1, Dimension1); 
	double result = 0.0f;
	for (size_t i = 0; i < Dimension1; i++){
		for (size_t j = 0; j < Dimension1; j++){
			x(i,j) = 2*i-3.0-j;
		}
		result += x(i,i);
	}
	BOOST_CHECK_CLOSE(trace(x),result, 1.e-6);
}

BOOST_AUTO_TEST_CASE_TEMPLATE( Remora_norm_1, Orientation, result_orientations )
{
	matrix<double, Orientation> x(Dimension1, Dimension1); 
	vector<double> col_sum(Dimension2);
	for (size_t i = 0; i < Dimension1; i++){
		for (size_t j = 0; j < Dimension1; j++){
			x(i,j) = 2*i-3.0-j;
			col_sum(j) += std::abs(x(i,j));
		}
	}
	double result = max(col_sum);
	BOOST_CHECK_CLOSE(norm_1(x),result, 1.e-6);
}
BOOST_AUTO_TEST_CASE_TEMPLATE( Remora_norm_inf, Orientation, result_orientations )
{
	matrix<double, Orientation> x(Dimension1, Dimension1); 
	vector<double> row_sum(Dimension2);
	for (size_t i = 0; i < Dimension1; i++){
		for (size_t j = 0; j < Dimension1; j++){
			x(i,j) = 2*i-3.0-j;
			row_sum(i) += std::abs(x(i,j));
		}
	}
	double result = max(row_sum);
	BOOST_CHECK_CLOSE(norm_inf(x),result, 1.e-6);
}

BOOST_AUTO_TEST_CASE_TEMPLATE( Remora_norm_Frobenius, Orientation, result_orientations )
{
	matrix<double, Orientation> x(Dimension1, Dimension1); 
	double result = 0;
	for (size_t i = 0; i < Dimension1; i++){
		for (size_t j = 0; j < Dimension1; j++){
			x(i,j) = 2*i-3.0-j;
			result += x(i,j)*x(i,j);
		}
	}
	result = std::sqrt(result);
	BOOST_CHECK_CLOSE(norm_frobenius(x),result, 1.e-6);
}

BOOST_AUTO_TEST_CASE_TEMPLATE( Remora_sum, Orientation, result_orientations )
{
	matrix<double, Orientation> x(Dimension1, Dimension1); 
	double result = 0;
	for (size_t i = 0; i < Dimension1; i++){
		for (size_t j = 0; j < Dimension1; j++){
			x(i,j) = 2*i-3.0-j;
			result +=x(i,j);
		}
	}
	BOOST_CHECK_CLOSE(sum(x),result, 1.e-6);
}

BOOST_AUTO_TEST_CASE_TEMPLATE( Remora_max, Orientation, result_orientations )
{
	matrix<double, Orientation> x(Dimension1, Dimension1); 
	double result = std::numeric_limits<double>::min();
	for (size_t i = 0; i < Dimension1; i++){
		for (size_t j = 0; j < Dimension1; j++){
			x(i,j) = 2*i-3.0-j;
			result = std::max(x(i,j),result);
		}
	}
	BOOST_CHECK_CLOSE(max(x),result, 1.e-6);
}

BOOST_AUTO_TEST_CASE_TEMPLATE( Remora_min, Orientation, result_orientations )
{
	matrix<double, Orientation> x(Dimension1, Dimension1); 
	double result = std::numeric_limits<double>::max();
	for (size_t i = 0; i < Dimension1; i++){
		for (size_t j = 0; j < Dimension1; j++){
			x(i,j) = 2*i-3.0-j;
			result = std::min(x(i,j),result);
		}
	}
	BOOST_CHECK_CLOSE(min(x),result, 1.e-6);
}

BOOST_AUTO_TEST_CASE_TEMPLATE( Remora_frobenius_prod, Orientation, result_orientations )
{
	matrix<double, Orientation> x(Dimension1, Dimension2); 
	matrix<double> y(Dimension1, Dimension2); 
	double result = 0;
	for (size_t i = 0; i < Dimension1; i++){
		for (size_t j = 0; j < Dimension1; j++){
			x(i,j) = 2*i-3.0-j;
			y(i,j) = i+j+1;
			result +=x(i,j)*y(i,j);
		}
	}
	BOOST_CHECK_CLOSE(frobenius_prod(x,y),result, 1.e-6);
	BOOST_CHECK_CLOSE(frobenius_prod(y,x),result, 1.e-6);
}

BOOST_AUTO_TEST_SUITE_END()
