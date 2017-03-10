#define BOOST_TEST_MODULE Remora_Matrix_Expression
#include <boost/test/unit_test.hpp>
#include <boost/test/floating_point_comparison.hpp>


#include <remora/matrix_expression.hpp>
#include <remora/matrix.hpp>
#include <remora/vector.hpp>

using namespace remora;


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
	//check that row iterator work
	for(std::size_t i = 0; i != op().size1(); ++i){
		auto pos = op().row_begin(i);
		for(std::size_t j = 0; j != op().size2(); ++j,++pos){
			BOOST_CHECK_EQUAL(j, pos.index());
			BOOST_CHECK_CLOSE(result(i,j), *pos,1.e-5);
		}
	}
	
	//check that column iterator work
	for(std::size_t j = 0; j != op().size2(); ++j){
		auto pos = op().column_begin(j);
		for(std::size_t i = 0; i != op().size1(); ++i,++pos){
			BOOST_CHECK_EQUAL(i, pos.index());
			BOOST_CHECK_CLOSE(result(i,j), *pos,1.e-5);
		}
	}
}


template<class Operation, class Result>
void checkDenseExpressionEquality(
	vector_expression<Operation, cpu_tag> const& op, Result const& result
){
	for(std::size_t i = 0; i != op().size(); ++i){
		BOOST_CHECK_CLOSE(result(i), op()(i),1.e-5);
	}
}

template<class M, class D>
void checkDiagonalMatrix(M const& diagonal, D const& diagonalElements){
	BOOST_REQUIRE_EQUAL(diagonal.size1(),diagonalElements.size());
	BOOST_REQUIRE_EQUAL(diagonal.size2(),diagonalElements.size());
	for(std::size_t i = 0; i != diagonalElements.size(); ++i){
		for(std::size_t j = 0; j != diagonalElements.size(); ++j){
			if(i != j)
				BOOST_CHECK_EQUAL(diagonal(i,j),0);
			else
				BOOST_CHECK_EQUAL(diagonal(i,i),diagonalElements(i));
		}
		auto row_begin = diagonal.row_begin(i);
		auto col_begin = diagonal.column_begin(i);
		BOOST_CHECK_EQUAL(row_begin.index(),i);
		BOOST_CHECK_EQUAL(col_begin.index(),i);
		BOOST_CHECK_EQUAL(std::distance(row_begin, diagonal.row_end(i)),1);
		BOOST_CHECK_EQUAL(std::distance(col_begin, diagonal.column_end(i)),1);
		BOOST_CHECK_EQUAL(*row_begin,diagonalElements(i));
		BOOST_CHECK_EQUAL(*col_begin,diagonalElements(i));
	}
	
}

std::size_t Dimension1 = 50;
std::size_t Dimension2 = 100;

BOOST_AUTO_TEST_SUITE (Remora_matrix_expression)

/////////////////////////////////////////////////////////////
//////Vector->Matrix expansions///////
////////////////////////////////////////////////////////////

BOOST_AUTO_TEST_CASE( Remora_matrix_Outer_Prod ){
	vector<double> x(Dimension1); 
	vector<double> y(Dimension2); 
	matrix<double> result(Dimension1, Dimension2);
	
	for (size_t i = 0; i < Dimension1; i++)
		x(i) = i-3.0;
	for (size_t j = 0; j < Dimension2; j++)
		y(j) = 2*j;
	
	for (size_t i = 0; i < Dimension1; i++){
		for (size_t j = 0; j < Dimension2; j++){
			result(i,j)= x(i)*y(j);
		}
	}
	checkDenseExpressionEquality(outer_prod(x,y),result);
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
BOOST_AUTO_TEST_CASE( Remora_matrix_Unary_Minus )
{
	matrix<double> x(Dimension1, Dimension2); 
	matrix<double> result(Dimension1, Dimension2);
	
	for (size_t i = 0; i < Dimension1; i++){
		for (size_t j = 0; j < Dimension2; j++){
			x(i,j) = i-3.0+j;
			result(i,j)= -x(i,j);
		}
	}
	checkDenseExpressionEquality(-x,result);
}
BOOST_AUTO_TEST_CASE( Remora_matrix_Scalar_Multiply )
{
	matrix<double> x(Dimension1, Dimension2); 
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
BOOST_AUTO_TEST_CASE( Remora_matrix_Scalar_Add )
{
	matrix<double> x(Dimension1, Dimension2); 
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
BOOST_AUTO_TEST_CASE( Remora_matrix_Scalar_Subtract )
{
	matrix<double> x(Dimension1, Dimension2); 
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
BOOST_AUTO_TEST_CASE( Remora_matrix_Scalar_Div )
{
	matrix<double> x(Dimension1, Dimension2); 
	matrix<double> result(Dimension1, Dimension2);
	
	for (size_t i = 0; i < Dimension1; i++){
		for (size_t j = 0; j < Dimension2; j++){
			x(i,j) = i-3.0+j;
			result(i,j)= x(i,j)/5.0;
		}
	}
	checkDenseExpressionEquality(x/5.0f,result);
}
BOOST_AUTO_TEST_CASE( Remora_matrix_Scalar_elem_inv)
{
	matrix<double> x(Dimension1, Dimension2); 
	matrix<double> result(Dimension1, Dimension2);
	
	for (size_t i = 0; i < Dimension1; i++){
		for (size_t j = 0; j < Dimension2; j++){
			x(i,j) = i+j+3.0;
			result(i,j)= 1.0/x(i,j);
		}
	}
	checkDenseExpressionEquality(elem_inv(x),result);
}
BOOST_AUTO_TEST_CASE( Remora_matrix_Abs )
{
	matrix<double> x(Dimension1, Dimension2); 
	matrix<double> result(Dimension1, Dimension2);
	
	for (size_t i = 0; i < Dimension1; i++){
		for (size_t j = 0; j < Dimension2; j++){
			x(i,j) = i-3.0-j;
			result(i,j)= std::abs(x(i,j));
		}
	}
	checkDenseExpressionEquality(abs(x),result);
}
BOOST_AUTO_TEST_CASE( Remora_matrix_Sqr )
{
	matrix<double> x(Dimension1, Dimension2); 
	matrix<double> result(Dimension1, Dimension2);
	for (size_t i = 0; i < Dimension1; i++){
		for (size_t j = 0; j < Dimension2; j++){
			x(i,j) = i-3.0-j;
			result(i,j)= x(i,j) * x(i,j);
		}
	}
	checkDenseExpressionEquality(sqr(x),result);
}
BOOST_AUTO_TEST_CASE( Remora_matrix_Sqrt )
{
	matrix<double> x(Dimension1, Dimension2); 
	matrix<double> result(Dimension1, Dimension2);
	
	for (size_t i = 0; i < Dimension1; i++){
		for (size_t j = 0; j < Dimension2; j++){
			x(i,j) = i+j+3.0;
			result(i,j)= std::sqrt(x(i,j));
		}
	}
	checkDenseExpressionEquality(sqrt(x),result);
}
BOOST_AUTO_TEST_CASE( Remora_matrix_Cbrt )
{
	matrix<double> x(Dimension1, Dimension2); 
	matrix<double> result(Dimension1, Dimension2);
	
	for (size_t i = 0; i < Dimension1; i++){
		for (size_t j = 0; j < Dimension2; j++){
			x(i,j) = i+j+3.0;
			result(i,j)= std::cbrt(x(i,j));
		}
	}
	checkDenseExpressionEquality(cbrt(x),result);
}
BOOST_AUTO_TEST_CASE( Remora_matrix_Exp )
{
	matrix<double> x(Dimension1, Dimension2); 
	matrix<double> result(Dimension1, Dimension2);
	
	for (size_t i = 0; i < Dimension1; i++){
		for (size_t j = 0; j < Dimension2; j++){
			x(i,j) = i-3.0-double(j);
			result(i,j)= std::exp(x(i,j));
		}
	}
	checkDenseExpressionEquality(exp(x),result);
}
BOOST_AUTO_TEST_CASE( Remora_matrix_Log )
{

	matrix<double> x(Dimension1, Dimension2); 
	matrix<double> result(Dimension1, Dimension2);
	
	for (size_t i = 0; i < Dimension1; i++){
		for (size_t j = 0; j < Dimension2; j++){
			x(i,j) = i+j+3.0;
			result(i,j)= std::log(x(i,j));
		}
	}
	checkDenseExpressionEquality(log(x),result);
}
BOOST_AUTO_TEST_CASE( Remora_matrix_sin )
{
	matrix<double> x(Dimension1, Dimension2); 
	matrix<double> result(Dimension1, Dimension2);
	
	for (size_t i = 0; i < Dimension1; i++){
		for (size_t j = 0; j < Dimension2; j++){
			x(i,j) = double(i)/Dimension1-double(j)/Dimension2;
			result(i,j)= std::sin(x(i,j));
		}
	}
	checkDenseExpressionEquality(sin(x),result);
}

BOOST_AUTO_TEST_CASE( Remora_matrix_cos )
{
	matrix<double> x(Dimension1, Dimension2); 
	matrix<double> result(Dimension1, Dimension2);
	
	for (size_t i = 0; i < Dimension1; i++){
		for (size_t j = 0; j < Dimension2; j++){
			x(i,j) = double(i)/Dimension1-double(j)/Dimension2;
			result(i,j)= std::cos(x(i,j));
		}
	}
	checkDenseExpressionEquality(cos(x),result);
}

BOOST_AUTO_TEST_CASE( Remora_matrix_tan )
{
	matrix<double> x(Dimension1, Dimension2); 
	matrix<double> result(Dimension1, Dimension2);
	
	for (size_t i = 0; i < Dimension1; i++){
		for (size_t j = 0; j < Dimension2; j++){
			x(i,j) = double(i)/Dimension1-double(j)/Dimension2;
			result(i,j)= std::tan(x(i,j));
		}
	}
	checkDenseExpressionEquality(tan(x),result);
}

BOOST_AUTO_TEST_CASE( Remora_matrix_asin )
{
	matrix<double> x(Dimension1, Dimension2); 
	matrix<double> result(Dimension1, Dimension2);
	
	for (size_t i = 0; i < Dimension1; i++){
		for (size_t j = 0; j < Dimension2; j++){
			x(i,j) = double(i)/Dimension1-double(j)/Dimension2;
			result(i,j)= std::asin(x(i,j));
		}
	}
	checkDenseExpressionEquality(asin(x),result);
}

BOOST_AUTO_TEST_CASE( Remora_matrix_acos )
{
	matrix<double> x(Dimension1, Dimension2); 
	matrix<double> result(Dimension1, Dimension2);
	
	for (size_t i = 0; i < Dimension1; i++){
		for (size_t j = 0; j < Dimension2; j++){
			x(i,j) = double(i)/Dimension1-double(j)/Dimension2;
			result(i,j)= std::acos(x(i,j));
		}
	}
	checkDenseExpressionEquality(acos(x),result);
}

BOOST_AUTO_TEST_CASE( Remora_matrix_atan )
{
	matrix<double> x(Dimension1, Dimension2); 
	matrix<double> result(Dimension1, Dimension2);
	
	for (size_t i = 0; i < Dimension1; i++){
		for (size_t j = 0; j < Dimension2; j++){
			x(i,j) = double(i)/Dimension1-double(j)/Dimension2;
			result(i,j)= std::atan(x(i,j));
		}
	}
	checkDenseExpressionEquality(atan(x),result);
}

BOOST_AUTO_TEST_CASE( Remora_matrix_erf )
{
	matrix<double> x(Dimension1, Dimension2); 
	matrix<double> result(Dimension1, Dimension2);
	
	for (size_t i = 0; i < Dimension1; i++){
		for (size_t j = 0; j < Dimension2; j++){
			x(i,j) = double(i)/Dimension1-double(j)/Dimension2;
			result(i,j)= std::erf(x(i,j));
		}
	}
	checkDenseExpressionEquality(erf(x),result);
}

BOOST_AUTO_TEST_CASE( Remora_matrix_erfc )
{
	matrix<double> x(Dimension1, Dimension2); 
	matrix<double> result(Dimension1, Dimension2);
	
	for (size_t i = 0; i < Dimension1; i++){
		for (size_t j = 0; j < Dimension2; j++){
			x(i,j) = double(i)/Dimension1-double(j)/Dimension2;
			result(i,j)= std::erfc(x(i,j));
		}
	}
	checkDenseExpressionEquality(erfc(x),result);
}

BOOST_AUTO_TEST_CASE( Remora_matrix_Tanh )
{
	matrix<double> x(Dimension1, Dimension2); 
	matrix<double> result(Dimension1, Dimension2);
	
	for (size_t i = 0; i < Dimension1; i++){
		for (size_t j = 0; j < Dimension2; j++){
			x(i,j) = i-3.0-j;
			result(i,j)= std::tanh(x(i,j));
		}
	}
	checkDenseExpressionEquality(tanh(x),result);
}
BOOST_AUTO_TEST_CASE( Remora_matrix_Sigmoid )
{
	matrix<double> x(Dimension1, Dimension2); 
	matrix<double> result(Dimension1, Dimension2);
	
	for (size_t i = 0; i < Dimension1; i++){
		for (size_t j = 0; j < Dimension2; j++){
			x(i,j) = 0.01*(i-3.0-j);
			result(i,j)= 1.0/(1.0+std::exp(-x(i,j)));
		}
	}
	checkDenseExpressionEquality(sigmoid(x),result);
}
BOOST_AUTO_TEST_CASE( Remora_matrix_SoftPlus )
{
	matrix<double> x(Dimension1, Dimension2); 
	matrix<double> result(Dimension1, Dimension2);
	
	for (size_t i = 0; i < Dimension1; i++){
		for (size_t j = 0; j < Dimension2; j++){
			x(i,j) = i-3.0-j;
			result(i,j)= std::log(1+std::exp(x(i,j)));
		}
	}
	checkDenseExpressionEquality(softPlus(x),result);
}
BOOST_AUTO_TEST_CASE( Remora_matrix_Pow )
{
	matrix<double> x(Dimension1, Dimension2); 
	matrix<double> result(Dimension1, Dimension2);
	
	for (size_t i = 0; i < Dimension1; i++){
		for (size_t j = 0; j < Dimension2; j++){
			x(i,j) = 0.001*(i+j+2.0);
			result(i,j)= std::pow(x(i,j),3.2);
		}
	}
	checkDenseExpressionEquality(pow(x,3.2),result);
}

BOOST_AUTO_TEST_CASE( Remora_matrix_Unary_Min)
{
	matrix<double> x(Dimension1, Dimension2); 
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
BOOST_AUTO_TEST_CASE( Remora_matrix_Unary_Max)
{
	matrix<double> x(Dimension1, Dimension2); 
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
BOOST_AUTO_TEST_CASE( Remora_matrix_Binary_Minus)
{
	matrix<double> x(Dimension1, Dimension2); 
	matrix<double> y(Dimension1, Dimension2); 
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

BOOST_AUTO_TEST_CASE( Remora_matrix_Binary_Multiply)
{
	matrix<double> x(Dimension1, Dimension2); 
	matrix<double> y(Dimension1, Dimension2); 
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

BOOST_AUTO_TEST_CASE( Remora_matrix_Binary_Div)
{
	matrix<double> x(Dimension1, Dimension2); 
	matrix<double> y(Dimension1, Dimension2); 
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

BOOST_AUTO_TEST_CASE( Remora_matrix_Safe_Div )
{
	matrix<double> x(Dimension1, Dimension2); 
	matrix<double> y(Dimension1, Dimension2); 
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

BOOST_AUTO_TEST_CASE( Remora_matrix_Binary_Pow)
{
	matrix<double> x(Dimension1, Dimension2); 
	matrix<double> y(Dimension1, Dimension2); 
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

BOOST_AUTO_TEST_CASE( Remora_matrix_Binary_Max)
{
	matrix<double> x(Dimension1, Dimension2); 
	matrix<double> y(Dimension1, Dimension2); 
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
BOOST_AUTO_TEST_CASE( Remora_matrix_Binary_Min)
{
	matrix<double> x(Dimension1, Dimension2); 
	matrix<double> y(Dimension1, Dimension2); 
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

BOOST_AUTO_TEST_CASE( Remora_Concat_Matrix_Matrix_Right)
{
	matrix<double> x(Dimension1, Dimension2); 
	matrix<double> y(Dimension1,2 * Dimension2); 
	matrix<double> result(Dimension1, 3 * Dimension2);
	
	for (size_t i = 0; i < Dimension1; i++){
		for (size_t j = 0; j < Dimension2; j++){
			x(i,j) = 50.0+i-j;
			y(i,j) = i+j+1;
			y(i,j+Dimension2) = i+j+2;
			result(i,j)= x(i,j);
			result(i,j+Dimension2)= y(i,j);
			result(i,j+2*Dimension2)= y(i,j+Dimension2);
		}
	}
	matrix<double> test_assign = x|y;
	matrix<double> test_plus_assign(Dimension1, 3 * Dimension2,1.0); 
	noalias(test_plus_assign) += x|y;
	checkDenseExpressionEquality(test_assign,result);
	checkDenseExpressionEquality(test_plus_assign,result+1.0);
}

BOOST_AUTO_TEST_CASE( Remora_Concat_Matrix_Matrix_Bottom)
{
	matrix<double> x(Dimension1, Dimension2); 
	matrix<double> y(2 * Dimension1,Dimension2); 
	matrix<double> result(3 *Dimension1, Dimension2);
	
	for (size_t i = 0; i < Dimension1; i++){
		for (size_t j = 0; j < Dimension2; j++){
			x(i,j) = 50.0+i-j;
			y(i,j) = i+j+1;
			y(i+Dimension1, j) = i+j+2;
			result(i,j)= x(i,j);
			result(i + Dimension1, j)= y(i,j);
			result(i + 2 * Dimension1, j)= y(i + Dimension1,j);
		}
	}
	matrix<double> test_assign = x & y;
	matrix<double> test_plus_assign(3 * Dimension1, Dimension2,1.0); 
	noalias(test_plus_assign) += x & y;
	checkDenseExpressionEquality(test_assign,result);
	checkDenseExpressionEquality(test_plus_assign,result+1.0);
}

BOOST_AUTO_TEST_CASE( Remora_Concat_Matrix_Vector_Right)
{
	matrix<double> x(Dimension1, Dimension2); 
	vector<double> y(Dimension1); 
	matrix<double> result(Dimension1, Dimension2+1);
	
	for (size_t i = 0; i < Dimension1; i++){
		for (size_t j = 0; j < Dimension2; j++){
			x(i,j) = 50.0+i-j;
			result(i,j)= x(i,j);
		}
		y(i) = i;
		result(i,Dimension2) = y(i);
	}
	matrix<double> test_assign = x|y;
	matrix<double> test_plus_assign(Dimension1, Dimension2 +1 ,1.0); 
	noalias(test_plus_assign) += x|y;
	checkDenseExpressionEquality(test_assign,result);
	checkDenseExpressionEquality(test_plus_assign,result+1.0);
}

BOOST_AUTO_TEST_CASE( Remora_Concat_Matrix_Vector_Left)
{
	matrix<double> x(Dimension1, Dimension2); 
	vector<double> y(Dimension1); 
	matrix<double> result(Dimension1, Dimension2+1);
	
	for (size_t i = 0; i < Dimension1; i++){
		for (size_t j = 0; j < Dimension2; j++){
			x(i,j) = 50.0+i-j;
			result(i,j + 1)= x(i,j);
		}
		y(i) = i;
		result(i,0) = y(i);
	}
	matrix<double> test_assign = y|x;
	matrix<double> test_plus_assign(Dimension1, Dimension2 +1 ,1.0); 
	noalias(test_plus_assign) += y|x;
	checkDenseExpressionEquality(test_assign,result);
	checkDenseExpressionEquality(test_plus_assign,result+1.0);
}

BOOST_AUTO_TEST_CASE( Remora_Concat_Matrix_Vector_Bottom)
{
	matrix<double> x(Dimension1, Dimension2); 
	vector<double> y(Dimension2); 
	matrix<double> result(Dimension1 + 1, Dimension2);
	
	for (size_t i = 0; i < Dimension1; i++){
		for (size_t j = 0; j < Dimension2; j++){
			x(i,j) = 50.0+i-j;
			result(i,j)= x(i,j);
		}
	}
	for (size_t i = 0; i < Dimension2; i++){
		y(i) = i;
		result(Dimension1,i) = y(i);
	}
	matrix<double> test_assign = x&y;
	matrix<double> test_plus_assign(Dimension1 + 1, Dimension2 ,1.0); 
	noalias(test_plus_assign) += x&y;
	checkDenseExpressionEquality(test_assign,result);
	checkDenseExpressionEquality(test_plus_assign,result+1.0);
}

BOOST_AUTO_TEST_CASE( Remora_Concat_Matrix_Vector_Top)
{
	matrix<double> x(Dimension1, Dimension2); 
	vector<double> y(Dimension2); 
	matrix<double> result(Dimension1 + 1, Dimension2);
	
	for (size_t i = 0; i < Dimension1; i++){
		for (size_t j = 0; j < Dimension2; j++){
			x(i,j) = 50.0+i-j;
			result(i + 1,j)= x(i,j);
		}
	}
	for (size_t i = 0; i < Dimension2; i++){
		y(i) = i;
		result(0,i) = y(i);
	}
	matrix<double> test_assign = y&x;
	matrix<double> test_plus_assign(Dimension1 + 1, Dimension2 ,1.0); 
	noalias(test_plus_assign) += y&x;
	checkDenseExpressionEquality(test_assign,result);
	checkDenseExpressionEquality(test_plus_assign,result+1.0);
}

BOOST_AUTO_TEST_CASE( Remora_Concat_Matrix_Scalar_Left){
	matrix<double> x(Dimension1, Dimension2); 
	double t = 2.0;
	matrix<double> result(Dimension1, Dimension2 + 1);
	
	for (size_t i = 0; i < Dimension1; i++){
		for (size_t j = 0; j < Dimension2; j++){
			x(i,j) = 50.0+i-j;
			result(i,j+1)= x(i,j);
			result(i,0) = t;
		}
	}
	matrix<double> test_assign = t | x;
	matrix<double> test_plus_assign(Dimension1, Dimension2 + 1,1.0); 
	noalias(test_plus_assign) += t | x;
	checkDenseExpressionEquality(test_assign,result);
	checkDenseExpressionEquality(test_plus_assign,result+1.0);
}

BOOST_AUTO_TEST_CASE( Remora_Concat_Matrix_Scalar_Right){
	matrix<double> x(Dimension1, Dimension2); 
	double t = 2.0;
	matrix<double> result(Dimension1, Dimension2 + 1);
	
	for (size_t i = 0; i < Dimension1; i++){
		for (size_t j = 0; j < Dimension2; j++){
			x(i,j) = 50.0+i-j;
			result(i,j)= x(i,j);
			result(i,Dimension2) = t;
		}
	}
	matrix<double> test_assign = x | t;
	matrix<double> test_plus_assign(Dimension1, Dimension2 + 1,1.0); 
	noalias(test_plus_assign) += x | t;
	checkDenseExpressionEquality(test_assign,result);
	checkDenseExpressionEquality(test_plus_assign,result+1.0);
}

BOOST_AUTO_TEST_CASE( Remora_Concat_Matrix_Scalar_Bottom){
	matrix<double> x(Dimension1, Dimension2); 
	double t = 2.0;
	matrix<double> result(Dimension1 + 1, Dimension2);
	
	for (size_t i = 0; i < Dimension1; i++){
		for (size_t j = 0; j < Dimension2; j++){
			x(i,j) = 50.0+i-j;
			result(i,j)= x(i,j);
			result(Dimension1, j) = t;
		}
	}
	matrix<double> test_assign = x & t;
	matrix<double> test_plus_assign(Dimension1 + 1, Dimension2,1.0); 
	noalias(test_plus_assign) += x & t;
	checkDenseExpressionEquality(test_assign,result);
	checkDenseExpressionEquality(test_plus_assign,result+1.0);
}

BOOST_AUTO_TEST_CASE( Remora_Concat_Matrix_Scalar_Top){
	matrix<double> x(Dimension1, Dimension2); 
	double t = 2.0;
	matrix<double> result(Dimension1 + 1, Dimension2);
	
	for (size_t i = 0; i < Dimension1; i++){
		for (size_t j = 0; j < Dimension2; j++){
			x(i,j) = 50.0+i-j;
			result(i + 1,j)= x(i,j);
			result(0, j) = t;
		}
	}
	matrix<double> test_assign = t & x;
	matrix<double> test_plus_assign(Dimension1 + 1, Dimension2,1.0); 
	noalias(test_plus_assign) += t & x;
	checkDenseExpressionEquality(test_assign,result);
	checkDenseExpressionEquality(test_plus_assign,result+1.0);
}

////////////////////////////////////////////////////////////////////////
////////////ROW-WISE REDUCTIONS
////////////////////////////////////////////////////////////////////////
BOOST_AUTO_TEST_CASE( Remora_sum_rows){
	matrix<double> x_row(Dimension1, Dimension2); 
	vector<double> result(Dimension2,0.0);
	
	for (size_t i = 0; i < Dimension1; i++){
		for (size_t j = 0; j < Dimension2; j++){
			x_row(i,j) = i-3.0-j;
			result(j) += x_row(i,j);
		}
	}
	matrix<double, column_major> x_col = x_row;
	checkDenseExpressionEquality(eval_block(sum_rows(x_row)),result);
	checkDenseExpressionEquality(eval_block(sum_rows(x_col)),result);
}
BOOST_AUTO_TEST_CASE( Remora_sum_columns){
	matrix<double> x_row(Dimension1, Dimension2); 
	vector<double> result(Dimension1,0.0);
	
	for (size_t i = 0; i < Dimension1; i++){
		for (size_t j = 0; j < Dimension2; j++){
			x_row(i,j) = i-3.0-j;
			result(i) += x_row(i,j);
		}
	}
	matrix<double, column_major> x_col = x_row;
	checkDenseExpressionEquality(eval_block(sum_columns(x_row)),result);
	checkDenseExpressionEquality(eval_block(sum_columns(x_col)),result);
}

////////////////////////////////////////////////////////////////////////
////////////REDUCTIONS
////////////////////////////////////////////////////////////////////////

BOOST_AUTO_TEST_CASE( Remora_trace){
	matrix<double> x_row(Dimension1, Dimension1); 
	double result = 0.0f;
	for (size_t i = 0; i < Dimension1; i++){
		for (size_t j = 0; j < Dimension1; j++){
			x_row(i,j) = 2*i-3.0-j;
		}
		result += x_row(i,i);
	}
	matrix<double,column_major> x_col = x_row;
	BOOST_CHECK_CLOSE(trace(x_row),result, 1.e-6);
	BOOST_CHECK_CLOSE(trace(x_col),result, 1.e-6);
}

BOOST_AUTO_TEST_CASE( Remora_norm_1){
	matrix<double> x_row(Dimension1, Dimension1); 
	vector<double> col_sum(Dimension2);
	for (size_t i = 0; i < Dimension1; i++){
		for (size_t j = 0; j < Dimension1; j++){
			x_row(i,j) = 2*i-3.0-j;
			col_sum(j) += std::abs(x_row(i,j));
		}
	}
	double result = max(col_sum);
	matrix<double, column_major> x_col = x_row; 
	BOOST_CHECK_CLOSE(norm_1(x_row),result, 1.e-6);
	BOOST_CHECK_CLOSE(norm_1(x_col),result, 1.e-6);
}
BOOST_AUTO_TEST_CASE( Remora_norm_inf){
	matrix<double> x_row(Dimension1, Dimension1); 
	vector<double> row_sum(Dimension2);
	for (size_t i = 0; i < Dimension1; i++){
		for (size_t j = 0; j < Dimension1; j++){
			x_row(i,j) = 2*i-3.0-j;
			row_sum(i) += std::abs(x_row(i,j));
		}
	}
	double result = max(row_sum);
	matrix<double, column_major> x_col = x_row; 
	BOOST_CHECK_CLOSE(norm_inf(x_row),result, 1.e-6);
	BOOST_CHECK_CLOSE(norm_inf(x_col),result, 1.e-6);
}

BOOST_AUTO_TEST_CASE( Remora_norm_Frobenius){
	matrix<double> x_row(Dimension1, Dimension1); 
	double result = 0;
	for (size_t i = 0; i < Dimension1; i++){
		for (size_t j = 0; j < Dimension1; j++){
			x_row(i,j) = 2*i-3.0-j;
			result += x_row(i,j)*x_row(i,j);
		}
	}
	result = std::sqrt(result);
	matrix<double,column_major> x_col = x_row;
	BOOST_CHECK_CLOSE(norm_frobenius(x_row),result, 1.e-6);
	BOOST_CHECK_CLOSE(norm_frobenius(x_col),result, 1.e-6);
}

BOOST_AUTO_TEST_CASE( Remora_sum){
	matrix<double> x_row(Dimension1, Dimension1); 
	double result = 0;
	for (size_t i = 0; i < Dimension1; i++){
		for (size_t j = 0; j < Dimension1; j++){
			x_row(i,j) = 2*i-3.0-j;
			result +=x_row(i,j);
		}
	}
	
	matrix<double,column_major> x_col = x_row;
	BOOST_CHECK_CLOSE(sum(x_row),result, 1.e-6);
	BOOST_CHECK_CLOSE(sum(x_col),result, 1.e-6);
}

BOOST_AUTO_TEST_CASE( Remora_max){
	matrix<double> x_row(Dimension1, Dimension1);
	double result = std::numeric_limits<double>::min();
	for (size_t i = 0; i < Dimension1; i++){
		for (size_t j = 0; j < Dimension1; j++){
			x_row(i,j) = 2*i-3.0-j;
			result = std::max(x_row(i,j),result);
		}
	}
	matrix<double,column_major> x_col = x_row;
	BOOST_CHECK_CLOSE(max(x_row),result, 1.e-6);
	BOOST_CHECK_CLOSE(max(x_col),result, 1.e-6);
}

BOOST_AUTO_TEST_CASE( Remora_min){
	matrix<double> x_row(Dimension1, Dimension1); 
	double result = std::numeric_limits<double>::max();
	for (size_t i = 0; i < Dimension1; i++){
		for (size_t j = 0; j < Dimension1; j++){
			x_row(i,j) = 2*i-3.0-j;
			result = std::min(x_row(i,j),result);
		}
	}
	matrix<double,column_major> x_col = x_row;
	BOOST_CHECK_CLOSE(min(x_row),result, 1.e-6);
	BOOST_CHECK_CLOSE(min(x_col),result, 1.e-6);
}

BOOST_AUTO_TEST_CASE( Remora_frobenius_prod){
	matrix<double> x_row(Dimension1, Dimension2); 
	matrix<double> y_row(Dimension1, Dimension2); 
	double result = 0;
	for (size_t i = 0; i < Dimension1; i++){
		for (size_t j = 0; j < Dimension1; j++){
			x_row(i,j) = 2*i-3.0-j;
			y_row(i,j) = i+j+1;
			result +=x_row(i,j)*y_row(i,j);
		}
	}
	matrix<double, column_major> x_col = x_row;
	matrix<double, column_major> y_col = y_row;
	BOOST_CHECK_CLOSE(frobenius_prod(x_row,y_row),result, 1.e-6);
	BOOST_CHECK_CLOSE(frobenius_prod(x_row,y_col),result, 1.e-6);
	BOOST_CHECK_CLOSE(frobenius_prod(x_col,y_row),result, 1.e-6);
	BOOST_CHECK_CLOSE(frobenius_prod(trans(x_col),trans(y_col)),result, 1.e-6);
}

BOOST_AUTO_TEST_SUITE_END()
