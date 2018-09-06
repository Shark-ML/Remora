#define BOOST_TEST_MODULE Remora_HIP_Vector_Set_Expression
#include <boost/test/unit_test.hpp>
#include <boost/test/floating_point_comparison.hpp>

#include <remora/matrix_expression.hpp>
#include <remora/dense.hpp>
#include <remora/device_copy.hpp>

#include <boost/mpl/list.hpp>

using namespace remora;


template<class Operation, class Result>
void checkDenseExpressionEquality(
	vector_expression<Operation, hip_tag> const& op_hip, Result const& result
){
	BOOST_REQUIRE_EQUAL(op_hip().size(), result.size());
	
	vector<float> op = copy_to_cpu(op_hip());
	for(std::size_t i = 0; i != op.size(); ++i){
		BOOST_CHECK_CLOSE(result(i), op(i),1.e-4);
	}
}
template<class Operation, class Result>
void checkDenseExpressionEquality(
	vector_set_expression<Operation, hip_tag> const& op_hip, Result const& result
){
	matrix<float, row_major, hip_tag> mat_hip = op_hip;
	matrix<float> op = copy_to_cpu(mat_hip);
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
template<class Orientation>
void sum_set_test(Orientation){
	
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
	
	matrix<float, Orientation, hip_tag> x = copy_to_device(x_cpu, hip_tag());
	
	checkDenseExpressionEquality(sum(as_rows(x)),resultRows);
	checkDenseExpressionEquality(sum(as_columns(x)),resultColumns);
}
template<class Orientation>
void max_set_test(Orientation){
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
	
	matrix<float, Orientation, hip_tag> x = copy_to_device(x_cpu, hip_tag());
	
	checkDenseExpressionEquality(max(as_rows(x)),resultRows);
	checkDenseExpressionEquality(max(as_columns(x)),resultColumns);
}
template<class Orientation>
void min_set_test(Orientation){
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
	
	matrix<float, Orientation, hip_tag> x = copy_to_device(x_cpu, hip_tag());
	
	checkDenseExpressionEquality(min(as_rows(x)),resultRows);
	checkDenseExpressionEquality(min(as_columns(x)),resultColumns);
}
template<class Orientation>
void norm_1_set_test(Orientation){
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
	
	matrix<float, Orientation, hip_tag> x = copy_to_device(x_cpu, hip_tag());
	
	checkDenseExpressionEquality(norm_1(as_rows(x)),resultRows);
	checkDenseExpressionEquality(norm_1(as_columns(x)),resultColumns);
}

template<class Orientation>
void norm_sqr_set_test(Orientation){
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
	
	matrix<float, Orientation, hip_tag> x = copy_to_device(x_cpu, hip_tag());
	
	checkDenseExpressionEquality(norm_sqr(as_rows(x)),resultRows);
	checkDenseExpressionEquality(norm_sqr(as_columns(x)),resultColumns);
}

template<class Orientation>
void norm_2_set_test(Orientation){
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
	
	matrix<float, Orientation, hip_tag> x = copy_to_device(x_cpu, hip_tag());
	
	checkDenseExpressionEquality(norm_2(as_rows(x)),resultRows);
	checkDenseExpressionEquality(norm_2(as_columns(x)),resultColumns);
}

template<class Orientation>
void norm_inf_set_test(Orientation){
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
	
	matrix<float, Orientation, hip_tag> x = copy_to_device(x_cpu, hip_tag());
	
	checkDenseExpressionEquality(norm_inf(as_rows(x)),resultRows);
	checkDenseExpressionEquality(norm_inf(as_columns(x)),resultColumns);
}

BOOST_AUTO_TEST_CASE(Remora_Hip_Set_Reductions){
	std::cout<<"sum set test row"<<std::endl;
	sum_set_test(row_major());
	std::cout<<"sum set test column"<<std::endl;
	sum_set_test(column_major());
	
	std::cout<<"max set test row"<<std::endl;
	max_set_test(row_major());
	std::cout<<"max set test column"<<std::endl;
	max_set_test(column_major());
	
	std::cout<<"min set test row"<<std::endl;
	min_set_test(row_major());
	std::cout<<"min set test column"<<std::endl;
	min_set_test(column_major());
	
	std::cout<<"norm_1 set test row"<<std::endl;
	norm_1_set_test(row_major());
	std::cout<<"norm_1 set test column"<<std::endl;
	norm_1_set_test(column_major());
	
	std::cout<<"norm_sqr set test row"<<std::endl;
	norm_sqr_set_test(row_major());
	std::cout<<"norm_sqr set test column"<<std::endl;
	norm_sqr_set_test(column_major());
	
	std::cout<<"norm_2 set test row"<<std::endl;
	norm_2_set_test(row_major());
	std::cout<<"norm_2 set test column"<<std::endl;
	norm_2_set_test(column_major());
	
	std::cout<<"norm_inf set test row"<<std::endl;
	norm_inf_set_test(row_major());
	std::cout<<"norm_inf set test column"<<std::endl;
	norm_inf_set_test(column_major());
}


/////////////////////////////////////////////////////////////
//////UNARY TRANSFORMATIONS///////
////////////////////////////////////////////////////////////
template<class Orientation>
void unary_minus_set_test(Orientation){
	matrix<float, Orientation> x_cpu(Dimension1, Dimension2);
	matrix<float> result(Dimension1, Dimension2);
	
	for (size_t i = 0; i < Dimension1; i++){
		for (size_t j = 0; j < Dimension2; j++){
			x_cpu(i,j) = i-3.0+j;
			result(i,j)= -x_cpu(i,j);
		}
	}
	matrix<float, Orientation, hip_tag> x = copy_to_device(x_cpu, hip_tag());
	checkDenseExpressionEquality(-as_rows(x),result);
	checkDenseExpressionEquality(-as_columns(x),result);
}
template<class Orientation>
void scalar_inv_set_test(Orientation){
	matrix<float, Orientation> x_cpu(Dimension1, Dimension2); 
	matrix<float> result(Dimension1, Dimension2);
	
	for (size_t i = 0; i < Dimension1; i++){
		for (size_t j = 0; j < Dimension2; j++){
			x_cpu(i,j) = i+j+3.0;
			result(i,j)= 1.0/x_cpu(i,j);
		}
	}
	matrix<float, Orientation, hip_tag> x = copy_to_device(x_cpu, hip_tag());
	checkDenseExpressionEquality(elem_inv(as_rows(x)),result);
	checkDenseExpressionEquality(elem_inv(as_columns(x)),result);
}
template<class Orientation>
void abs_set_test(Orientation){
	matrix<float, Orientation> x_cpu(Dimension1, Dimension2); 
	matrix<float> result(Dimension1, Dimension2);
	
	for (size_t i = 0; i < Dimension1; i++){
		for (size_t j = 0; j < Dimension2; j++){
			x_cpu(i,j) = i-3.0-j;
			result(i,j)= std::abs(x_cpu(i,j));
		}
	}
	matrix<float, Orientation, hip_tag> x = copy_to_device(x_cpu, hip_tag());
	checkDenseExpressionEquality(abs(as_rows(x)),result);
	checkDenseExpressionEquality(abs(as_columns(x)),result);
}
template<class Orientation>
void sqr_set_test(Orientation){
	matrix<float, Orientation> x_cpu(Dimension1, Dimension2); 
	matrix<float> result(Dimension1, Dimension2);
	for (size_t i = 0; i < Dimension1; i++){
		for (size_t j = 0; j < Dimension2; j++){
			x_cpu(i,j) = i-3.0-j;
			result(i,j)= x_cpu(i,j) * x_cpu(i,j);
		}
	}
	matrix<float, Orientation, hip_tag> x = copy_to_device(x_cpu, hip_tag());
	checkDenseExpressionEquality(sqr(as_rows(x)),result);
	checkDenseExpressionEquality(sqr(as_columns(x)),result);
}
template<class Orientation>
void sqrt_set_test(Orientation){
	matrix<float, Orientation> x_cpu(Dimension1, Dimension2); 
	matrix<float> result(Dimension1, Dimension2);
	
	for (size_t i = 0; i < Dimension1; i++){
		for (size_t j = 0; j < Dimension2; j++){
			x_cpu(i,j) = i+j+3.0;
			result(i,j)= std::sqrt(x_cpu(i,j));
		}
	}
	matrix<float, Orientation, hip_tag> x = copy_to_device(x_cpu, hip_tag());
	checkDenseExpressionEquality(sqrt(as_rows(x)),result);
	checkDenseExpressionEquality(sqrt(as_columns(x)),result);
}
template<class Orientation>
void cbrt_set_test(Orientation){
	matrix<float, Orientation> x_cpu(Dimension1, Dimension2); 
	matrix<float> result(Dimension1, Dimension2);
	
	for (size_t i = 0; i < Dimension1; i++){
		for (size_t j = 0; j < Dimension2; j++){
			x_cpu(i,j) = i+j+3.0;
			result(i,j)= std::cbrt(x_cpu(i,j));
		}
	}
	matrix<float, Orientation, hip_tag> x = copy_to_device(x_cpu, hip_tag());
	checkDenseExpressionEquality(cbrt(as_rows(x)),result);
	checkDenseExpressionEquality(cbrt(as_columns(x)),result);
}
template<class Orientation>
void exp_set_test(Orientation){
	matrix<float, Orientation> x_cpu(Dimension1, Dimension2); 
	matrix<float> result(Dimension1, Dimension2);
	
	for (size_t i = 0; i < Dimension1; i++){
		for (size_t j = 0; j < Dimension2; j++){
			x_cpu(i,j) = 0.01*(i-3.0-float(j));
			result(i,j)= std::exp(x_cpu(i,j));
		}
	}
	matrix<float, Orientation, hip_tag> x = copy_to_device(x_cpu, hip_tag());
	checkDenseExpressionEquality(exp(as_rows(x)),result);
	checkDenseExpressionEquality(exp(as_columns(x)),result);
}
template<class Orientation>
void log_set_test(Orientation){
	matrix<float, Orientation> x_cpu(Dimension1, Dimension2); 
	matrix<float> result(Dimension1, Dimension2);
	
	for (size_t i = 0; i < Dimension1; i++){
		for (size_t j = 0; j < Dimension2; j++){
			x_cpu(i,j) = i+j+3.0;
			result(i,j)= std::log(x_cpu(i,j));
		}
	}
	matrix<float, Orientation, hip_tag> x = copy_to_device(x_cpu, hip_tag());
	checkDenseExpressionEquality(log(as_rows(x)),result);
	checkDenseExpressionEquality(log(as_columns(x)),result);
}
template<class Orientation>
void sin_set_test(Orientation){
	matrix<float, Orientation> x_cpu(Dimension1, Dimension2); 
	matrix<float> result(Dimension1, Dimension2);
	
	for (size_t i = 0; i < Dimension1; i++){
		for (size_t j = 0; j < Dimension2; j++){
			x_cpu(i,j) = float(i)/Dimension1-float(j)/Dimension2;
			result(i,j)= std::sin(x_cpu(i,j));
		}
	}
	matrix<float, Orientation, hip_tag> x = copy_to_device(x_cpu, hip_tag());
	checkDenseExpressionEquality(sin(as_rows(x)),result);
	checkDenseExpressionEquality(sin(as_columns(x)),result);
}

template<class Orientation>
void cos_set_test(Orientation){
	matrix<float, Orientation> x_cpu(Dimension1, Dimension2); 
	matrix<float> result(Dimension1, Dimension2);
	
	for (size_t i = 0; i < Dimension1; i++){
		for (size_t j = 0; j < Dimension2; j++){
			x_cpu(i,j) = float(i)/Dimension1-float(j)/Dimension2;
			result(i,j)= std::cos(x_cpu(i,j));
		}
	}
	matrix<float, Orientation, hip_tag> x = copy_to_device(x_cpu, hip_tag());
	checkDenseExpressionEquality(cos(as_rows(x)),result);
	checkDenseExpressionEquality(cos(as_columns(x)),result);
}

template<class Orientation>
void tan_set_test(Orientation){
	matrix<float, Orientation> x_cpu(Dimension1, Dimension2); 
	matrix<float> result(Dimension1, Dimension2);
	
	for (size_t i = 0; i < Dimension1; i++){
		for (size_t j = 0; j < Dimension2; j++){
			x_cpu(i,j) = float(i)/Dimension1-float(j)/Dimension2;
			result(i,j)= std::tan(x_cpu(i,j));
		}
	}
	matrix<float, Orientation, hip_tag> x = copy_to_device(x_cpu, hip_tag());
	checkDenseExpressionEquality(tan(as_rows(x)),result);
	checkDenseExpressionEquality(tan(as_columns(x)),result);
}

template<class Orientation>
void asin_set_test(Orientation){
	matrix<float, Orientation> x_cpu(Dimension1, Dimension2); 
	matrix<float> result(Dimension1, Dimension2);
	
	for (size_t i = 0; i < Dimension1; i++){
		for (size_t j = 0; j < Dimension2; j++){
			x_cpu(i,j) = float(i)/Dimension1-float(j)/Dimension2;
			result(i,j)= std::asin(x_cpu(i,j));
		}
	}
	matrix<float, Orientation, hip_tag> x = copy_to_device(x_cpu, hip_tag());
	checkDenseExpressionEquality(asin(as_rows(x)),result);
	checkDenseExpressionEquality(asin(as_columns(x)),result);
}

template<class Orientation>
void acos_set_test(Orientation){
	matrix<float, Orientation> x_cpu(Dimension1, Dimension2); 
	matrix<float> result(Dimension1, Dimension2);
	
	for (size_t i = 0; i < Dimension1; i++){
		for (size_t j = 0; j < Dimension2; j++){
			x_cpu(i,j) = float(i)/Dimension1-float(j)/Dimension2;
			result(i,j)= std::acos(x_cpu(i,j));
		}
	}
	matrix<float, Orientation, hip_tag> x = copy_to_device(x_cpu, hip_tag());
	checkDenseExpressionEquality(acos(as_rows(x)),result);
	checkDenseExpressionEquality(acos(as_columns(x)),result);
}

template<class Orientation>
void atan_set_test(Orientation){
	matrix<float, Orientation> x_cpu(Dimension1, Dimension2); 
	matrix<float> result(Dimension1, Dimension2);
	
	for (size_t i = 0; i < Dimension1; i++){
		for (size_t j = 0; j < Dimension2; j++){
			x_cpu(i,j) = float(i)/Dimension1-float(j)/Dimension2;
			result(i,j)= std::atan(x_cpu(i,j));
		}
	}
	matrix<float, Orientation, hip_tag> x = copy_to_device(x_cpu, hip_tag());
	checkDenseExpressionEquality(atan(as_rows(x)),result);
	checkDenseExpressionEquality(atan(as_columns(x)),result);
}

template<class Orientation>
void erf_set_test(Orientation){
	matrix<float, Orientation> x_cpu(Dimension1, Dimension2); 
	matrix<float> result(Dimension1, Dimension2);
	
	for (size_t i = 0; i < Dimension1; i++){
		for (size_t j = 0; j < Dimension2; j++){
			x_cpu(i,j) = float(i)/Dimension1-float(j)/Dimension2;
			result(i,j)= std::erf(x_cpu(i,j));
		}
	}
	matrix<float, Orientation, hip_tag> x = copy_to_device(x_cpu, hip_tag());
	checkDenseExpressionEquality(erf(as_rows(x)),result);
	checkDenseExpressionEquality(erf(as_columns(x)),result);
}

template<class Orientation>
void erfc_set_test(Orientation){
	matrix<float, Orientation> x_cpu(Dimension1, Dimension2); 
	matrix<float> result(Dimension1, Dimension2);
	
	for (size_t i = 0; i < Dimension1; i++){
		for (size_t j = 0; j < Dimension2; j++){
			x_cpu(i,j) = float(i)/Dimension1-float(j)/Dimension2;
			result(i,j)= std::erfc(x_cpu(i,j));
		}
	}
	matrix<float, Orientation, hip_tag> x = copy_to_device(x_cpu, hip_tag());
	checkDenseExpressionEquality(erfc(as_rows(x)),result);
	checkDenseExpressionEquality(erfc(as_columns(x)),result);
}

template<class Orientation>
void tanh_set_test(Orientation){
	matrix<float, Orientation> x_cpu(Dimension1, Dimension2); 
	matrix<float> result(Dimension1, Dimension2);
	
	for (size_t i = 0; i < Dimension1; i++){
		for (size_t j = 0; j < Dimension2; j++){
			x_cpu(i,j) = i-3.0-j;
			result(i,j)= std::tanh(x_cpu(i,j));
		}
	}
	matrix<float, Orientation, hip_tag> x = copy_to_device(x_cpu, hip_tag());
	checkDenseExpressionEquality(tanh(as_rows(x)),result);
	checkDenseExpressionEquality(tanh(as_columns(x)),result);
}
template<class Orientation>
void sigmoid_set_test(Orientation){
	matrix<float, Orientation> x_cpu(Dimension1, Dimension2); 
	matrix<float> result(Dimension1, Dimension2);
	
	for (size_t i = 0; i < Dimension1; i++){
		for (size_t j = 0; j < Dimension2; j++){
			x_cpu(i,j) = 0.01*(i-3.0-j);
			result(i,j)= 1.0/(1.0+std::exp(-x_cpu(i,j)));
		}
	}
	matrix<float, Orientation, hip_tag> x = copy_to_device(x_cpu, hip_tag());
	checkDenseExpressionEquality(sigmoid(as_rows(x)),result);
	checkDenseExpressionEquality(sigmoid(as_columns(x)),result);
}
template<class Orientation>
void softPlus_set_test(Orientation){
	matrix<float, Orientation> x_cpu(Dimension1, Dimension2); 
	matrix<float> result(Dimension1, Dimension2);
	
	for (size_t i = 0; i < Dimension1; i++){
		for (size_t j = 0; j < Dimension2; j++){
			x_cpu(i,j) = i-3.0-j;
			result(i,j)= std::log(1+std::exp(x_cpu(i,j)));
		}
	}
	matrix<float, Orientation, hip_tag> x = copy_to_device(x_cpu, hip_tag());
	checkDenseExpressionEquality(softPlus(as_rows(x)),result);
	checkDenseExpressionEquality(softPlus(as_columns(x)),result);
}

BOOST_AUTO_TEST_CASE(Remora_Hip_Set_Unary_Op){
	std::cout<<"unary minus test row"<<std::endl;
	unary_minus_set_test(row_major());
	std::cout<<"unary_minus set test column"<<std::endl;
	unary_minus_set_test(column_major());
	
	std::cout<<"scalar inv test row"<<std::endl;
	scalar_inv_set_test(row_major());
	std::cout<<"scalar inv set test column"<<std::endl;
	scalar_inv_set_test(column_major());
	
	std::cout<<"abs test row"<<std::endl;
	abs_set_test(row_major());
	std::cout<<"abs set test column"<<std::endl;
	abs_set_test(column_major());
	
	std::cout<<"sqr test row"<<std::endl;
	sqr_set_test(row_major());
	std::cout<<"sqr set test column"<<std::endl;
	sqr_set_test(column_major());
	
	std::cout<<"sqrt test row"<<std::endl;
	sqrt_set_test(row_major());
	std::cout<<"sqrt set test column"<<std::endl;
	sqrt_set_test(column_major());
	
	std::cout<<"cbrt test row"<<std::endl;
	cbrt_set_test(row_major());
	std::cout<<"cbrt set test column"<<std::endl;
	cbrt_set_test(column_major());
	
	std::cout<<"exp test row"<<std::endl;
	exp_set_test(row_major());
	std::cout<<"exp set test column"<<std::endl;
	exp_set_test(column_major());
	
	std::cout<<"log test row"<<std::endl;
	log_set_test(row_major());
	std::cout<<"log set test column"<<std::endl;
	log_set_test(column_major());
	
	std::cout<<"sin test row"<<std::endl;
	sin_set_test(row_major());
	std::cout<<"sin set test column"<<std::endl;
	sin_set_test(column_major());
	
	std::cout<<"cos test row"<<std::endl;
	cos_set_test(row_major());
	std::cout<<"cos set test column"<<std::endl;
	cos_set_test(column_major());
	
	std::cout<<"tan test row"<<std::endl;
	tan_set_test(row_major());
	std::cout<<"tan set test column"<<std::endl;
	tan_set_test(column_major());
	
	std::cout<<"asin test row"<<std::endl;
	asin_set_test(row_major());
	std::cout<<"asin set test column"<<std::endl;
	asin_set_test(column_major());
	
	std::cout<<"acos test row"<<std::endl;
	acos_set_test(row_major());
	std::cout<<"acos set test column"<<std::endl;
	acos_set_test(column_major());
	
	std::cout<<"atan test row"<<std::endl;
	atan_set_test(row_major());
	std::cout<<"atan set test column"<<std::endl;
	atan_set_test(column_major());
	
	std::cout<<"erf test row"<<std::endl;
	erf_set_test(row_major());
	std::cout<<"erf set test column"<<std::endl;
	erf_set_test(column_major());
	
	std::cout<<"erfc test row"<<std::endl;
	erfc_set_test(row_major());
	std::cout<<"erfc set test column"<<std::endl;
	erfc_set_test(column_major());
	
	std::cout<<"tanh test row"<<std::endl;
	tanh_set_test(row_major());
	std::cout<<"tanh set test column"<<std::endl;
	tanh_set_test(column_major());
	
	std::cout<<"sigmoid test row"<<std::endl;
	sigmoid_set_test(row_major());
	std::cout<<"sigmoid set test column"<<std::endl;
	sigmoid_set_test(column_major());
	
	std::cout<<"softPlus test row"<<std::endl;
	softPlus_set_test(row_major());
	std::cout<<"softPlus set test column"<<std::endl;
	softPlus_set_test(column_major());
}
BOOST_AUTO_TEST_SUITE_END()
