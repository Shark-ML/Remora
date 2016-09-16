/*!
 * \brief       Expression templates for expressions involving matrices
 * 
 * \author      O. Krause
 * \date        2016
 *
 *
 * \par Copyright 1995-2015 Shark Development Team
 * 
 * <BR><HR>
 * This file is part of Shark.
 * <http://image.diku.dk/shark/>
 * 
 * Shark is free software: you can redistribute it and/or modify
 * it under the terms of the GNU Lesser General Public License as published 
 * by the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 * 
 * Shark is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU Lesser General Public License for more details.
 * 
 * You should have received a copy of the GNU Lesser General Public License
 * along with Shark.  If not, see <http://www.gnu.org/licenses/>.
 *
 */
#ifndef SHARK_LINALG_BLAS_MATRIX_EXPRESSION_HPP
#define SHARK_LINALG_BLAS_MATRIX_EXPRESSION_HPP

#include "detail/expression_optimizers.hpp"
#include <boost/utility/enable_if.hpp>


namespace shark {
namespace blas {


///\brief Computes the outer product of two vectors.
///
/// The outer product of two vectors v1 and v2 is defined as the matrix
/// (outer_prod (v1, v2))_ij [i] [j] = v1[i] * v2 [j]
template<class MatA, class MatB, class Device>
outer_product<MatA, MatB >
outer_prod(
	vector_expression<MatA, Device> const& A,
        vector_expression<MatB, Device> const& B
) {
	return outer_product<MatA, MatB>(A(), B());
}



///\brief Creates a matrix from a vector by repeating the vector in every row of the matrix.
///
///example: vector = (1,2,3)
///repeat(vector,3) results in
///(1,2,3)
///(1,2,3)
///(1,2,3)
///@param vector the vector which is to be repeated as the rows of the resulting matrix
///@param rows the number of rows of the matrix
template<class VecV, class Device>
vector_repeater<VecV> repeat(vector_expression<VecV, Device> const& vector, std::size_t rows){
	return vector_repeater<VecV>(vector(),rows);
}

/// \brief Repeats a single element to form a matrix  of size rows x columns.
///
///@param scalar the value which is repeated
///@param rows the number of rows of the resulting vector
///@param columns the number of columns of the resulting vector
template<class T>
typename boost::enable_if<std::is_arithmetic<T>, scalar_matrix<T> >::type
repeat(T scalar, std::size_t rows, std::size_t columns){
	return scalar_matrix<T>(rows, columns, scalar);
}


/// \brief Computes the multiplication of a matrix-expression A with a scalar t.
///
/// \f$ (A*t)_{ij} = e_{ij}*t \f$
template<class MatA, class T, class Device>
typename boost::enable_if<
	std::is_convertible<T, typename MatA::scalar_type >,
        matrix_scalar_multiply<MatA> 
>::type
operator* (matrix_expression<MatA, Device> const& A, T scalar){
	return matrix_scalar_multiply<MatA>(A(), typename MatA::scalar_type(scalar));
}

/// \brief Computes the multiplication of a matrix-expression A with a scalar t.
///
/// \f$ (t*A)_{ij} = t*e_{ij} \f$
template<class T, class MatA, class Device>
typename boost::enable_if<
	std::is_convertible<T, typename MatA::scalar_type >,
        matrix_scalar_multiply<MatA> 
>::type
operator* (T scalar, matrix_expression<MatA, Device> const& A){
	return matrix_scalar_multiply<MatA>(A(), typename MatA::scalar_type(scalar));
}

/// \brief Negates the matrix-expression A.
///
/// \f$ (-A)_{ij} = - e_{ij} \f$
template<class MatA, class Device>
matrix_scalar_multiply<MatA> operator-(matrix_expression<MatA, Device> const& A){
	return matrix_scalar_multiply<MatA>(A(), typename MatA::scalar_type(-1));
}

#define SHARK_UNARY_MATRIX_TRANSFORMATION(name, F)\
template<class MatA, class Device>\
matrix_unary<MatA,F<typename MatA::scalar_type> >\
name(matrix_expression<MatA, Device> const& A){\
	typedef F<typename MatA::scalar_type> functor_type;\
	return matrix_unary<MatA, functor_type>(A(), functor_type());\
}
SHARK_UNARY_MATRIX_TRANSFORMATION(conj, scalar_conj)
SHARK_UNARY_MATRIX_TRANSFORMATION(real, scalar_real)
SHARK_UNARY_MATRIX_TRANSFORMATION(imag, scalar_imag)
SHARK_UNARY_MATRIX_TRANSFORMATION(abs, scalar_abs)
SHARK_UNARY_MATRIX_TRANSFORMATION(log, scalar_log)
SHARK_UNARY_MATRIX_TRANSFORMATION(exp, scalar_exp)
SHARK_UNARY_MATRIX_TRANSFORMATION(sin, scalar_sin)
SHARK_UNARY_MATRIX_TRANSFORMATION(cos, scalar_cos)
SHARK_UNARY_MATRIX_TRANSFORMATION(tanh,scalar_tanh)
SHARK_UNARY_MATRIX_TRANSFORMATION(atanh,scalar_atanh)
SHARK_UNARY_MATRIX_TRANSFORMATION(sqr, scalar_sqr)
SHARK_UNARY_MATRIX_TRANSFORMATION(abs_sqr, scalar_abs_sqr)
SHARK_UNARY_MATRIX_TRANSFORMATION(sqrt, scalar_sqrt)
SHARK_UNARY_MATRIX_TRANSFORMATION(sigmoid, scalar_sigmoid)
SHARK_UNARY_MATRIX_TRANSFORMATION(softPlus, scalar_soft_plus)
SHARK_UNARY_MATRIX_TRANSFORMATION(elem_inv, scalar_inverse)
#undef SHARK_UNARY_MATRIX_TRANSFORMATION

#define SHARK_MATRIX_SCALAR_TRANSFORMATION(name, F)\
template<class MatA, class T, class Device> \
typename boost::enable_if< \
	std::is_convertible<T, typename MatA::scalar_type >,\
        matrix_unary<MatA,F<typename MatA::scalar_type,T> > \
>::type \
name (matrix_expression<MatA, Device> const& A, T scalar){ \
	typedef F<typename MatA::scalar_type, T> functor_type; \
	return matrix_unary<MatA, functor_type>(A(), functor_type(scalar)); \
}
SHARK_MATRIX_SCALAR_TRANSFORMATION(operator/, scalar_divide)
SHARK_MATRIX_SCALAR_TRANSFORMATION(operator<, scalar_less_than)
SHARK_MATRIX_SCALAR_TRANSFORMATION(operator<=, scalar_less_equal_than)
SHARK_MATRIX_SCALAR_TRANSFORMATION(operator>, scalar_bigger_than)
SHARK_MATRIX_SCALAR_TRANSFORMATION(operator>=, scalar_bigger_equal_than)
SHARK_MATRIX_SCALAR_TRANSFORMATION(operator==, scalar_equal)
SHARK_MATRIX_SCALAR_TRANSFORMATION(operator!=, scalar_not_equal)
SHARK_MATRIX_SCALAR_TRANSFORMATION(min, scalar_min)
SHARK_MATRIX_SCALAR_TRANSFORMATION(max, scalar_max)
SHARK_MATRIX_SCALAR_TRANSFORMATION(pow, scalar_pow)
#undef SHARK_MATRIX_SCALAR_TRANSFORMATION

// operations of the form op(t,v)[i,j] = op(t,v[i,j])
#define SHARK_MATRIX_SCALAR_TRANSFORMATION_2(name, F)\
template<class T, class MatA, class Device> \
typename boost::enable_if< \
	std::is_convertible<T, typename MatA::scalar_type >,\
        matrix_unary<MatA,F<typename MatA::scalar_type,T> > \
>::type \
name (T scalar, matrix_expression<MatA, Device> const& A){ \
	typedef F<typename MatA::scalar_type, T> functor_type; \
	return matrix_unary<MatA, functor_type>(A(), functor_type(scalar)); \
}
SHARK_MATRIX_SCALAR_TRANSFORMATION_2(min, scalar_min)
SHARK_MATRIX_SCALAR_TRANSFORMATION_2(max, scalar_max)
#undef SHARK_MATRIX_SCALAR_TRANSFORMATION_2

///\brief Adds two Matrices
template<class MatA, class MatB, class Device>
matrix_addition<MatA, MatB > operator+ (
	matrix_expression<MatA, Device> const& A,
	matrix_expression<MatB, Device> const& B
){
	SIZE_CHECK(A().size1() == B().size1());
	SIZE_CHECK(A().size2() == B().size2());
	return matrix_addition<MatA, MatB>(A(),B());
}

///\brief Subtracts two Matrices
template<class MatA, class MatB, class Device>
matrix_addition<MatA, matrix_scalar_multiply<MatB> > operator- (
	matrix_expression<MatA, Device> const& A,
	matrix_expression<MatB, Device> const& B
){
	SIZE_CHECK(A().size1() == B().size1());
	SIZE_CHECK(A().size2() == B().size2());
	return matrix_addition<MatA, matrix_scalar_multiply<MatB> >(A(),-B());
}

///\brief Adds a matrix plus a scalar which is interpreted as a constant matrix
template<class MatA, class T, class Device>
typename boost::enable_if<
	std::is_convertible<T, typename MatA::scalar_type>, 
	matrix_addition<MatA, scalar_matrix<T> >
>::type operator+ (
	matrix_expression<MatA, Device> const& A,
	T t
){
	return A + scalar_matrix<T>(A().size1(),A().size2(),t);
}

///\brief Adds a matrix plus a scalar which is interpreted as a constant matrix
template<class T, class MatA, class Device>
typename boost::enable_if<
	std::is_convertible<T, typename MatA::scalar_type>,
	matrix_addition<MatA, scalar_matrix<T> >
>::type operator+ (
	T t,
	matrix_expression<MatA, Device> const& A
){
	return A + scalar_matrix<T>(A().size1(),A().size2(),t);
}

///\brief Subtracts a scalar which is interpreted as a constant matrix from a matrix.
template<class MatA, class T, class Device>
typename boost::enable_if<
	std::is_convertible<T, typename MatA::scalar_type> ,
	matrix_addition<MatA, matrix_scalar_multiply<scalar_matrix<T> > >
>::type operator- (
	matrix_expression<MatA, Device> const& A,
	T t
){
	return A - scalar_matrix<T>(A().size1(),A().size2(),t);
}

///\brief Subtracts a matrix from a scalar which is interpreted as a constant matrix
template<class MatA, class T, class Device>
typename boost::enable_if<
	std::is_convertible<T, typename MatA::scalar_type>,
	matrix_addition<scalar_matrix<T>, matrix_scalar_multiply<MatA> >
>::type operator- (
	T t,
	matrix_expression<MatA, Device> const& A
){
	return scalar_matrix<T>(A().size1(),A().size2(),t) - A;
}

#define SHARK_BINARY_MATRIX_EXPRESSION(name, F)\
template<class MatA, class MatB, class Device>\
matrix_binary<MatA, MatB, F<typename MatA::value_type, typename MatB::value_type> >\
name(matrix_expression<MatA, Device> const& A, matrix_expression<MatB, Device> const& B){\
	SIZE_CHECK(A().size1() == B().size1());\
	SIZE_CHECK(A().size2() == B().size2());\
	typedef F<typename MatA::value_type, typename MatB::value_type> functor_type;\
	return matrix_binary<MatA, MatB, functor_type>(A(),B(), functor_type());\
}
SHARK_BINARY_MATRIX_EXPRESSION(operator*, scalar_binary_multiply)
SHARK_BINARY_MATRIX_EXPRESSION(element_prod, scalar_binary_multiply)
SHARK_BINARY_MATRIX_EXPRESSION(operator/, scalar_binary_divide)
SHARK_BINARY_MATRIX_EXPRESSION(pow,scalar_binary_pow)
SHARK_BINARY_MATRIX_EXPRESSION(element_div, scalar_binary_divide)
#undef SHARK_BINARY_MATRIX_EXPRESSION

template<class MatA, class MatB, class Device>
matrix_binary<MatA, MatB, 
	scalar_binary_safe_divide<typename MatA::value_type, typename MatB::value_type> 
>
safe_div(
	matrix_expression<MatA, Device> const& A, 
	matrix_expression<MatB, Device> const& B, 
	decltype(typename MatA::value_type()/typename MatB::value_type()) defaultValue
){
	SIZE_CHECK(A().size1() == B().size1());
	SIZE_CHECK(A().size2() == B().size2());
	typedef scalar_binary_safe_divide<typename MatA::value_type, typename MatB::value_type> functor_type;
	return matrix_binary<MatA, MatB, functor_type>(A(),B(), functor_type(defaultValue));
}


/// \brief computes the matrix-vector product x+=Av
///
/// The call to prod does not compute the product itself but instead, as all other expressions,
/// it returns an expression-object which can compute it. In contrast to other expression,
/// this expression is optimized to make use of well known mathematical identities to reduce run time of the algorithm.
template<class MatA, class VecV, class Device>
typename detail::matrix_vector_prod_optimizer<MatA,VecV>::type prod(
	matrix_expression<MatA, Device> const& A,vector_expression<VecV, Device> const& v
) {
	SIZE_CHECK(A().size2() == v().size());
	return detail::matrix_vector_prod_optimizer<MatA,VecV>::create(A(),v());
}

/// \brief computes the matrix-vector product x+=v^TA
///
/// it is computed via the identity (v^TA)^T= A^Tv
///
/// The call to prod does not compute the product itself but instead, as all other expressions,
/// it returns an expression-object which can compute it. In contrast to other expression,
/// this expression is optimized to make use of well known mathematical identities to reduce run time of the algorithm.
template<class MatA, class VecV, class Device>
auto prod(vector_expression<VecV, Device> const& v,matrix_expression<MatA, Device> const& A) -> decltype(prod(trans(A),v)){
	SIZE_CHECK(A().size1() == v().size());
	return prod(trans(A),v);
}

/// \brief Computes the matrix-vector product x+= alpha * Av or x= alpha * Av
///
/// A is interpreted as triangular matrix.
/// The first template argument governs the type
/// of triangular matrix: lower, upper, unit_lower and unit_upper.
///
///Example: x += triangular_prod<lower>(A,v);
template<class TriangularType, class MatA, class VecV, class Device>
matrix_vector_prod<detail::dense_triangular_proxy<MatA const,TriangularType> ,VecV> triangular_prod(
	matrix_expression<MatA, Device> const& A,
	vector_expression<VecV, Device>& v
) {
	SIZE_CHECK(A().size2() == v().size());
	typedef detail::dense_triangular_proxy<MatA const,TriangularType> Wrapper;
	return matrix_vector_prod<Wrapper ,VecV>(Wrapper(A()), v());
}

/// \brief computes the matrix-matrix product X+=AB
template<class MatA, class MatB, class Device>
matrix_matrix_prod<MatA,MatB> prod(
	matrix_expression<MatA, Device> const& A,
	matrix_expression<MatB, Device> const& B
) {
	SIZE_CHECK(A().size2() == B().size1());
	static_assert(std::is_base_of<linear_structure, typename MatA::orientation>::value, "A must be linearly stored");
	static_assert(std::is_base_of<linear_structure, typename MatB::orientation>::value, "B must be linearly stored");
	return matrix_matrix_prod<MatA,MatB>(A(),B());
}

/// \brief Computes the matrix-vector product x+= alpha * AB or x= alpha * AB
///
/// A is interpreted as triangular matrix.
/// The first template argument governs the type
/// of triangular matrix: lower, upper, unit_lower and unit_upper.
/// B is interpreted as dense matrix.
///
///Example: x += triangular_prod<lower>(A,v);
template<class TriangularType, class MatA, class MatB, class Device>
matrix_matrix_prod<detail::dense_triangular_proxy<MatA const,TriangularType> ,MatB>
triangular_prod(
	matrix_expression<MatA, Device> const& A,
	matrix_expression<MatB, Device> const& B
) {
	SIZE_CHECK(A().size2() == B().size1());
	static_assert(std::is_base_of<linear_structure, typename MatA::orientation>::value, "A must be linearly stored");
	static_assert(std::is_base_of<linear_structure, typename MatB::orientation>::value, "B must be linearly stored");
	typedef detail::dense_triangular_proxy<MatA const,TriangularType> Wrapper;
	return matrix_matrix_prod<Wrapper ,MatB>(Wrapper(A()), B());
}

namespace detail{

	//TODO: This is cpu-specific. has to be moved to kernels
	
	template<class MatA>
	typename MatA::scalar_type sum_impl(MatA const& A, column_major){
		typename MatA::scalar_type totalSum = 0;
		for(std::size_t j = 0; j != A.size2(); ++j){
			totalSum += sum(column(A,j));
		}
		return totalSum;
	}

	template<class MatA>
	typename MatA::scalar_type sum_impl(MatA const& A, row_major){
		typename MatA::scalar_type totalSum = 0;
		for(std::size_t i = 0; i != A.size1(); ++i){
			totalSum += sum(row(A,i));
		}
		return totalSum;
	}

	template<class MatA>
	typename MatA::scalar_type sum_impl(MatA const& A, unknown_orientation){
		return sum_impl(A,row_major());
	}


	//dispatcher for triangular matrix
	template<class MatA,class Orientation,class Triangular>
	typename MatA::scalar_type sum_impl(MatA const& A, triangular<Orientation,Triangular>){
		return sum_impl(A,Orientation());
	}

	//dispatcher
	template<class MatA>
	typename MatA::scalar_type sum_impl(MatA const& A){
		return sum_impl(A,typename MatA::orientation());
	}

	template<class MatA>
	typename MatA::scalar_type max_impl(MatA const& A, column_major){
		typename MatA::scalar_type maximum = 0;
		for(std::size_t j = 0; j != A.size2(); ++j){
			maximum = std::max(maximum, max(column(A,j)));
		}
		return maximum;
	}

	template<class MatA>
	typename MatA::scalar_type max_impl(MatA const& A, row_major){
		typename MatA::scalar_type maximum = 0;
		for(std::size_t i = 0; i != A.size1(); ++i){
			maximum= std::max(maximum, max(row(A,i)));
		}
		return maximum;
	}

	template<class MatA>
	typename MatA::scalar_type max_impl(MatA const& A, unknown_orientation){
		return max_impl(A,row_major());
	}

	//dispatcher for triangular matrix
	template<class MatA,class Orientation,class Triangular>
	typename MatA::scalar_type max_impl(MatA const& A, triangular<Orientation, Triangular>){
		return std::max(max_impl(A,Orientation()),0.0);
	}

	//dispatcher
	template<class MatA>
	typename MatA::scalar_type max_impl(MatA const& A){
		return max_impl(A,typename MatA::orientation());
	}

	template<class MatA>
	typename MatA::scalar_type min_impl(MatA const& A, column_major){
		typename MatA::scalar_type minimum = 0;
		for(std::size_t j = 0; j != A.size2(); ++j){
			minimum= std::min(minimum, min(column(A,j)));
		}
		return minimum;
	}

	template<class MatA>
	typename MatA::scalar_type min_impl(MatA const& A, row_major){
		typename MatA::scalar_type minimum = 0;
		for(std::size_t i = 0; i != A.size1(); ++i){
			minimum= std::min(minimum, min(row(A,i)));
		}
		return minimum;
	}

	template<class MatA>
	typename MatA::scalar_type min_impl(MatA const& A, unknown_orientation){
		return min_impl(A,row_major());
	}

	//dispatcher for triangular matrix
	template<class MatA,class Orientation,class Triangular>
	typename MatA::scalar_type min_impl(MatA const& A, triangular<Orientation,Triangular>){
		return std::min(min_impl(A,Orientation()),0.0);
	}

	//dispatcher
	template<class MatA>
	typename MatA::scalar_type min_impl(MatA const& A){
		return min_impl(A,typename MatA::orientation());
	}

}//end detail, END TODO: has to be moved to kernels


template<class MatA, class Device>
sum_matrix_rows<MatA>
sum_rows(matrix_expression<MatA, Device> const& A){
	return sum_matrix_rows<MatA>(A());
}

template<class MatA, class Device>
sum_matrix_rows<typename detail::matrix_transpose_optimizer<typename const_expression<MatA>::type >::type >
sum_columns(matrix_expression<MatA, Device> const& A){
	return sum_rows(trans(A));
}


template<class MatA, class Device>
typename MatA::scalar_type sum(matrix_expression<MatA, Device> const& A){
	return detail::sum_impl(eval_block(A));
}

template<class MatA, class Device>
typename MatA::scalar_type max(matrix_expression<MatA, Device> const& A){
	return detail::max_impl(eval_block(A));
}

template<class MatA, class Device>
typename MatA::scalar_type min(matrix_expression<MatA, Device> const& A){
	return detail::min_impl(eval_block(A));
}

/// \brief Returns the frobenius inner-product between matrices exprssions 1 and B.
///
///The frobenius inner product is defined as \f$ <A,B>_F=\sum_{ij} A_ij*B_{ij} \f$. It induces the
/// Frobenius norm \f$ ||A||_F = \sqrt{<A,A>_F} \f$
template<class MatA, class MatB, class Device>
decltype(typename MatA::scalar_type() * typename MatB::scalar_type())
frobenius_prod(
	matrix_expression<MatA, Device> const& A,
	matrix_expression<MatB, Device> const& B
) {
	SIZE_CHECK(A().size1() == B().size1());
	SIZE_CHECK(A().size2() == B().size2());
	return sum(eval_block(A*B));
}

/// \brief Computes the matrix 1-norm |A|_1
/// 
/// It is defined as \f$ \max_i \sum_j |A_{ij}| \f$ 
template<class MatA, class Device>
typename real_traits<typename MatA::scalar_type>::type
norm_1(matrix_expression<MatA, Device> const& A) {
	return max(sum_rows(abs(A)));
}

/// \brief computes the frobenius norm |A|_F
///
/// It is defined as \f$ \sqrt{Tr(A^TA)}=\sqrt{\sum_{ij} A_{ij}^2} \f$
template<class MatA, class Device>
typename real_traits<typename MatA::scalar_type>::type
norm_frobenius(matrix_expression<MatA, Device> const& A) {
	using std::sqrt;
	return sqrt(sum(abs_sqr(eval_block(A))));
}

/// \brief Computes the matrix inf-norm |A|_inf
/// 
/// It is defined as \f$ \max_i \sum_j |A_{ij}| \f$ 
template<class MatA, class Device>
typename real_traits<typename MatA::scalar_type>::type
norm_inf(matrix_expression<MatA, Device> const& A) {
	return max(sum_columns(abs(A)));
}

/// \brief Evaluates the trace of matrix A
///
/// The rtace is defined as the sum of the diagonal elements of A,
/// \f$ \text{trace}(A) = \sum_i A_{ii}\f$
///
/// \param  A square matrix
/// \return the sum of the values at the diagonal of \em A
template < class MatA, class Device>
typename MatA::scalar_type trace(matrix_expression<MatA, Device> const& A)
{
	SIZE_CHECK(A().size1() == A().size2());
	return sum(diag(A));
}

/** \brief An identity matrix with values of type \c T
 *
 * Elements or cordinates \f$(i,i)\f$ are equal to 1 (one) and all others to 0 (zero).
 */
template<class T>
class identity_matrix: public diagonal_matrix<scalar_vector<T> > {
	typedef diagonal_matrix<scalar_vector<T> > base_type;
public:
	identity_matrix(){}
	identity_matrix(std::size_t size):base_type(scalar_vector<T>(size,T(1))){}
};


template<class MatA, class Device>
diagonal_matrix<MatA> to_diagonal(blas::vector_expression<MatA, Device> const& A){
	return diagonal_matrix<MatA>(A());
}

}
}

#endif
