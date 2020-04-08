/*!
 * \brief       Expression templates for expressions involving tensors of arbitrary dimensions
 * 
 * \author      O. Krause
 * \date        2020
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
#ifndef REMORA_EXPRESSIONS_HPP
#define REMORA_EXPRESSIONS_HPP

#include "detail/expression_optimizers.hpp"
#include "proxy_expressions.hpp"

namespace remora{

/////////////////////////////////////////////
//////////Tensor-Unary Operations
/////////////////////////////////////////////


template<std::size_t Dim, class TensorA, class Device>
auto operator-(tensor_expression<Dim, TensorA, Device> const& A){
	return detail::scalar_multiply_optimizer<TensorA>::create(A(), typename TensorA::value_type(-1));
}

#define REMORA_UNARY_TENSOR_TRANSFORMATION(name, F)\
template<std::size_t Dim, class TensorA, class Device>\
auto name(tensor_expression<Dim, TensorA, Device> const& m){\
	typedef typename device_traits<Device>:: template F<typename TensorA::value_type> functor_type;\
	return detail::tensor_unary_optimizer<TensorA, functor_type >::create(m(), functor_type());\
}

REMORA_UNARY_TENSOR_TRANSFORMATION(abs, abs)
REMORA_UNARY_TENSOR_TRANSFORMATION(log, log)
REMORA_UNARY_TENSOR_TRANSFORMATION(exp, exp)
REMORA_UNARY_TENSOR_TRANSFORMATION(tanh,tanh)
REMORA_UNARY_TENSOR_TRANSFORMATION(sin,sin)
REMORA_UNARY_TENSOR_TRANSFORMATION(cos,cos)
REMORA_UNARY_TENSOR_TRANSFORMATION(tan,tan)
REMORA_UNARY_TENSOR_TRANSFORMATION(asin,asin)
REMORA_UNARY_TENSOR_TRANSFORMATION(acos,acos)
REMORA_UNARY_TENSOR_TRANSFORMATION(atan,atan)
REMORA_UNARY_TENSOR_TRANSFORMATION(erf,erf)
REMORA_UNARY_TENSOR_TRANSFORMATION(erfc,erfc)
REMORA_UNARY_TENSOR_TRANSFORMATION(sqr, sqr)
REMORA_UNARY_TENSOR_TRANSFORMATION(sqrt, sqrt)
REMORA_UNARY_TENSOR_TRANSFORMATION(cbrt, cbrt)
REMORA_UNARY_TENSOR_TRANSFORMATION(sigmoid, sigmoid)
REMORA_UNARY_TENSOR_TRANSFORMATION(softPlus, soft_plus)
REMORA_UNARY_TENSOR_TRANSFORMATION(elem_inv, inv)
#undef REMORA_UNARY_TENSOR_TRANSFORMATION


/////////////////////////////////////////////
//////////Simple Matrix-Binary Operations
/////////////////////////////////////////////


///\brief Adds two Tensors
template<std::size_t Dim, class TensorA, class TensorB, class Device>
auto operator+ (
	tensor_expression<Dim, TensorA, Device> const& A,
	tensor_expression<Dim, TensorB, Device> const& B
){
	REMORA_SIZE_CHECK(A().shape() == B().shape());
	return tensor_addition<TensorA, TensorB>(A(),B());
}

///\brief Subtracts two Tensors
template<std::size_t Dim, class TensorA, class TensorB, class Device>
auto operator- (
	tensor_expression<Dim, TensorA, Device> const& A,
	tensor_expression<Dim, TensorB, Device> const& B
) -> decltype(A() + (-B)){
	REMORA_SIZE_CHECK(A().shape() == B().shape());
	return A() + (-B);
}

template<std::size_t Dim, class TensorA, class TensorB, class Device>
auto safe_div(
	tensor_expression<Dim, TensorA, Device> const& A, 
	tensor_expression<Dim, TensorB, Device> const& B, 
	typename common_value_type<TensorA,TensorB>::type defaultValue
){
	REMORA_SIZE_CHECK(A().shape() == B().shape());
	typedef typename common_value_type<TensorA,TensorB>::type result_type;
	typedef typename device_traits<Device>:: template safe_divide<result_type> functor_type;
	return tensor_binary<TensorA, TensorB, functor_type>(A(),B(), functor_type(defaultValue));
}


#define REMORA_BINARY_MATRIX_EXPRESSION(name, F)\
template<std::size_t Dim, class TensorA, class TensorB, class Device>\
auto name(\
	tensor_expression<Dim, TensorA, Device> const& A,\
	tensor_expression<Dim, TensorB, Device> const& B \
){\
	REMORA_SIZE_CHECK(A().shape() == B().shape());\
	typedef typename common_value_type<TensorA,TensorB>::type type;\
	typedef typename device_traits<Device>:: template F<type> functor_type;\
	return tensor_binary<TensorA, TensorB, functor_type >(A(),B(), functor_type());\
}
REMORA_BINARY_MATRIX_EXPRESSION(operator*, multiply)
REMORA_BINARY_MATRIX_EXPRESSION(operator/, divide)
REMORA_BINARY_MATRIX_EXPRESSION(pow,pow)
REMORA_BINARY_MATRIX_EXPRESSION(min,min)
REMORA_BINARY_MATRIX_EXPRESSION(max,max)
#undef REMORA_BINARY_MATRIX_EXPRESSION


/////////////////////////////////////////////
//////////Tensor-Scalar Operations
/////////////////////////////////////////////

/// \brief Computes the multiplication of a tensor-expression A with a scalar t.
///
/// \f$ (A*t)_{i...} = a_{i...}*t \f$
template<std::size_t Dim, class TensorA, class T, class Device>
typename std::enable_if<
	std::is_convertible<T, typename TensorA::value_type >::value,
	typename detail::scalar_multiply_optimizer<TensorA>::type
>::type
operator* (tensor_expression<Dim, TensorA, Device> const& A, T scalar){
	return detail::scalar_multiply_optimizer<TensorA>::create(A(), typename TensorA::value_type(scalar));
}

/// \brief Computes the multiplication of a tensor-expression A with a scalar t.
///
/// \f$ (t*A)_{ij} = t*e_{ij} \f$
template<std::size_t Dim, class T, class TensorA, class Device>
typename std::enable_if<
	std::is_convertible<T, typename TensorA::value_type >::value,
        typename detail::scalar_multiply_optimizer<TensorA>::type
>::type
operator* (T scalar, tensor_expression<Dim, TensorA, Device> const& A){
	return detail::scalar_multiply_optimizer<TensorA>::create(A(), typename TensorA::value_type(scalar));
}


///\brief Adds a tensor plus a scalar which is interpreted as a constant tensor
template<std::size_t Dim, class TensorA, class T, class Device>
typename std::enable_if<
	std::is_convertible<T, typename TensorA::value_type>::value, 
	tensor_addition<TensorA, scalar_tensor<T, typename TensorA::axis, Device> >
>::type operator+ (
	tensor_expression<Dim, TensorA, Device> const& A,
	T t
){
	return A + scalar_tensor<T, typename TensorA::axis, Device>(A().shape(),t);
}

///\brief Adds a tensor plus a scalar which is interpreted as a constant tensor
template<std::size_t Dim, class T, class TensorA, class Device>
typename std::enable_if<
	std::is_convertible<T, typename TensorA::value_type>::value,
	tensor_addition<TensorA, scalar_tensor<T, typename TensorA::axis, Device> >
>::type operator+ (
	T t,
	tensor_expression<Dim, TensorA, Device> const& A
){
	return A + scalar_tensor<T, typename TensorA::axis, Device>(A().shape(),t);
}

///\brief Subtracts a scalar which is interpreted as a constant tensor from a tensor.
template<std::size_t Dim, class TensorA, class T, class Device>
typename std::enable_if<
	std::is_convertible<T, typename TensorA::value_type>::value ,
	decltype(std::declval<TensorA const&>() + T())
>::type operator- (
	tensor_expression<Dim, TensorA, Device> const& A,
	T t
){
	return A + (-t);
}

///\brief Subtracts a tensor from a scalar which is interpreted as a constant tensor
template<std::size_t Dim, class TensorA, class T, class Device>
typename std::enable_if<
	std::is_convertible<T, typename TensorA::value_type>::value,
	decltype(T() + (-std::declval<TensorA const&>()))
>::type operator- (
	T t,
	tensor_expression<Dim, TensorA, Device> const& A
){
	return t + (-A);
}

#define REMORA_TENSOR_SCALAR_TRANSFORMATION(name, F)\
template<std::size_t Dim, class T, class TensorA, class Device> \
typename std::enable_if< \
	std::is_convertible<T, typename TensorA::value_type >::value,\
        tensor_binary<TensorA, scalar_tensor<typename TensorA::value_type, typename TensorA::axis, Device>,typename device_traits<Device>:: template  F<typename TensorA::value_type> > \
>::type \
name (tensor_expression<Dim, TensorA, Device> const& m, T t){ \
	typedef typename TensorA::value_type type;\
	typedef typename device_traits<Device>:: template F<type> functor_type;\
	typedef scalar_tensor<type, typename TensorA::axis, Device> mat_type;\
	return tensor_binary<TensorA, mat_type, functor_type >(m(), mat_type(m().shape(), type(t)) ,functor_type()); \
}
REMORA_TENSOR_SCALAR_TRANSFORMATION(operator/, divide)
REMORA_TENSOR_SCALAR_TRANSFORMATION(operator<, less)
REMORA_TENSOR_SCALAR_TRANSFORMATION(operator<=, less_equal)
REMORA_TENSOR_SCALAR_TRANSFORMATION(operator>, greater)
REMORA_TENSOR_SCALAR_TRANSFORMATION(operator>=, greater_equal)
REMORA_TENSOR_SCALAR_TRANSFORMATION(operator==, equal)
REMORA_TENSOR_SCALAR_TRANSFORMATION(operator!=, not_equal)
REMORA_TENSOR_SCALAR_TRANSFORMATION(min, min)
REMORA_TENSOR_SCALAR_TRANSFORMATION(max, max)
REMORA_TENSOR_SCALAR_TRANSFORMATION(pow, pow)
#undef REMORA_TENSOR_SCALAR_TRANSFORMATION

// operations of the form op(t,v)[i,j] = op(t,v[i,j])
#define REMORA_TENSOR_SCALAR_TRANSFORMATION_2(name, F)\
template<std::size_t Dim, class T, class TensorA, class Device> \
typename std::enable_if< \
	std::is_convertible<T, typename TensorA::value_type >::value,\
	tensor_binary<scalar_tensor< typename TensorA::value_type, typename TensorA::axis, Device>, TensorA, typename device_traits<Device>:: template F< typename TensorA::value_type> > \
>::type \
name (T t, tensor_expression<Dim, TensorA, Device> const& m){ \
	typedef typename TensorA::value_type type;\
	typedef typename device_traits<Device>:: template F<type> functor_type;\
	typedef scalar_tensor<type, typename TensorA::axis, Device> mat_type;\
	return  tensor_binary<mat_type, TensorA, functor_type >(mat_type(m().shape(), t), m(), functor_type()); \
}
REMORA_TENSOR_SCALAR_TRANSFORMATION_2(min, min)
REMORA_TENSOR_SCALAR_TRANSFORMATION_2(max, max)
#undef REMORA_MATRIX_SCALAR_TRANSFORMATION_2


/////////////////////////////////////////
//////////TENSOR REDUCTIONS
/////////////////////////////////////////
/*
/// \brief Computes the elementwise sum over all elements of A
///
/// returns a scalar s = sum_ij A_ij
template<class TensorA, class Device>
typename TensorA::value_type sum(tensor_expression<TensorA, Device> const& A){
	typedef typename std::conditional<
		std::is_same<typename TensorA::axis , unknown_axis>::value,
		row_major,
		typename TensorA::axis 
	>::type axis;
	//sum first tensor-rows/columns together followed by summing those results
	return sum(sum(as_set(A, axis())));
}



/// \brief Computes the elementwise maximum over all elements of A
///
/// returns a scalar s = max_ij A_ij
template<class TensorA, class Device>
typename TensorA::value_type max(tensor_expression<TensorA, Device> const& A){
	typedef typename std::conditional<
		std::is_same<typename TensorA::axis , unknown_axis>::value,
		row_major,
		typename TensorA::axis 
	>::type axis;
	//compute first maximum of tensor-rows/columns and take the maximum of those results
	return max(max(as_set(A, axis())));
}

/// \brief Computes the elementwise minimum over all elements of A
///
/// returns a scalar s = min_ij A_ij
template<class TensorA, class Device>
typename TensorA::value_type min(tensor_expression<TensorA, Device> const& A){
	typedef typename std::conditional<
		std::is_same<typename TensorA::axis , unknown_axis>::value,
		row_major,
		typename TensorA::axis 
	>::type axis;
	//compute first minimum of tensor-rows/columns and take the minimum of those results
	return min(min(as_set(A, axis())));
}

/// \brief Returns the frobenius inner-product between matrices exprssions 1 and B.
///
///The frobenius inner product is defined as \f$ <A,B>_F=\sum_{ij} A_ij*B_{ij} \f$. It induces the
/// Frobenius norm \f$ ||A||_F = \sqrt{<A,A>_F} \f$
template<class TensorA, class TensorB, class Device>
decltype(typename TensorA::value_type() * typename TensorB::value_type())
frobenius_prod(
	tensor_expression<TensorA, Device> const& A,
	tensor_expression<TensorB, Device> const& B
){
	REMORA_SIZE_CHECK(A().shape() == B().shape());
	return sum(A*B);
}

/// \brief Computes the tensor 1-norm |A|_1
/// 
/// It is defined as \f$ \max_i \sum_j |A_{ij}| \f$ 
template<class TensorA, class Device>
typename real_traits<typename TensorA::value_type>::type
norm_1(tensor_expression<TensorA, Device> const& A) {
	return max(norm_1(as_columns(A)));
}

/// \brief computes the frobenius norm |A|_F
///
/// It is defined as \f$ \sqrt{Tr(A^TA)}=\sqrt{\sum_{ij} A_{ij}^2} \f$
template<class TensorA, class Device>
typename real_traits<typename TensorA::value_type>::type
norm_frobenius(tensor_expression<TensorA, Device> const& A) {
	using std::sqrt;
	return sqrt(sum(sqr(eval_block(A))));
}

/// \brief Computes the tensor inf-norm |A|_inf
/// 
/// It is defined as \f$ \max_i \sum_j |A_{ij}| \f$ 
template<class TensorA, class Device>
typename real_traits<typename TensorA::value_type>::type
norm_inf(tensor_expression<TensorA, Device> const& A) {
	return max(norm_1(as_rows(A)));
}

/// \brief Evaluates the trace of tensor A
///
/// The trace is defined as the sum of the diagonal elements of A,
/// \f$ \text{trace}(A) = \sum_i A_{ii}\f$
///
/// \param  A square tensor
/// \return the sum of the values at the diagonal of \em A
template < class TensorA, class Device>
typename TensorA::value_type trace(tensor_expression<TensorA, Device> const& A){
	REMORA_SIZE_CHECK(A().size1() == A().size2());
	return sum(diag(A));
}
*/

}
#endif