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
/////Broadcast
/////////////////////////////////////////////

namespace detail{
	template<unsigned N, class TensorA>
	auto broadcast_impl(TensorA const& A, std::size_t size){
		return detail::broadcast_optimizer<TensorA, N>::create(A, size);
	}
	template<unsigned N, class TensorA>
	auto broadcast_impl(TensorA const& A, ax::merge<1>){
		return A;
	}
	
	template<unsigned N, class TensorA>
	auto broadcast_dispatcher(TensorA const& A){
		return A;
	}
	
	template<unsigned N, class TensorA, class Arg, class... Args>
	auto broadcast_dispatcher(TensorA const& A, Arg arg, Args... args){
		auto Abc = broadcast_impl<N>(A,arg);
		return broadcast_dispatcher<N + 1>(Abc, args...);
	}
}


/// \brief Broadcast a tensor to a tensor with larger shape
///
/// Broadcasting is the operation of adding one or more additional axis to a tensor and copying
/// the vector multiple times along those axes. Possible arguments for axis descriptions are ax::same and positive integer values.
///
/// Example:
/// let A be a 3D tensor of shape (a,b,c) The call to
/// B= broadcast(A,ax::same, N, ax::same, M, ax::same)
///
/// will result in B being a 5 dimensional tensor with shape (a, N, b, M, c)
/// and behaviour B(i,j,k,l,m)=A(i,k,m)
///
/// ax::same arguments at the end can be discarded, therefore we could shorten the example above by:
///
/// B= broadcast(A,ax::same, N, ax::same, M)
///
/// Similarly, broadcasting a K dimensional vector as a matrix with N rows can be achieved via:
/// B= broadcast(A,N)
/// whereas we get a K xN matrix via
/// B= broadcast(A,ax::same, N)
///
/// It is allowed to add multiple consecutive new axes:
/// B= broadcast(A,N1, N2)
/// Would lead to tensor with shape (N1, N2, K) 
template<std::size_t Dim, class TensorA, class Device, class... Args>
auto broadcast(tensor_expression<Dim, TensorA, Device> const& A, Args... args){
	return detail::broadcast_dispatcher<0u>(typename TensorA::const_closure_type(A()), args...);
}

/////////////////////////////////////////////
/////Tensor-Unary Operations
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
/////Elementwise Binary-Tensor Operations
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

template<std::size_t Dim, class TensorA, class TensorB, class Scalar, class Device>
auto safe_div(
	tensor_expression<Dim, TensorA, Device> const& A, 
	tensor_expression<Dim, TensorB, Device> const& B, 
	scalar_expression<Scalar, Device> const& defaultValue
){
	REMORA_SIZE_CHECK(A().shape() == B().shape());
	typedef typename common_value_type<TensorA,TensorB>::type result_type;
	typedef typename device_traits<Device>:: template safe_divide<result_type> functor_type;
	return tensor_binary<TensorA, TensorB, functor_type>(A(),B(), functor_type(defaultValue()));
}


#define REMORA_BINARY_TENSOR_EXPRESSION(name, F)\
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
REMORA_BINARY_TENSOR_EXPRESSION(operator*, multiply)
REMORA_BINARY_TENSOR_EXPRESSION(operator/, divide)
REMORA_BINARY_TENSOR_EXPRESSION(operator<, less)
REMORA_BINARY_TENSOR_EXPRESSION(operator<=, less_equal)
REMORA_BINARY_TENSOR_EXPRESSION(operator>, greater)
REMORA_BINARY_TENSOR_EXPRESSION(operator>=, greater_equal)
REMORA_BINARY_TENSOR_EXPRESSION(operator==, equal)
REMORA_BINARY_TENSOR_EXPRESSION(operator!=, not_equal)
REMORA_BINARY_TENSOR_EXPRESSION(pow,pow)
REMORA_BINARY_TENSOR_EXPRESSION(min,min)
REMORA_BINARY_TENSOR_EXPRESSION(max,max)
#undef REMORA_BINARY_TENSOR_EXPRESSION


/////////////////////////////////////////////
/////Tensor-Scalar Operations
/////////////////////////////////////////////

/// \brief Computes the multiplication of a tensor-expression A with a t t.
///
/// \f$ (A*t)_{i...} = a_{i...}*t \f$
template<std::size_t Dim, class TensorA, class T, class Device>
typename std::enable_if<
	std::is_arithmetic<T>::value,
	typename detail::scalar_multiply_optimizer<TensorA>::type
>::type
operator* (tensor_expression<Dim, TensorA, Device> const& A, T t){
	return detail::scalar_multiply_optimizer<TensorA>::create(A(), typename TensorA::value_type(t));
}
template<std::size_t Dim, class Scalar, class TensorA, class Device>
typename std::enable_if<
	(Dim > 0), typename detail::scalar_multiply_optimizer<TensorA>::type
>::type
operator* (tensor_expression<Dim, TensorA, Device> const& A, scalar_expression<Scalar, Device> const& t){
	return detail::scalar_multiply_optimizer<TensorA>::create(A(), typename TensorA::value_type(t()));
}




/// \brief Computes the multiplication of a tensor-expression A with a t t.
///
/// \f$ (t*A)_{ij} = t*e_{ij} \f$
template<std::size_t Dim, class T, class TensorA, class Device>
typename std::enable_if<
	std::is_arithmetic<T>::value,
        typename detail::scalar_multiply_optimizer<TensorA>::type
>::type
operator* (T t, tensor_expression<Dim, TensorA, Device> const& A){
	return detail::scalar_multiply_optimizer<TensorA>::create(A(), typename TensorA::value_type(t));
}

template<std::size_t Dim, class Scalar, class TensorA, class Device>
typename std::enable_if<
	(Dim > 0), typename detail::scalar_multiply_optimizer<TensorA>::type
>::type
operator* (scalar_expression<Scalar, Device> const& t, tensor_expression<Dim, TensorA, Device> const& A){
	return detail::scalar_multiply_optimizer<TensorA>::create(A(), typename TensorA::value_type(t()));
}


///\brief Adds a tensor plus a t which is interpreted as a constant tensor
template<std::size_t Dim, class TensorA, class T, class Device>
typename std::enable_if<
	std::is_arithmetic<T>::value, 
	tensor_addition<TensorA, scalar_tensor<T, typename TensorA::axis, Device> >
>::type operator+ (
	tensor_expression<Dim, TensorA, Device> const& A,
	T t
){
	return A + scalar_tensor<T, typename TensorA::axis, Device>(A().shape(),t);
}

template<std::size_t Dim, class TensorA, class Scalar, class Device>
typename std::enable_if<
	(Dim > 0), 
	tensor_addition<TensorA, scalar_tensor<typename Scalar::value_type, typename TensorA::axis, Device> >
>::type operator+ (
	tensor_expression<Dim, TensorA, Device> const& A,
	scalar_expression<Scalar, Device> const& t
){
	return A + scalar_tensor<typename Scalar::value_type, typename TensorA::axis, Device>(A().shape(),t());
}

///\brief Adds a tensor plus a t which is interpreted as a constant tensor
template<std::size_t Dim, class T, class TensorA, class Device>
typename std::enable_if<
	std::is_arithmetic<T>::value,
	tensor_addition<TensorA, scalar_tensor<T, typename TensorA::axis, Device> >
>::type operator+ (
	T t,
	tensor_expression<Dim, TensorA, Device> const& A
){
	return A + scalar_tensor<T, typename TensorA::axis, Device>(A().shape(),t);
}

template<std::size_t Dim, class TensorA, class Scalar, class Device>
typename std::enable_if<
	(Dim > 0), 
	tensor_addition<TensorA, scalar_tensor<typename Scalar::value_type, typename TensorA::axis, Device> >
>::type operator+ (
	scalar_expression<Scalar, Device> const& t,
	tensor_expression<Dim, TensorA, Device> const& A
){
	return A + scalar_tensor<typename Scalar::value_type, typename TensorA::axis, Device>(A().shape(),t());
}



///\brief Subtracts a t which is interpreted as a constant tensor from a tensor.
template<std::size_t Dim, class TensorA, class T, class Device>
typename std::enable_if<
	std::is_arithmetic<T>::value,
	decltype(std::declval<TensorA const&>() + T())
>::type operator- (
	tensor_expression<Dim, TensorA, Device> const& A,
	T t
){
	return A + (-t);
}

template<std::size_t Dim, class TensorA, class Scalar, class Device>
typename std::enable_if<
	(Dim > 0),
	decltype(std::declval<TensorA const&>() + typename Scalar::value_type())
>::type operator- (
	tensor_expression<Dim, TensorA, Device> const& A,
	scalar_expression<Scalar, Device> const& t
){
	return A + (-t());
}



///\brief Subtracts a tensor from a t which is interpreted as a constant tensor
template<std::size_t Dim, class TensorA, class T, class Device>
typename std::enable_if<
	std::is_arithmetic<T>::value,
	decltype(T() + (-std::declval<TensorA const&>()))
>::type operator- (
	T t,
	tensor_expression<Dim, TensorA, Device> const& A
){
	return t + (-A);
}

template<std::size_t Dim, class TensorA, class Scalar, class Device>
typename std::enable_if<
	(Dim > 0),
	decltype(typename Scalar::value_type() + (-std::declval<TensorA const&>()))
>::type operator- (
	scalar_expression<Scalar, Device> const& t,
	tensor_expression<Dim, TensorA, Device> const& A
){
	return t() + (-A);
}





#define REMORA_TENSOR_SCALAR_TRANSFORMATION(name, F)\
template<std::size_t Dim, class T, class TensorA, class Device> \
typename std::enable_if< \
	std::is_arithmetic<T>::value,\
	tensor_binary<TensorA, scalar_tensor<typename TensorA::value_type, typename TensorA::axis, Device>,typename device_traits<Device>:: template  F<typename TensorA::value_type> > \
>::type \
name (tensor_expression<Dim, TensorA, Device> const& m, T t){ \
	typedef typename TensorA::value_type type;\
	typedef typename device_traits<Device>:: template F<type> functor_type;\
	typedef scalar_tensor<type, typename TensorA::axis, Device> mat_type;\
	return tensor_binary<TensorA, mat_type, functor_type >(m(), mat_type(m().shape(), type(t)) ,functor_type()); \
}\
template<std::size_t Dim, class Scalar, class TensorA, class Device> \
typename std::enable_if< \
	(Dim > 0),\
	tensor_binary<TensorA, scalar_tensor<typename TensorA::value_type, typename TensorA::axis, Device>,typename device_traits<Device>:: template  F<typename TensorA::value_type> > \
>::type \
name (tensor_expression<Dim, TensorA, Device> const& m, scalar_expression<Scalar, Device> const& t){ \
	typedef typename TensorA::value_type type;\
	typedef typename device_traits<Device>:: template F<type> functor_type;\
	typedef scalar_tensor<type, typename TensorA::axis, Device> mat_type;\
	return tensor_binary<TensorA, mat_type, functor_type >(m(), mat_type(m().shape(), type(t())) ,functor_type()); \
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
	std::is_arithmetic<T>::value,\
	tensor_binary<scalar_tensor< typename TensorA::value_type, typename TensorA::axis, Device>, TensorA, typename device_traits<Device>:: template F< typename TensorA::value_type> > \
>::type \
name (T t, tensor_expression<Dim, TensorA, Device> const& m){ \
	typedef typename TensorA::value_type type;\
	typedef typename device_traits<Device>:: template F<type> functor_type;\
	typedef scalar_tensor<type, typename TensorA::axis, Device> mat_type;\
	return  tensor_binary<mat_type, TensorA, functor_type >(mat_type(m().shape(), t), m(), functor_type()); \
}\
template<std::size_t Dim, class Scalar, class TensorA, class Device> \
typename std::enable_if< \
	(Dim > 0),\
	tensor_binary<scalar_tensor< typename TensorA::value_type, typename TensorA::axis, Device>, TensorA, typename device_traits<Device>:: template F< typename TensorA::value_type> > \
>::type \
name (scalar_expression<Scalar, Device> const& t, tensor_expression<Dim, TensorA, Device> const& m){ \
	typedef typename TensorA::value_type type;\
	typedef typename device_traits<Device>:: template F<type> functor_type;\
	typedef scalar_tensor<type, typename TensorA::axis, Device> mat_type;\
	return  tensor_binary<mat_type, TensorA, functor_type >(mat_type(m().shape(), t()), m(), functor_type()); \
}
REMORA_TENSOR_SCALAR_TRANSFORMATION_2(min, min)
REMORA_TENSOR_SCALAR_TRANSFORMATION_2(max, max)
REMORA_TENSOR_SCALAR_TRANSFORMATION_2(operator/, divide)
#undef REMORA_TENSOR_SCALAR_TRANSFORMATION_2


/////////////////////////////////////////
/////TENSOR REDUCTIONS
/////////////////////////////////////////


namespace detail{
	template<std::size_t Dim, class TensorA, class Device, unsigned N, class F>
	auto reduce(tensor_expression<Dim, TensorA, Device> const& A, axis_set<N> ax, F f){
		//permute the axis to the last position
		auto A_last = permute_axis_back(A, ax);
		return detail::tensor_reduce_last_optimizer<decltype(A_last), F>::create(A_last, f);
	}

	template<std::size_t Dim, class TensorA, class Device, unsigned N, unsigned... Ns, class F>
	auto reduce(tensor_expression<Dim, TensorA, Device> const& A, axis_set<N, Ns...>, F f){
		auto Areduced = reduce(A, axis_set<N>(), f);
		return reduce(Areduced, axis_set<(Ns>N? Ns-1: Ns)...>(), f);
	}
	
	template<class TensorA, class Device, class F>
	auto reduce(tensor_expression<0, TensorA, Device> const& A, axis_set<>, F f){
		return typename TensorA::const_closure_type(A());
	}
}

/// \brief Computes the elementwise sum over the elements of the chosen Axes of A
template<std::size_t Dim, class TensorA, class Device, unsigned... Ns>
auto sum(tensor_expression<Dim, TensorA, Device> const& A, axis_set<Ns...> ax){
	typedef typename TensorA::value_type value_type;
	typedef typename device_traits<Device>::template add<value_type> Add;
	return detail::reduce(A, ax, Add());
}
/// \brief Computes the elementwise sum over the elements of A
template<std::size_t Dim, class TensorA, class Device>
auto sum(tensor_expression<Dim, TensorA, Device> const& A){
	typedef typename TensorA::value_type value_type;
	typedef typename device_traits<Device>::template add<value_type> Add;
	return detail::reduce(A, typename TensorA::axis::inverse_t(), Add());
}


/// \brief Computes the elementwise maximum over the elements of the chosen Axes of A
template<std::size_t Dim, class TensorA, class Device, unsigned... Ns>
auto max(tensor_expression<Dim, TensorA, Device> const& A, axis_set<Ns...> ax){
	typedef typename TensorA::value_type value_type;
	typedef typename device_traits<Device>::template max<value_type> Max;
	return detail::reduce(A, ax, Max());
}
/// \brief Computes the elementwise maximum over the elements of A
template<std::size_t Dim, class TensorA, class Device>
auto max(tensor_expression<Dim, TensorA, Device> const& A){
	typedef typename TensorA::value_type value_type;
	typedef typename device_traits<Device>::template max<value_type> Max;
	return detail::reduce(A, typename TensorA::axis::inverse_t(), Max());
}

/// \brief Computes the elementwise minimum over the elements of the chosen Axes of A
template<std::size_t Dim, class TensorA, class Device, unsigned... Ns>
auto min(tensor_expression<Dim, TensorA, Device> const& A, axis_set<Ns...> ax){
	typedef typename TensorA::value_type value_type;
	typedef typename device_traits<Device>::template min<value_type> Min;
	return detail::reduce(A, ax, Min());
}
/// \brief Computes the elementwise minimum over the elements of A
template<std::size_t Dim, class TensorA, class Device, unsigned... Ns>
auto min(tensor_expression<Dim, TensorA, Device> const& A){
	typedef typename TensorA::value_type value_type;
	typedef typename device_traits<Device>::template min<value_type> Min;
	return detail::reduce(A, typename TensorA::axis::inverse_t(), Min());
}

/// \brief Evaluates the trace of tensor A over two selected axis
///
/// The trace is defined as the sum of the diagonal elements of A,
/// \f$ \text{trace}(A) = \sum_i A_{ii}\f$
/// The ax argument chooses the axis (i,j) over which the diagonal is computed
/// If the input is a D-dimensional tensor, the result is D-2 dimensional which holds the result of the trace. 
template <std::size_t Dim,  class TensorA, class Device, unsigned N0, unsigned N1>
auto trace(tensor_expression<Dim, TensorA, Device> const& A, axis_set<N0, N1> ax){
	static_assert(N0 < Dim);
	static_assert(N1 < Dim);
	static_assert(N0 != N1);
	//diag permutes the last two axes of A to the back and afterwards removes one axes. so we sum over the new last axis.
	return sum(diag(A, ax), axis_set<Dim - 2>());
}
/// \brief Evaluates the trace of tensor A
///
/// The trace is defined as the sum of the diagonal elements of A,
/// \f$ \text{trace}(A) = \sum_i A_{ii}\f$
/// The last two axes are summed over. If the input is a D-dimensional tensor,
/// the result is D-2 dimensional which holds the result of the trace. 
template <std::size_t Dim,  class TensorA, class Device>
auto trace(tensor_expression<Dim, TensorA, Device> const& A){
	static_assert(Dim >= 2);
	return trace(A, axis_set<Dim - 2, Dim - 1>());
}

/////////////////////////////////////////
/////Matrix-Products
/////////////////////////////////////////


//todo: outer_product

/// \brief Returns an expression that computes the inner product v1^T v2
template<class VecV1, class VecV2, class Device>
auto operator%(
	vector_expression<VecV1, Device> const& v1,vector_expression<VecV2, Device> const& v2
) {
	REMORA_SIZE_CHECK(v1().shape() == v2().shape());
	
	return sum(eval_block(v1 * v2));
}


/// \brief Returns an expression that computes the matrix-vector product Av
template<class MatA, class VecV, class Device>
auto operator%(
	matrix_expression<MatA, Device> const& A,vector_expression<VecV, Device> const& v
) {
	REMORA_SIZE_CHECK(A().shape()[1] == v().shape()[0]);
	return detail::tensor_prod_reduce_optimizer<MatA,VecV, axis<0> >::create(A(), v(), 1);
}

/// \brief Returns an expression that computes the matrix-vector product v^TA
template<class MatA, class VecV, class Device>
auto operator%(vector_expression<VecV, Device> const& v,matrix_expression<MatA, Device> const& A){
	REMORA_SIZE_CHECK(A().shape()[0] == v().shape()[0]);
	return detail::tensor_prod_reduce_optimizer<VecV, MatA, axis<0> >::create(v(), A(), 1);
}

/// \brief Returns an expression that computes the matrix-vector product AB
template<class MatA, class MatB, class Device>
auto operator%(
	matrix_expression<MatA, Device> const& A,matrix_expression<MatB, Device> const& B
) {
	REMORA_SIZE_CHECK(A().shape()[1] == B().shape()[0]);
	return detail::tensor_prod_reduce_optimizer<MatA,MatB, axis<0,1> >::create(A(), B(), 1);
}


}
#endif
