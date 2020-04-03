/*!
 * 
 *
 * \brief      Assignment and evaluation of tensor expressions
 * 
 * 
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

#ifndef REMORA_ASSIGNMENT_HPP
#define REMORA_ASSIGNMENT_HPP

#include "kernels/assign.hpp"
#include "kernels/device_traits.hpp"
#include "detail/traits.hpp"
#include "detail/proxy_optimizers_fwd.hpp"

#include <type_traits>
namespace remora{
	
//////////////////////////////////////////////////////////////////////
/////Evaluate expressions
//////////////////////////////////////////////////////////////////////

///\brief conditionally evaluates a tensor expression if it is a block expression
///
/// If the expression is a block expression, a temporary tensor is created to which
/// the expression is assigned, which is then returned, otherwise the expression itself
/// is returned
template<std::size_t N, class E, class Device>
typename std::conditional<
	std::is_base_of<
		blockwise_tag,
		typename E::evaluation_category
	>::value,
	typename tensor_temporary<E>::type,
	E const&
>::type
eval_block(tensor_expression<N, E, Device> const& e){
	return e();//either casts to E const& or returns the copied expression
}

///\brief Evaluates an expression if it does not have a standard storage layout
///
/// This function evaluates an expression to a temporary if it does not have
/// a known storage type. i.e. proxy expressions and containers are not evaluated but passed
/// through while everything else is evaluated.
template<std::size_t N, class E, class Device>
typename std::conditional<
	std::is_same<
		unknown_storage,
		typename E::storage_type
	>::value,
	typename tensor_temporary<E>::type,
	E const&
>::type
eval_expression(tensor_expression<N, E, Device> const& e){
	return e();//either casts to E const& or returns the evaluated expression
}
	
/////////////////////////////////////////////////////////////////////////////////////
////// Assign
////////////////////////////////////////////////////////////////////////////////////
	
namespace detail{
	template<std::size_t N, class TensorX, class TensorV, class Device>
	void assign(
		tensor_expression<N, TensorX, Device>& x, tensor_expression<N, TensorV, Device> const& v,
		elementwise_tag
	){
		kernels::assign(x, v);
	}
	template<std::size_t N, class TensorX, class TensorV, class Device>
	void assign(
		tensor_expression<N, TensorX, Device>& x, tensor_expression<N, TensorV, Device> const& v,
		blockwise_tag
	){
		v().assign_to(x);
	}
	template<std::size_t N, class TensorX, class TensorV, class Device>
	void plus_assign(tensor_expression<N, TensorX, Device>& x, tensor_expression<N, TensorV, Device> const& v,
		elementwise_tag
	){
		typename device_traits<Device>:: template add<typename common_value_type<TensorX,TensorV>::type> f;
		kernels::assign(x, v, f);
	}
	template<std::size_t N, class TensorX, class TensorV, class Device>
	void plus_assign(
		tensor_expression<N, TensorX, Device>& x, tensor_expression<N, TensorV, Device> const& v,
		blockwise_tag
	){
		v().plus_assign_to(x);
	}
}

/// \brief Dispatches tensor assignment on an expression level
///
/// This dispatcher takes care for whether the blockwise evaluation
/// or the elementwise evaluation is called
template<std::size_t N, class TensorX, class TensorV, class Device>
TensorX& assign(tensor_expression<N, TensorX, Device>& x, tensor_expression<N, TensorV, Device> const& v){
	REMORA_SIZE_CHECK(x().shape() == v().shape());
	detail::assign(x,v,typename TensorV::evaluation_category());
	return x();
}

/// \brief Dispatches tensor plus-assignment on an expression level
///
/// This dispatcher takes care for whether the blockwise evaluation
/// or the elementwise evaluation is called
template<std::size_t N, class TensorX, class TensorV, class Device>
TensorX& plus_assign(tensor_expression<N, TensorX, Device>& x, tensor_expression<N, TensorV, Device> const& v){
	REMORA_SIZE_CHECK(x().shape() == v().shape());
	detail::plus_assign(x,v,typename TensorV::evaluation_category());
	return x();
}

/// \brief Dispatches tensor minus-assignment on an expression level
///
/// This dispatcher takes care for whether the blockwise evaluation
/// or the elementwise evaluation is called
template<std::size_t N, class TensorX, class TensorV, class Device>
TensorX& minus_assign(tensor_expression<N, TensorX, Device>& x, tensor_expression<N, TensorV, Device> const& v){
	REMORA_SIZE_CHECK(x().shape() == v().shape());
	typedef typename TensorV::value_type value_type;
	auto minusV = detail::scalar_multiply_optimizer<TensorV>::create(v(),value_type(-1));
	detail::plus_assign(x,minusV,typename TensorV::evaluation_category());
	return x();
}

/// \brief Dispatches tensor multiply-assignment on an expression level
///
/// This dispatcher takes care for whether the blockwise evaluation
/// or the elementwise evaluation is called
template<std::size_t N, class TensorX, class TensorV, class Device>
TensorX& multiply_assign(tensor_expression<N, TensorX, Device>& x, tensor_expression<N, TensorV, Device> const& v){
	REMORA_SIZE_CHECK(x().shape() == v().shape());
	auto&& veval = eval_block(v);
	typedef typename device_traits<Device>:: template multiply<typename common_value_type<TensorX,TensorV>::type> F;
	kernels::assign(x, veval, F());
	return x();
}

/// \brief Dispatches tensor multiply-assignment on an expression level
///
/// This dispatcher takes care for whether the blockwise evaluation
/// or the elementwise evaluation is called
template<std::size_t N, class TensorX, class TensorV, class Device>
TensorX& divide_assign(tensor_expression<N, TensorX, Device>& x, tensor_expression<N, TensorV, Device> const& v){
	REMORA_SIZE_CHECK(x().shape() == v().shape());
	auto&& veval = eval_block(v);
	typedef typename device_traits<Device>:: template divide<typename common_value_type<TensorX,TensorV>::type> F;
	kernels::assign(x, veval, F());
	return x();
}

//////////////////////////////////////////////////////////////////////////////////////
///// Tensor Operators
/////////////////////////////////////////////////////////////////////////////////////

/// \brief  Add-Assigns two tensor expressions
///
/// Performs the operation x_i+=v_i for all elements.
/// Assumes that the right and left hand side aliases and therefore 
/// performs a copy of the right hand side before assigning
/// use noalias as in noalias(x)+=v to avoid this if A and B do not alias
template<std::size_t N, class TensorX, class TensorV, class Device>
TensorX& operator+=(tensor_expression<N, TensorX, Device>& x, tensor_expression<N, TensorV, Device> const& v){
	REMORA_SIZE_CHECK(x().shape() == v().shape());
	typename tensor_temporary<TensorX>::type temporary(v);
	return plus_assign(x,temporary);
}

template<std::size_t N, class TensorX, class TensorV, class Device>
typename TensorX::closure_type operator+=(tensor_expression<N, TensorX, Device>&& x, tensor_expression<N, TensorV, Device> const& v){
	REMORA_SIZE_CHECK(x().shape() == v().shape());
	typename tensor_temporary<TensorX>::type temporary(v);
	return plus_assign(x,temporary);
}

/// \brief  Subtract-Assigns two tensor expressions
///
/// Performs the operation x_i-=v_i for all elements.
/// Assumes that the right and left hand side aliases and therefore 
/// performs a copy of the right hand side before assigning
/// use noalias as in noalias(x)-=v to avoid this if A and B do not alias
template<std::size_t N, class TensorX, class TensorV, class Device>
TensorX& operator-=(tensor_expression<N, TensorX, Device>& x, tensor_expression<N, TensorV, Device> const& v){
	REMORA_SIZE_CHECK(x().shape() == v().shape());
	typename tensor_temporary<TensorX>::type temporary(v);
	return minus_assign(x,temporary);
}

template<std::size_t N, class TensorX, class TensorV, class Device>
typename TensorX::closure_type operator-=(tensor_expression<N, TensorX, Device>&& x, tensor_expression<N, TensorV, Device> const& v){
	REMORA_SIZE_CHECK(x().shape() == v().shape());
	typename tensor_temporary<TensorX>::type temporary(v);
	return minus_assign(x,temporary);
}

/// \brief  Multiply-Assigns two tensor expressions
///
/// Performs the operation x_i*=v_i for all elements.
/// Assumes that the right and left hand side aliases and therefore 
/// performs a copy of the right hand side before assigning
/// use noalias as in noalias(x)*=v to avoid this if A and B do not alias
template<std::size_t N, class TensorX, class TensorV, class Device>
TensorX& operator*=(tensor_expression<N, TensorX, Device>& x, tensor_expression<N, TensorV, Device> const& v){
	REMORA_SIZE_CHECK(x().shape() == v().shape());
	typename tensor_temporary<TensorX>::type temporary(v);
	return multiply_assign(x,temporary);
}

template<std::size_t N, class TensorX, class TensorV, class Device>
typename TensorX::closure_type operator*=(tensor_expression<N, TensorX, Device>&& x, tensor_expression<N, TensorV, Device> const& v){
	REMORA_SIZE_CHECK(x().shape() == v().shape());
	typename tensor_temporary<TensorX>::type temporary(v);
	multiply_assign(x,temporary);
}

/// \brief  Divide-Assigns two tensor expressions
///
/// Performs the operation x_i/=v_i for all elements.
/// Assumes that the right and left hand side aliases and therefore 
/// performs a copy of the right hand side before assigning
/// use noalias as in noalias(x)/=v to avoid this if A and B do not alias
template<std::size_t N, class TensorX, class TensorV, class Device>
TensorX& operator/=(tensor_expression<N, TensorX, Device>& x, tensor_expression<N, TensorV, Device> const& v){
	REMORA_SIZE_CHECK(x().shape() == v().shape());
	typename tensor_temporary<TensorX>::type temporary(v);
	return divide_assign(x,temporary);
}

template<std::size_t N, class TensorX, class TensorV, class Device>
typename TensorX::closure_type operator/=(tensor_expression<N, TensorX, Device>&& x, tensor_expression<N, TensorV, Device> const& v){
	REMORA_SIZE_CHECK(x().shape() == v().shape());
	typename tensor_temporary<TensorX>::type temporary(v);
	divide_assign(x,temporary);
}

/// \brief  Adds a scalar to all elements of the tensor
///
/// Performs the operation x_i += t for all elements.
template<std::size_t N, class TensorX, class T, class Device>
typename std::enable_if<std::is_convertible<T, typename TensorX::value_type>::value,TensorX&>::type
operator+=(tensor_expression<N, TensorX, Device>& x, T t){
	typedef typename TensorX::value_type value_type;
	typename device_traits<Device>:: template add_scalar<value_type> Functor;
	kernels::apply(x, Functor(value_type(t)));
	return x();
}

template<std::size_t N, class TensorX, class T, class Device>
typename std::enable_if<std::is_convertible<T, typename TensorX::value_type>::value,typename TensorX::closure_type>::type
operator+=(tensor_expression<N, TensorX, Device>&& x, T t){
	typedef typename TensorX::value_type value_type;
	typename device_traits<Device>:: template add_scalar<value_type> Functor;
	kernels::apply(x, Functor(value_type(t)));
	return x();
}

/// \brief  Subtracts a scalar from all elements of the tensor
///
/// Performs the operation x_i += t for all elements.
template<std::size_t N, class TensorX, class T, class Device>
typename std::enable_if<std::is_convertible<T, typename TensorX::value_type>::value,TensorX&>::type
operator-=(tensor_expression<N, TensorX, Device>& x, T t){
	typedef typename TensorX::value_type value_type;
	typename device_traits<Device>:: template add_scalar<value_type> Functor;
	kernels::apply(x, Functor(-value_type(t)));
	return x();
}

template<std::size_t N, class TensorX, class T, class Device>
typename std::enable_if<std::is_convertible<T, typename TensorX::value_type>::value,typename TensorX::closure_type>::type
operator-=(tensor_expression<N, TensorX, Device>&& x, T t){
	typedef typename TensorX::value_type value_type;
	typename device_traits<Device>:: template add_scalar<value_type> Functor;
	kernels::apply(x, Functor(-value_type(t)));
	return x();
}

/// \brief  Multiplies a scalar with all elements of the tensor
///
/// Performs the operation x_i *= t for all elements.
template<std::size_t N, class TensorX, class T, class Device>
typename std::enable_if<std::is_convertible<T, typename TensorX::value_type>::value,TensorX&>::type
operator*=(tensor_expression<N, TensorX, Device>& x, T t){
	typedef typename TensorX::value_type value_type;
	typename device_traits<Device>:: template multiply_scalar<value_type> Functor;
	kernels::apply(x, Functor(value_type(t)));
	return x();
}

template<std::size_t N, class TensorX, class T, class Device>
typename std::enable_if<std::is_convertible<T, typename TensorX::value_type>::value,typename TensorX::closure_type>::type
operator*=(tensor_expression<N, TensorX, Device>&& x, T t){
	typedef typename TensorX::value_type value_type;
	typename device_traits<Device>:: template multiply_scalar<value_type> Functor;
	kernels::apply(x, Functor(value_type(t)));
	return x();
}

/// \brief  Divides all elements of the tensor by a scalar
///
/// Performs the operation x_i /= t for all elements.
template<std::size_t N, class TensorX, class T, class Device>
typename std::enable_if<std::is_convertible<T, typename TensorX::value_type>::value,TensorX&>::type
operator/=(tensor_expression<N, TensorX, Device>& x, T t){
	typedef typename TensorX::value_type value_type;
	typename device_traits<Device>:: template divide_scalar<value_type> Functor;
	kernels::apply(x, Functor(value_type(t)));
	return x();
}

template<std::size_t N, class TensorX, class T, class Device>
typename std::enable_if<std::is_convertible<T, typename TensorX::value_type>::value,typename TensorX::closure_type>::type
operator/=(tensor_expression<N, TensorX, Device>&& x, T t){
	typedef typename TensorX::value_type value_type;
	typename device_traits<Device>:: template divide_scalar<value_type> Functor;
	kernels::apply(x, Functor(value_type(t)));
	return x();
}

// Assignment proxy.
// Provides temporary free assigment when LHS has no alias on RHS
template<class C>
class noalias_proxy{
public:
	typedef typename C::closure_type closure_type;
	typedef typename C::value_type value_type;

	noalias_proxy(C &lval): m_lval(lval) {}

	noalias_proxy(const noalias_proxy &p):m_lval(p.m_lval) {}

	template <class E>
	closure_type &operator= (const E &e) {
		return assign(m_lval, e);
	}

	template <class E>
	closure_type &operator+= (const E &e) {
		return plus_assign(m_lval, e);
	}

	template <class E>
	closure_type &operator-= (const E &e) {
		return minus_assign(m_lval, e);
	}
	
	template <class E>
	closure_type &operator*= (const E &e) {
		return multiply_assign(m_lval, e);
	}

	template <class E>
	closure_type &operator/= (const E &e) {
		return divide_assign(m_lval, e);
	}
	
	//this is not needed, but prevents errors when for example doing noalias(x)+=2;
	closure_type &operator+= (value_type t) {
		return m_lval += t;
	}

	//this is not needed, but prevents errors when for example doing noalias(x)-=2;
	closure_type &operator-= (value_type t) {
		return m_lval -= t;
	}
	
	//this is not needed, but prevents errors when for example doing noalias(x)*=2;
	closure_type &operator*= (value_type t) {
		return m_lval *= t;
	}

	//this is not needed, but prevents errors when for example doing noalias(x)/=2;
	closure_type &operator/= (value_type t) {
		return m_lval /= t;
	}

private:
	closure_type m_lval;
};

// Improve syntax of efficient assignment where no aliases of LHS appear on the RHS
//  noalias(lhs) = rhs_expression
template <std::size_t N, class C, class Device>
noalias_proxy<C> noalias(tensor_expression<N, C, Device>& lvalue) {
	return noalias_proxy<C> (lvalue());
}


template <std::size_t N, class C, class Device>
noalias_proxy<C> noalias(tensor_expression<N, C, Device>&& lvalue) {
	return noalias_proxy<C> (lvalue());
}



}
#endif