/*!
 * \brief       Classes used for tensor expressions.
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
#ifndef REMORA_TENSOR_EXPRESSION_CLASSES_HPP
#define REMORA_TENSOR_EXPRESSION_CLASSES_HPP

#include "traits.hpp"
// #include "../kernels/fold_rows.hpp"
#include "../kernels/device_traits.hpp"
#include "../assignment.hpp"
#include <type_traits>

namespace remora{

template<class E>
class scalar_multiply:public tensor_expression<E::num_dims, scalar_multiply<E>, typename E::device_type >{
public:
	typedef typename device_traits<typename E::device_type>::template multiply_scalar<typename E::value_type> functor_type;
	typedef typename E::const_closure_type expression_closure_type;
	typedef typename E::size_type size_type;
	typedef typename E::value_type value_type;
	typedef value_type const_reference;
	typedef const_reference reference;

	typedef scalar_multiply const_closure_type;
	typedef const_closure_type closure_type;
	typedef unknown_storage storage_type;
	typedef unknown_storage const_storage_type;
	typedef typename E::axis axis;
	typedef typename E::evaluation_category evaluation_category;
	typedef typename E::device_type device_type;
	static constexpr std::size_t num_dims = E::num_dims;

	// Construction
	explicit scalar_multiply(expression_closure_type const& e, value_type scalar)
	:m_expression(e), m_scalar(scalar){}

	//Accessors 
	tensor_shape<num_dims> const shape()const{
		return m_expression.shape();
	}
	value_type scalar()const{
		return m_scalar;
	}
	expression_closure_type const& expression() const{
		return m_expression;
	};
	typename device_traits<device_type>::queue_type& queue()const{
		return m_expression.queue();
	}

	// Element Functor
	auto elements() const{
		return device_traits<device_type>::make_compose(
			m_expression.elements(), functor_type(m_scalar)
		);
	}

	// Computation Kernels
	template<class TensorX>
	void assign_to(tensor_expression<num_dims, TensorX, device_type>& X)const{
		auto eval_e = eval_block(m_expression);
		typename device_traits<device_type>::template multiply_assign<value_type> f(m_scalar);
		kernels::assign(X, eval_e, f);
	}
	template<class TensorX>
	void plus_assign_to(tensor_expression<num_dims, TensorX, device_type>& X)const{
		auto eval_e = eval_block(m_expression);
		typename device_traits<device_type>::template multiply_and_add<value_type> f(m_scalar);
		kernels::assign(X, eval_e, f);
	}

private:
	expression_closure_type m_expression;
	value_type m_scalar;
};	
	
template<class E1, class E2>
class tensor_addition: public tensor_expression<E1::num_dims, tensor_addition<E1, E2>, typename E1::device_type >{
private:
	typedef typename device_traits<typename E1::device_type>:: template add<typename E1::value_type> functor_type;
	typedef typename E1::const_closure_type lhs_closure_type;
	typedef typename E2::const_closure_type rhs_closure_type;
public:

	typedef typename E1::size_type size_type;
	typedef decltype(typename E1::value_type() + typename E2::value_type()) value_type;
	typedef value_type const_reference;
	typedef const_reference reference;

	typedef tensor_addition const_closure_type;
	typedef const_closure_type closure_type;
	typedef unknown_storage storage_type;
	typedef unknown_storage const_storage_type;
	typedef typename std::conditional< //the axis is only known if E1 and E2 have the same
		std::is_same<typename E1::axis, typename E2::axis>::value,
		typename E1::axis,
		unknown_axis<E1::num_dims>
	>::type axis;
	//the evaluation category is blockwise if one of the expressions is blockwise or
	// if the axis is unknown (efficient for expressions like A=B+C^T
	typedef typename std::conditional<
		std::is_same<axis, unknown_axis<E1::num_dims> >::value,
		blockwise<typename evaluation_restrict_traits<E1,E2>::type::tag>,
		typename evaluation_restrict_traits<E1,E2>::type
	>::type evaluation_category;
	typedef typename E1::device_type device_type;
	static constexpr std::size_t num_dims = E1::num_dims;

    // Construction
    explicit tensor_addition(
		lhs_closure_type const& e1,
		rhs_closure_type const& e2
	): m_lhs (e1), m_rhs (e2){}
	
	//Accessors
	tensor_shape<num_dims> const shape()const{
		return m_lhs.shape();
	}
	lhs_closure_type const& lhs()const{
		return m_lhs;
	}
	rhs_closure_type const& rhs()const{
		return m_rhs;
	}
	typename device_traits<device_type>::queue_type& queue()const{
		return m_lhs.queue();
	}
	
	// Element Functor
	auto elements() const{
		return device_traits<device_type>::make_compose_binary(
			m_lhs.elements(), m_rhs.elements(), functor_type()
		);
	}

	//Computation Kernels
	template<class TensorX>
	void assign_to(tensor_expression<num_dims, TensorX, device_type>& X)const{
		//working around a bug in non-dense assign
		if(!std::is_base_of<dense_tag, typename E1::evaluation_category::tag>::value){
			X().clear();
			plus_assign(X, m_lhs);
		}else{
			assign(X, m_lhs);
		}
		plus_assign(X, m_rhs);
	}
	template<class TensorX>
	void plus_assign_to(tensor_expression<num_dims, TensorX, device_type>& X)const{
		plus_assign(X,m_lhs);
		plus_assign(X,m_rhs);
	}

private:
	lhs_closure_type m_lhs;
    rhs_closure_type m_rhs;
	functor_type m_functor;
};

/// \brief A tensor with all values of type \c T equal to the same value
///
/// \tparam T the type of object stored in the tensor (like double, float, complex, etc...)
/// \tparam Axis the prefered order of the scalar object
/// \tparam Device the device the tensor is located on
template<class T, class Axis, class Device>
class scalar_tensor:public tensor_expression<Axis::num_dims, scalar_tensor<T, Axis, Device>, Device >{
public:
	typedef std::size_t size_type;
	typedef T value_type;
	typedef const T& const_reference;
	typedef const_reference reference;

	typedef scalar_tensor const_closure_type;
	typedef scalar_tensor closure_type;
	typedef unknown_storage storage_type;
	typedef unknown_storage const_storage_type;
	typedef Axis axis;
	typedef elementwise<dense_tag> evaluation_category;
	static constexpr std::size_t num_dims = axis::num_dims;
	
	// Construction
	scalar_tensor(tensor_shape<num_dims> const& shape, const value_type& value):
		m_shape(shape), m_value(value){}
	
	//Accessors
	tensor_shape<num_dims> const& shape()const{
		return m_shape;
	}
	T scalar() const{
		return m_value;
	}
	typename device_traits<Device>::queue_type& queue()const{
		return device_traits<Device>::default_queue();
	}
	
	// Element Functor
	typename device_traits<Device>:: template constant<value_type> elements() const{
		return {m_value};
	}
private:
	tensor_shape<num_dims> m_shape;
	value_type m_value;
};

///\brief class which allows for tensor transformations
///
///transforms a tensor expression e of type E using a Function f of type F as an elementwise transformation f(e(i,j,...))
///This transformation needs to leave f constant, meaning that applying f(x), f(y), f(z) yields the same results independent of the
///order of application.
///F must provide a boolean flag F::zero_identity which indicates that f(0) = 0. This is needed for correct usage with sparse
///arguments - if f(0) != 0 this expression will be dense!
template<class E, class F>
class tensor_unary:public tensor_expression<E::num_dims, tensor_unary<E, F>, typename E::device_type >{
public:
	typedef typename E::const_closure_type expression_closure_type;

	typedef F functor_type;
	typedef typename F::result_type value_type;
	typedef value_type const_reference;
	typedef const_reference reference;
	typedef typename E::size_type size_type;

	typedef tensor_unary const_closure_type;
	typedef const_closure_type closure_type;
	typedef unknown_storage storage_type;
	typedef unknown_storage const_storage_type;
	typedef typename E::axis axis;
	typedef typename E::evaluation_category evaluation_category;
	typedef typename E::device_type device_type;
	static constexpr std::size_t num_dims = E::num_dims;

	// Construction
	explicit tensor_unary(expression_closure_type const& e, functor_type const& functor):
		m_expression(e), m_functor(functor){}
		
	// Accessors
	tensor_shape<num_dims> const shape()const{
		return m_expression.shape();
	}
	expression_closure_type const& expression() const{
		return m_expression;
	}
	functor_type const& functor() const{
		return m_functor;
	}
	typename device_traits<device_type>::queue_type& queue()const{
		return m_expression.queue();
	}
	
	// Element Functor
	auto elements() const{
		return device_traits<device_type>::make_compose(m_expression.elements(), m_functor);
	}

	//Computation Kernels
	template<class TensorX>
	void assign_to(tensor_expression<num_dims, TensorX, device_type>& X) const{
		assign(X,m_expression);
		kernels::apply(X,m_functor);
	}
	template<class TensorX>
	void plus_assign_to(tensor_expression<num_dims, TensorX, device_type>& X) const{
		
		//assign expects functions b=f(b,a)
		//for b=b+g(a)
		//we implement b=b+g(a) = add(identity(b),g(a))
		auto eval_rhs = eval_block(m_expression);
		typename device_traits<device_type>:: template identity<value_type> identity;
		typename device_traits<device_type>:: template add<value_type> add;
		kernels::assign(X,eval_rhs, device_traits<device_type>::make_transform_arguments(identity,m_functor,add));
	}

private:
	expression_closure_type m_expression;
	functor_type m_functor;
};

template<class E1, class E2, class F>
class tensor_binary:public tensor_expression<E1::num_dims, tensor_binary<E1, E2, F>, typename E1::device_type >{
public:
	typedef typename E1::const_closure_type lhs_closure_type;
	typedef typename E2::const_closure_type rhs_closure_type;

	typedef typename E1::size_type size_type;
	typedef typename F::result_type value_type;
	typedef value_type const_reference;
	typedef const_reference reference;

	typedef tensor_binary const_closure_type;
	typedef const_closure_type closure_type;
	typedef unknown_storage storage_type;
	typedef unknown_storage const_storage_type;
	typedef typename std::conditional< //the axis is only known if E1 and E2 have the same
		std::is_same<typename E1::axis, typename E2::axis>::value,
		typename E1::axis,
		unknown_axis<E1::num_dims>
	>::type axis;
	typedef typename std::conditional<
		std::is_same<axis, unknown_axis<E1::num_dims> >::value,
		blockwise<typename evaluation_restrict_traits<E1,E2>::type::tag>,
		typename evaluation_restrict_traits<E1,E2>::type
	>::type evaluation_category;
	typedef typename E1::device_type device_type;
	static constexpr std::size_t num_dims = E1::num_dims;

	typedef F functor_type;
    explicit tensor_binary (
		lhs_closure_type const& e1,  rhs_closure_type const& e2, functor_type functor 
	): m_lhs(e1), m_rhs(e2),m_functor(functor){}
	
	//Accessors
	tensor_shape<num_dims> const shape()const{
		return m_lhs.shape();
	}
	lhs_closure_type const& lhs() const{
		return m_lhs;
	}
	rhs_closure_type const& rhs() const{
		return m_rhs;
	}
	functor_type const& functor() const{
		return m_functor;
	}
	typename device_traits<device_type>::queue_type& queue()const{
		return m_lhs.queue();
	}

	// Element Functor
	auto elements() const{
		return device_traits<device_type>::make_compose_binary(
			m_lhs.elements(), m_rhs.elements(), m_functor
		);
	}
	
	//Computation Kernels
	template<class TensorX>
	void assign_to(tensor_expression<num_dims, TensorX, device_type>& X)const{
		assign(X,m_lhs);
		kernels::assign(X,eval_block(m_rhs),m_functor);
	}
	template<class TensorX>
	void plus_assign_to(tensor_expression<num_dims, TensorX, device_type>& X)const{
		auto eval_lhs = eval_block(m_lhs);
		auto eval_rhs = eval_block(m_rhs);
		tensor_binary<decltype(eval_lhs),decltype(eval_rhs),F> e(eval_lhs,eval_rhs, m_functor);
		plus_assign(X,e);		
	}

private:
	lhs_closure_type m_lhs;
    rhs_closure_type m_rhs;
	functor_type m_functor;
};

/*
template<class E, class F, class G, class Axis>
class tensor_row_transform:public tensor_expression<E::num_dims - Axis::num_dims, tensor_row_transform<E, F, G, Axis>, typename E::device_type >{
public:
	typedef typename E::const_closure_type tensor_closure_type;
public:
	typedef typename G::result_type value_type;
	typedef typename E::size_type size_type;
	typedef value_type const_reference;
	typedef const_reference reference;

	typedef tensor_row_transform const_closure_type;
	typedef const_closure_type closure_type;
	typedef unknown_storage storage_type;
	typedef unknown_storage const_storage_type;
	typedef typename E::device_type device_type;
	typedef blockwise<typename E::evaluation_category::tag> evaluation_category;
	static constexpr std::size_t num_dims = E::num_dims - Axis::num_dims;
	// Construction
	explicit tensor_row_transform(
		tensor_closure_type const& tensor, F const& f, G const& g
	):m_expression(tensor), m_f(f), m_g(g){}

	// Accessors 
	tensor_shape<num_dims> const shape()const{
		return {m_expression.shape()[0]};
	}
	tensor_closure_type const& tensor() const{
		return m_expression;
	}
	F const& f() const{
		return m_f;
	}
	G const& g() const{
		return m_g;
	}
	typename device_traits<device_type>::queue_type& queue()const{
		return m_expression.queue();
	}
	
	//Element Functor
	no_functor elements() const{return no_functor();}
	
	// Computation Kernels
	template<class TensorX>
	void assign_to(tensor_expression<num_dims, TensorX, device_type>& X)const{
		X().clear();
		plus_assign_to(X);
	}
	template<class TensorX>
	void plus_assign_to(tensor_expression<num_dims, TensorX, device_type>& X)const{
		kernels::fold_rows(eval_block(m_expression), X, m_f, m_g);
	}
	
	// Iterator Access 
	// typedef no_iterator const_iterator;
	// typedef no_iterator iterator;
private:
	tensor_closure_type m_expression;
	F m_f;
	G m_g;
};
*/
}
#endif
