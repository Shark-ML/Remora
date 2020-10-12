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
#include "../kernels/reduce.hpp"
#include "../kernels/gemm.hpp"
#include "../kernels/gemv.hpp"
#include "../kernels/device_traits.hpp"
#include "../assignment.hpp"
#include <type_traits>
#include <tuple>

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
	scalar_multiply(expression_closure_type const& e, value_type scalar)
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
	
	// conversion operator for scalar case
	template<std::size_t D = num_dims, typename = typename std::enable_if< D == 0 >::type>
	operator value_type()const{
		typename tensor_temporary<scalar_multiply>::type temp = *this;
		return temp;
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
    tensor_addition(
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
	
	// conversion operator for scalar case
	template<std::size_t D = num_dims, typename = typename std::enable_if< D == 0 >::type>
	operator value_type()const{
		typename tensor_temporary<tensor_addition>::type temp = *this;
		return temp;
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
	
	// conversion operator for scalar case
	template<std::size_t D = num_dims, typename = typename std::enable_if< D == 0 >::type>
	operator value_type()const{
		return m_value;
	}
private:
	tensor_shape<num_dims> m_shape;
	value_type m_value;
};

/// \brief A Tensor that broadcasts a Tensor to a different shape
///
/// Broadcasting is the operation of adding one or more additional axis to a tensor and copying
/// the vector multiple times along those axes. 
///
/// Example:
///  
/// if we have a 3D tensor t_ijk that is broadcasted
/// to a tensur u_mijnk, broadcasting represents u as u_mijnk = t_ijk 
///
/// \tparam E the expression to broadcast
/// \tparam Axis the axis object of the tensor after broadcasting
/// \tparam BroadcastList boolean list indicating whether axis i (in Axis) is a broadcasted dimension.
template<class E, class Axis, class BroadcastList>
class tensor_broadcast:public tensor_expression<Axis::num_dims, tensor_broadcast<E, Axis, BroadcastList>, typename E::device_type >{
private:
	//transform the drop list to a list of indices to keep
	static constexpr std::size_t num_keep(){
		auto drop_list = BroadcastList::to_array();
		std::size_t count = 0;
		for(std::size_t i = 0; i != BroadcastList::num_dims; ++i){
			count += !drop_list[i];
		}
		return count;
	}
	
	//check invariants
	static_assert(E::num_dims == num_keep(), "number of zeros in BroadcastList must be equal to dimension of E");
	static_assert(Axis::num_dims == BroadcastList::num_dims, "BroadcastList must have same length as Axis");
	
	struct keep_list_helper{
		template<class Seq>
		static constexpr auto apply(Seq){
			auto drop_list = BroadcastList::to_array();
			typename Axis::array_type result={0};
			std::size_t pos = 0;
			for(std::size_t i = 0; i != Axis::num_dims; ++i){
				if(drop_list[i] == 0){
					result[pos] = i;
					++pos;
				}
			}
			return result;
		}
	};
public:
	//we abuse transform_t of Axis to transform the array into an integer_list and only take the front elements
	typedef typename Axis::template transform_t<keep_list_helper>::template front_t<num_keep()> keep_list;
	
	
	//todo: move to device_traits once this works
	//also figure out how to do this in cuda
	template<class F>
	struct broadcast_functor{
		typedef typename F::result_type result_type;
		template<std::size_t... Ns, class ArgTuple>
		result_type apply(std::index_sequence<Ns...>, ArgTuple const& args){
			auto constexpr keep_array = keep_list::to_array();
			(void) keep_array;//prevent warning for unused array
			return f(std::get<keep_array[Ns]>(args)...);
		}
		template<class... Args>
		result_type operator()(Args&&... args){
			return apply(std::make_index_sequence<keep_list::num_dims>(),std::make_tuple(args...));
		}
		F f;
	};

	typedef typename E::const_closure_type expression_closure_type;
	typedef typename E::size_type size_type;
	typedef typename E::value_type value_type;
	typedef typename E::value_type const_reference;
	typedef const_reference reference;

	typedef tensor_broadcast const_closure_type;
	typedef tensor_broadcast closure_type;
	typedef unknown_storage storage_type;
	typedef unknown_storage const_storage_type;
	typedef Axis  axis;
	typedef typename E::evaluation_category evaluation_category;
	typedef typename E::device_type device_type;
	static constexpr std::size_t num_dims = Axis::num_dims;

	// Construction
	tensor_broadcast(expression_closure_type const& e, tensor_shape<num_dims> const& shape)
	:m_expression(e), m_shape(shape){}

	//Accessors 
	auto const& shape()const{
		return m_shape;
	}
	expression_closure_type const& expression() const{
		return m_expression;
	};
	typename device_traits<device_type>::queue_type& queue()const{
		return m_expression.queue();
	}

	// Element Functor
	auto elements() const{
		auto f = m_expression.elements();
		return broadcast_functor<decltype(f)>{f};
	}

	// Computation Kernels
	template<class TensorX>
	void assign_to(tensor_expression<num_dims, TensorX, device_type>& X)const{
		auto eval_e = eval_block(m_expression);
		tensor_broadcast<decltype(eval_e), axis, BroadcastList> broadcast_eval_e(eval_e, m_shape);
		assign(X, broadcast_eval_e);
	}
	template<class TensorX>
	void plus_assign_to(tensor_expression<num_dims, TensorX, device_type>& X)const{
		auto eval_e = eval_block(m_expression);
		tensor_broadcast<decltype(eval_e), axis, BroadcastList> broadcast_eval_e(eval_e, m_shape);
		plus_assign(X, broadcast_eval_e);
	}

private:
	expression_closure_type m_expression;
	tensor_shape<num_dims> m_shape;
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
	tensor_unary(expression_closure_type const& e, functor_type const& functor):
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
	
	// conversion operator for scalar case
	template<std::size_t D = num_dims, typename = typename std::enable_if< D == 0 >::type>
	operator value_type()const{
		typename tensor_temporary<tensor_unary>::type temp = *this;
		return temp;
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
    tensor_binary (
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
	
	// conversion operator for scalar case
	template<std::size_t D = num_dims, typename = typename std::enable_if< D == 0 >::type>
	operator value_type()const{
		typename tensor_temporary<tensor_binary>::type temp = *this;
		return temp();
	}

private:
	lhs_closure_type m_lhs;
    rhs_closure_type m_rhs;
	functor_type m_functor;
};


template<class E, class F>
class tensor_reduce_last:public tensor_expression<E::num_dims - 1, tensor_reduce_last<E, F>, typename E::device_type >{
public:
	typedef typename E::const_closure_type expression_closure_type;
public:
	typedef typename F::result_type value_type;
	typedef typename E::size_type size_type;
	typedef value_type const_reference;
	typedef const_reference reference;

	typedef tensor_reduce_last const_closure_type;
	typedef const_closure_type closure_type;
	typedef unknown_storage storage_type;
	typedef unknown_storage const_storage_type;
	typedef typename E::device_type device_type;
	typedef blockwise<typename E::evaluation_category::tag> evaluation_category;
	static constexpr std::size_t num_dims = E::num_dims - 1;
	typedef typename E::axis::template slice_t<num_dims> axis;
	// Construction
	tensor_reduce_last(
		expression_closure_type const& expression, F const& f
	):m_expression(expression), m_functor(f){}

	// Accessors 
	tensor_shape<num_dims> const shape()const{
		return m_expression.shape().slice(num_dims);
	}
	expression_closure_type const& expression() const{
		return m_expression;
	}
	F const& functor() const{
		return m_functor;
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
		auto E_eval = eval_block(m_expression);//TODO: check/ensure that this uses the same layout as X
		kernels::reduce_last(E_eval, X, m_functor);
	}
	
	// Iterator Access 
	// typedef no_iterator const_iterator;
	// typedef no_iterator iterator;
	
	// conversion operator for scalar case
	template<std::size_t D = num_dims, typename = typename std::enable_if< D == 0 >::type>
	operator value_type()const{
		typename tensor_temporary<tensor_reduce_last>::type temp = *this;
		return temp;
	}
private:
	expression_closure_type m_expression;
	F m_functor;
};

template<class TensorA, class TensorB, class Permutation>
class tensor_prod_reduce;

template<class TensorA, class TensorB, unsigned... Permutation>
class tensor_prod_reduce<TensorA, TensorB, axis<Permutation...> >
: public tensor_expression<
	TensorA::num_dims + TensorB::num_dims - 2, 
	tensor_prod_reduce<TensorA, TensorB, axis<Permutation...> >, 
	typename TensorA::device_type
>{
private:
	template<unsigned... Seq0, unsigned... Seq1>
	static remora::axis<Seq0..., (Seq1+sizeof...(Seq0))...> concat_axes(remora::axis<Seq0...>, remora::axis<Seq1...>);
public:
	static_assert(sizeof...(Permutation) == TensorA::num_dims + TensorB::num_dims - 2);
	typedef typename TensorA::const_closure_type lhs_closure_type;
	typedef typename TensorB::const_closure_type rhs_closure_type;
	
	typedef typename TensorA::size_type size_type;
	typedef decltype(
		typename TensorA::value_type() * typename TensorB::value_type()
	) value_type;
	typedef value_type const& const_reference;
	typedef const_reference reference;

	typedef tensor_prod_reduce const_closure_type;
	typedef const_closure_type closure_type;
	typedef unknown_storage storage_type;
	typedef unknown_storage const_storage_type;
	typedef blockwise<typename evaluation_tag_restrict_traits<
		typename TensorA::evaluation_category::tag,
		typename TensorB::evaluation_category::tag
	>::type> evaluation_category;
	typedef typename TensorA::device_type device_type;
	static constexpr std::size_t num_dims = TensorA::num_dims + TensorB::num_dims - 2;
	//axis is created by slicing off the reduced axes from both tensors
	//concatenating them together and than applying the permutation on the result.
	typedef typename decltype(concat_axes(
		typename TensorA::axis::template slice_t<TensorA::num_dims - 1>(),
		typename TensorB::axis::template slice_t<0>()
	))::template permute_t<Permutation...> axis;

	// Construction
	tensor_prod_reduce(
		lhs_closure_type const& lhs,
		rhs_closure_type const& rhs,
		value_type scalar
	):m_lhs(lhs), m_rhs(rhs), m_scalar(scalar){}

	// Accessors 
	tensor_shape<num_dims> shape() const{
		auto shape_left = m_lhs.shape();
		auto shape_right = m_rhs.shape();
		tensor_shape<num_dims> result;
		std::size_t dim_left = TensorA::num_dims - 1;
		auto map = remora::axis<Permutation...>::inverse_t::to_array();
		for(std::size_t i = 0; i != dim_left; ++i){
			result[map[i]] = shape_left[i];
		}
		for(std::size_t i = 1; i != TensorB::num_dims; ++i){
			result[map[dim_left + i - 1]] = shape_right[i];
		}
		return result;
	}

	lhs_closure_type const& lhs() const{
		return m_lhs;
	}
	rhs_closure_type const& rhs() const{
		return m_rhs;
	}
	value_type scalar() const{
		return m_scalar;
	}
	
	typename device_traits<device_type>::queue_type& queue()const{
		return m_lhs.queue();
	}
	
	//Element Functor
	no_functor elements() const{return no_functor();}
	
	// Computation Kernels
	template<class TensorX>
	void assign_to(tensor_expression<num_dims, TensorX, device_type>& X)const{
		X().clear();
		//apply the inverse permutation on X instead of the result.
		auto X_permuted = permute(X, typename remora::axis<Permutation...>::inverse_t());
		//reshape X to matrix-form
		auto X_mat = reshape(X_permuted,ax::merge<TensorA::num_dims - 1>(),ax::merge<TensorB::num_dims - 1>());
		//reshape arguments into vector- or matrix-form
		auto lhs_mat = reshape(m_lhs, ax::merge<TensorA::num_dims - 1>(), ax::same);
		auto rhs_mat = reshape(m_rhs, ax::same, ax::merge<TensorB::num_dims - 1>());
		//call the kernel-dispatcher
		call_matmul(X_mat, lhs_mat, rhs_mat);
	}
	template<class TensorX>
	void plus_assign_to(tensor_expression<num_dims, TensorX, device_type>& X)const{
		//apply the inverse permutation on X instead of the result.
		auto X_permuted = permute(X, typename remora::axis<Permutation...>::inverse_t());
		//reshape X to matrix-form
		auto X_mat = reshape(X_permuted,ax::merge<TensorA::num_dims - 1>(),ax::merge<TensorB::num_dims - 1>());
		//reshape arguments into vector- or matrix-form
		auto lhs_mat = reshape(m_lhs, ax::merge<TensorA::num_dims - 1>(), ax::same);
		auto rhs_mat = reshape(m_rhs, ax::same, ax::merge<TensorB::num_dims - 1>());
		//call the kernel-dispatcher
		call_matmul(X_mat, lhs_mat, rhs_mat);
	}
	
private:
	//all the supported kernel operations

	//gemm
	template<class TensorX, class TensorL, class TensorR>
	void call_matmul(
		tensor_expression<2, TensorX, device_type>& X,
		tensor_expression<2, TensorL, device_type>& lhs,
		tensor_expression<2, TensorR, device_type>& rhs
	)const{
		kernels::gemm(eval_block(lhs), eval_block(rhs), X(), m_scalar);
	}
	//gemv, two versions base don position of vector.
	template<class TensorX, class TensorL, class TensorR>
	void call_matmul(
		tensor_expression<1, TensorX, device_type>& X,
		tensor_expression<2, TensorL, device_type>& lhs,
		tensor_expression<1, TensorR, device_type>& rhs
	)const{
		kernels::gemv(eval_block(lhs), eval_block(rhs), X(), m_scalar);
	}
	template<class TensorX, class TensorL, class TensorR>
	void call_matmul(
		tensor_expression<1, TensorX, device_type>& X,
		tensor_expression<1, TensorL, device_type>& lhs,
		tensor_expression<2, TensorR, device_type>& rhs
	)const{
		kernels::gemv(eval_block(trans(rhs)), eval_block(lhs), X(), m_scalar);
	}
	
/*	//trmv
	template<class TensorX>
	void assign_to(matrix_expression<TensorX, device_type>& X, triangular_structure, dense_tag)const{
		//assign the rhs and multiply in-place
		assign(X, m_rhs);
		kernels::trmm<TensorA::orientation::is_upper, TensorA::orientation::is_unit>(m_lhs.to_dense(), X);
		
		//perform multiplication with alpha if necessary
		if(m_scalar != value_type(1)){
			typedef typename device_traits<device_type>:: template multiply<value_type> Multiply;
			kernels::assign<Multiply>(X,m_scalar);
		}
	}
	template<class TensorX>
	void plus_assign_to(matrix_expression<TensorX, device_type>& X, triangular_structure, dense_tag )const{
		//computation of trmm is in-place so we need a temporary for plus-assign.
		typename matrix_temporary<TensorX>::type temp = m_rhs;
		kernels::trmm<TensorA::orientation::is_upper, TensorA::orientation::is_unit>(m_lhs.to_dense(), temp);
		
		//perform plus-assignment of temporary
		typename device_traits<device_type>:: template multiply_and_add<value_type> multiply(m_scalar);
		kernels::assign(X, temp, multiply);
	}*/
private:
	lhs_closure_type m_lhs;
	rhs_closure_type m_rhs;
	value_type m_scalar;
};



}
#endif
