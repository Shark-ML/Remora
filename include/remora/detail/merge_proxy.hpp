/*!
 * \brief       Class used to implement merging of axes which can not be represented via closure-proxies
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
#ifndef REMORA_TENSOR_MERGE_PROXY_HPP
#define REMORA_TENSOR_MERGE_PROXY_HPP

#include <type_traits>

#include "traits.hpp"
#include "../assignment.hpp"
#include "proxy_optimizers_fwd.hpp"


namespace remora{

template<class TensorE, std::size_t N>
class merge_proxy:public tensor_expression<TensorE::num_dims - 1, merge_proxy<TensorE, N>, typename TensorE::device_type >{
public:
	typedef typename closure<TensorE>::type expression_closure_type;
	typedef typename TensorE::size_type size_type;
	typedef typename TensorE::value_type value_type;
	typedef typename reference<TensorE>::type reference;
	typedef typename TensorE::const_reference const_reference;

	typedef merge_proxy<typename TensorE::const_closure_type, N> const_closure_type;
	typedef merge_proxy closure_type;
	typedef unknown_storage storage_type;
	typedef unknown_storage const_storage_type;
	typedef typename TensorE::axis::template slice_t<N> axis;
	typedef blockwise<typename TensorE::evaluation_category::tag> evaluation_category;
	typedef typename TensorE::device_type device_type;
	static constexpr std::size_t num_dims = axis::num_dims;

	// Construction
	merge_proxy(expression_closure_type const& E):m_expression(E){}
	
	// assignment 
	merge_proxy& operator = (merge_proxy const& rhs){
		REMORA_SIZE_CHECK(shape() == rhs.shape());
		auto merge_rhs = detail::axis_merge_optimizer<merge_proxy, N>::create(rhs);
		return assign(m_expression, merge_rhs);
	}
	template<class OE>
	merge_proxy& operator = (tensor_expression<num_dims, OE, cpu_tag> const& rhs){
		REMORA_SIZE_CHECK(shape() == rhs().shape());
		auto merge_rhs = detail::axis_merge_optimizer<OE, N>::create(rhs());
		return assign(m_expression, merge_rhs);
	}

	//Accessors 
	tensor_shape<num_dims> const shape()const{
		return m_expression.shape().merge(N);
	}
	expression_closure_type const& expression() const{
		return m_expression;
	};
	typename device_traits<device_type>::queue_type& queue()const{
		return m_expression.queue();
	}

	// Element Functor is unimplemented.
	no_functor elements() const;

	// Computation Kernels are implemented by splitting an axis of the target insteaf of trying to merge axis in E.
	template<class TensorX>
	void assign_to(tensor_expression<num_dims, TensorX, device_type>& X)const{
		auto shape = m_expression.shape();
		auto Xsplit = detail::axis_split_optimizer<typename closure<TensorX>::type, N>::create(X(), shape[N], shape[N+1]);
		assign(Xsplit, m_expression);
	}
	template<class TensorX>
	void plus_assign_to(tensor_expression<num_dims, TensorX, device_type>& X)const{
		auto shape = m_expression.shape();
		auto Xsplit = detail::axis_split_optimizer<typename closure<TensorX>::type, N>::create(X(), shape[N], shape[N+1]);
		plus_assign(Xsplit, m_expression);
	}

private:
	expression_closure_type m_expression;
	value_type m_scalar;
};


///////Implementation of necessary proxy optimizers///////
////////////////////////TENSOR SUBRANGE NOT IMPLEMENTED!//////////////////////
////////////////////////TENSOR SLICE//////////////////////
namespace detail{
template<class TensorE, std::size_t NE, std::size_t N>
struct slice_optimizer<merge_proxy<TensorE, NE>, N>{
	static constexpr std::size_t is_merged = (N == NE);
	
	//case 1: is_merged = false
	//implement by first slicing E and afterwards creating a new merge_proxy object
	typedef slice_optimizer<TensorE, (N > NE? N + 1 : N)> opt_merge_slice;
	typedef merge_proxy<typename opt_merge_slice::type, (N > NE? NE: NE - 1) > not_merged_type;
	
	static not_merged_type create(TensorE const& E, std::size_t i, integer_list<std::size_t, 0>){
		return opt_merge_slice::create(E,i);
	}
	
	//case 2: is_merged = true
	//implement this case simply by calculating the correct index-pair in E and slicing that
	typedef slice_optimizer<TensorE, NE + 1> opt_slice1;
	typedef slice_optimizer<typename opt_slice1::type, NE> opt_slice2;
	typedef typename opt_slice2::type merged_type;
	
	static merged_type create(TensorE const& E, std::size_t index, integer_list<std::size_t, 1>){
		//compute the split index and slice E using those values.
		auto size_j = E.shape()[NE + 1];
		std::size_t i = index / size_j;
		std::size_t j = index % size_j;
		return opt_slice2::create(opt_slice1::create(E,j), i);
	}
	
	//dispatcher
	typedef typename std::conditional<is_merged, merged_type, not_merged_type>::type type;
	static type create(
		merge_proxy<TensorE, NE> const& E, std::size_t index
	){
		REMORA_SIZE_CHECK(index < E.shape()[N]);
		return create(E.expression(), index, integer_list<std::size_t, is_merged>());
	}
};


////////////////////////TENSOR AXIS SPLIT//////////////////////
//only the part where an un-merged axis is split is taken into account, mainly to 
//allow constructs like reshape(A, merge<2>(), split<2>...) where the split is performed
//on the merged object. allowing the split of a merged axis would need yet another proxy
//which is a lot of work for very little gain - the case is almost theoretic.
template<class TensorE, std::size_t NE, std::size_t N>
struct axis_split_optimizer<merge_proxy<TensorE, NE>, N>{
	static_assert(N != NE, "Not implemented: can't split a merged axis, represented by a merge_proxy. Create a temporary instead before splitting.");

	//implement by first splitting E and afterwards creating a new merge_proxy object
	typedef axis_split_optimizer<TensorE, (N > NE? N + 1 : N)> opt;
	typedef merge_proxy<typename opt::type, (N > NE? NE: NE + 1) > type;
	
	static type create(
		merge_proxy<TensorE, NE> const& E, std::size_t size1, std::size_t size2
	){
		return opt::create(E.expression(),size1, size2);
	}
};



////////////////////////TENSOR AXIS MERGE//////////////////////
template<class TensorE, std::size_t NE, std::size_t N>
struct axis_merge_optimizer<merge_proxy<TensorE, NE>, N>{
	typedef merge_proxy<merge_proxy<TensorE, NE>, N> type;
	static constexpr std::size_t needs_proxy = true;
	static type create(merge_proxy<TensorE, NE> const& E){
		return E;
	}
};

////////////////////////TENSOR PERMUTE//////////////////////
template<class TensorE, std::size_t NE, class Axis>
struct axis_permute_optimizer<merge_proxy<TensorE, NE>, Axis >{
	//main difficulty here is to construct the permutation of E.
	static constexpr std::size_t NPermuted = Axis::template index_of_v<NE>;
	typedef typename Axis::template split_t<NPermuted> EAxis;
	
	typedef axis_permute_optimizer<TensorE, EAxis> opt;
	typedef merge_proxy<typename opt::type, NPermuted> type;

	static type create(merge_proxy<TensorE, NE> arg){
		return opt::create(arg.expression());
	}
};


}}
#endif
