/*!
 * 
 *
 * \brief       Matrix proxy expressions
 * 
 * 
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

#ifndef REMORA_MATRIX_PROXY_HPP
#define REMORA_MATRIX_PROXY_HPP

#include "detail/proxy_optimizers_fwd.hpp"
#include "expression_types.hpp"
#include "detail/traits.hpp"
#include "detail/check.hpp"
#include <tuple> //std::make_tuple etc
#include <array> //std::array
#include <utility> //std::integer_sequence
namespace remora{
	
namespace ax{
	/// \brief Defines a slice of a single dimension
	///
	/// range{start,end} is used to pick a slice of a dimension of a tensor.
	struct range{
		range(std::size_t start, std::size_t end): start(start), end(end){}
		std::size_t start;
		std::size_t end;
		
	};

	template<int N>
	struct merge{
		static_assert(N != 0, "merge<0> has no meaning");
		static constexpr unsigned num_input_axes = (N<0)? 0: N;
		static constexpr unsigned num_output_axes = 1;
	};

	template<std::size_t N>
	struct split{
		static constexpr unsigned num_input_axes = 1;
		static constexpr unsigned num_output_axes = N;
		template<class ...Size, class = typename std::enable_if<sizeof...(Size) == N,void>::type >
		constexpr split(Size... sizes): shape(std::size_t(sizes)...){}
		
		constexpr split(tensor_shape<N> shape): shape(shape){}
		tensor_shape<N> shape;
	};
	
	///\brief tag object describing taking one complete axis
	///
	/// This is equivalent to merge<1> which is equivalent to ':' in tensorflow/pytorch.
	merge<1> all = merge<1>();

	///\brief tag object describing an axis that has a wildcard size and can span multiple axis
	///
	/// The semantic meaning is that of merge<N> where N is chosen apropriately to make the size fit.
	/// This is roughly equivalent to "-1" as an axis-dimension in tensorflow/pytorch.
	/// Only one fit tag is allowed in any reshape operation
	merge<-1> fit = merge<-1>();
}



	
////////////////////////////////////
//// Tensor reshape
////////////////////////////////////


namespace detail{
	template<class Arg, class... Args>
	struct reshape_used_input_dims{
		static constexpr unsigned value = Arg::num_input_axes + reshape_used_input_dims<Args...>::value;
	};
	template<class Arg>
	struct reshape_used_input_dims<Arg>{
		static constexpr unsigned value = Arg::num_input_axes;
	};
	
	template<std::size_t N, class T>
	T reshape_handle_fit(T t){
		return t;
	}
	template<std::size_t N>
	ax::merge<N> reshape_handle_fit(ax::merge<-1> t){
		return ax::merge<N>();
	}
	
	template<std::size_t CurDim, class... Args>
	struct reshape_dispatcher;
	
	//end of recursion
	template<std::size_t CurDim>
	struct reshape_dispatcher<CurDim>{
		template<class TensorA>
		static auto create(TensorA const& A){
			static_assert(CurDim == TensorA::axis::num_dims, "reshape implementation Error!");
			return A;
		}
	};
	//implementation of all (Identity)
	template<std::size_t CurDim, class... Args>
	struct reshape_dispatcher<CurDim, ax::merge<1>, Args... >{
		template<class TensorA>
		static auto create(TensorA const& A, ax::merge<1>, Args... args){
			return reshape_dispatcher<CurDim + 1, Args...>::create(A, args...);
		}
	};
	
	//implementation of split with two axes
	template<std::size_t CurDim, class... Args>
	struct reshape_dispatcher<CurDim, ax::split<2>, Args... >{
		template<class TensorA>
		static auto create(TensorA const& A, ax::split<2> arg, Args... args){
			//check the split object fits the axis.
			REMORA_SIZE_CHECK(arg.shape.num_elements() == A.shape()[CurDim]);
			//split and recurse
			auto Asplit = axis_split_optimizer<TensorA, CurDim>::create(A, arg.shape[0], arg.shape[1]);
			return reshape_dispatcher<CurDim + 2, Args...>::create(Asplit, args...);
		}
	};
	
	//implementation of split with more than two axes
	template<std::size_t CurDim, std::size_t K, class... Args>
	struct reshape_dispatcher<CurDim, ax::split<K>, Args... >{
		template<class TensorA>
		static auto create(TensorA const& A, ax::split<K> arg, Args... args){
			//check the split object fits the axis.
			REMORA_SIZE_CHECK(arg.shape.num_elements() == A.shape()[CurDim]);
			//split off the first new dimension from the axis
			std::size_t tail_size = arg.shape.num_elements()/arg.shape[0];
			auto Asplit = axis_split_optimizer<TensorA, CurDim>::create(A, arg.shape[0], tail_size);
			//create a split-object for the remainder
			tensor_shape<K - 1> tail_shape;
			for (std::size_t i = 0; i != K - 1; ++i){
				tail_shape[i] = arg.shape[i+1];
			}
			return reshape_dispatcher<CurDim + 1, ax::split<K-1>, Args...>::create(Asplit,ax::split<K-1>{tail_shape}, args...);
		}
	};

	//implementation of merge with two axes
	template<std::size_t CurDim, class... Args>
	struct reshape_dispatcher<CurDim, ax::merge<2>, Args... >{
		template<class TensorA>
		static auto create(TensorA const& A, ax::merge<2> arg, Args... args){
			constexpr unsigned ax0 = TensorA::axis::template element_v<CurDim>;
			constexpr unsigned ax1 = TensorA::axis::template element_v<CurDim + 1>;
			static_assert (ax0 + 1 == ax1, "Not Implemented: merge with non-consecutive underlying axes.");
			auto Amerged = axis_merge_optimizer<TensorA, CurDim>::create(A);
			return reshape_dispatcher<CurDim + 1, Args...>::create(Amerged, args...);
		}
	};

	//implementation of merge with more than two axes (use that a merge of N axis can be written as N-1 merges of 2 axes)
	template<std::size_t CurDim, int K, class... Args>
	struct reshape_dispatcher<CurDim, ax::merge<K>, Args... >{
		template<class TensorA>
		static auto create(TensorA const& A, ax::merge<K> arg, Args... args){
			constexpr unsigned ax0 = TensorA::axis::template element_v<CurDim>;
			constexpr unsigned ax1 = TensorA::axis::template element_v<CurDim + 1>;
			static_assert (ax0 + 1 == ax1, "Not Implemented: merge with non-consecutive underlying axes.");
			auto Amerged = axis_merge_optimizer<TensorA, CurDim>::create(A);
			return reshape_dispatcher<CurDim, ax::merge<K-1> , Args...>::create(Amerged, ax::merge<K-1>(), args...);
		}
	};
	
}

/// \brief Reshapes a tensor to a different shape
///
/// Reshapes a vector given a set of semantic axis arguments to a target size.
/// The order of arguments is used to define which axis the arguments are applied to.
/// There are four different arguments that can be provided,
/// ax::all, ax::merge<N>, ax::split and ax::fit. 
/// Their semantic meaning is as follows:
///
/// split<N> takes the next axis of  A and splits them into the next N axes by B.
///     split takes N arguments which is the shape of the dimensions. Note that the
///     split is taken using the default-order of axis<0,1,...,N-1>, that is the first axis is the
///     leading dimension.
/// all takes the next axis from A and maps it to the next axis of B without changing its shape. 
///
/// merge<N> takes the next N consecutive axes of A and maps them to the next single axis of B
///
/// fit is equivalent to merge<N> with N chosen such that all unused input axes are merged.
///     fit can only be used once.
///
/// For a valid call to reshape, the number of axes in A and the number of used arguments by the operands must agree.
/// When using fit, this is always the case.
///
/// An example (assuming namespace remora is included)
/// A is a 6 dimensional tensor of shape (3,2,5,30,3,7)
/// B=reshape(A, merge<2>(), all, split<3>(3,5,2), fit)
/// will result in B with shape (6,5,3,5,2,21)
/// merge takes the first two axes of A and merges them as first axis of B. 
/// all takes the next axis (3) and maps it to the second axis of B.
/// split<3> takes axis 4 and splits it into the next 3 axis of B.
/// fit is replaced by merge<2> to take up the last unused axes of B which is then put at the last axis.
///
/// Todo: merge is not fully implemented and currently only allows merging of consecutive axes
/// as indicated by the tensor layout. This only affects Tensors with more than 2 axes. Normally,
/// A tensor does not have this issue, however when permuting axis, e.g. B= permute(A, 0,2,1)
/// a following call to reshape(B,merge<2>(), all) will fail.
/// 
/// Further, note that split or merge can not be applied to structured tensors if the split/merge operation
/// Destroys the structure.
template <std::size_t Dim, class TensorA, class Device, class... Args>
auto reshape(
	tensor_expression<Dim, TensorA, Device>& A, Args... args
){
	// handle fit and ensure that arguments lead to correct number of axis
	constexpr unsigned used_dims_pre_fill = detail::reshape_used_input_dims<Args...>::value;
	static_assert(used_dims_pre_fill <= TensorA::axis::num_dims, "too many axes used by arguments.");
	constexpr unsigned fill_size = TensorA::axis::num_dims - used_dims_pre_fill;
	constexpr unsigned used_dims_post_fill = detail::reshape_used_input_dims<decltype(detail::reshape_handle_fit<fill_size>(args))...>::value;
	static_assert(used_dims_post_fill == TensorA::axis::num_dims, "axis arguments do not lead to correct number of axis. Check that at most one fit argument is used");
	
	typename TensorA::closure_type Aclosure = A;
	return detail::reshape_dispatcher<0, decltype(detail::reshape_handle_fit<fill_size>(args))...>::create(
		Aclosure, detail::reshape_handle_fit<fill_size>(args)...
	);
}
template <std::size_t Dim, class TensorA, class Device, class... Args>
auto reshape(
	tensor_expression<Dim, TensorA, Device> const& A, Args... args
){
	typename TensorA::const_closure_type Aclosure = A;
	return reshape(Aclosure, args...);
}
template <std::size_t Dim, class TensorA, class Device, class... Args>
auto reshape(
	tensor_expression<Dim, TensorA, Device> && A, Args... args
){
	static_assert(!std::is_base_of<tensor_container<Dim, TensorA, Device>,TensorA>::value, "It is unsafe to create a proxy from a temporary container");
	return reshape(A, args...);
}


////////////////////////////////////
//// Tensor Axis Permutation
////////////////////////////////////

template <std::size_t Dim, class TensorA, class Device, unsigned... Axes>
auto permute(tensor_expression<Dim, TensorA, Device>& A, axis<Axes...> permutation){
	return detail::axis_permute_optimizer<typename TensorA::closure_type, axis<Axes...> >::create(A());
}

template <std::size_t Dim, class TensorA, class Device, unsigned... Axes>
auto permute(tensor_expression<Dim, TensorA, Device> const& A, axis<Axes...> permutation){
	return detail::axis_permute_optimizer<typename TensorA::const_closure_type, axis<Axes...> >::create(A());
}

template <std::size_t Dim, class TensorA, class Device, unsigned... Axes>
auto permute(tensor_expression<Dim, TensorA, Device> && A, axis<Axes...> permutation){
	static_assert(!std::is_base_of<tensor_container<Dim, TensorA, Device>,TensorA>::value, "It is unsafe to create a proxy from a temporary container");
	return permute(A, permutation);
}

////////////////////////////////////
//// Tensor-slice
////////////////////////////////////

namespace detail{
	template <std::size_t N, std::size_t Dim, class TensorA, class Device>
	typename TensorA::closure_type slice_helper(
		tensor_expression<Dim, TensorA, Device> const& A, ax::merge<1>
	){
		return A();
	}

	template <std::size_t N, class TensorA>
	auto slice_helper(
		TensorA const& A, ax::range const& range
	){
		return detail::subrange_optimizer<typename TensorA::closure_type, N>::create(A, range.start, range.end);
	}
	
	template <std::size_t N, class TensorA>
	auto slice_helper(
		TensorA const& A, std::size_t index
	){
		return detail::slice_optimizer<typename TensorA::closure_type, N>::create(A,index);
	}
	
	
	template <unsigned Nsliced, class TensorA, class... Args, unsigned N, unsigned I>
	auto slice_dispatcher(
		TensorA const& A, std::tuple<Args...> const& args, axis_set<N>, axis_set<I>
	){
		constexpr std::size_t slice_ax = TensorA::axis::template index_of_v<N-Nsliced>;
		return slice_helper<slice_ax>(A, std::get<I>(args));
	}
	
	template <unsigned Nsliced, class TensorA, class... Args, unsigned... Ns, unsigned... Is>
	auto slice_dispatcher(
		TensorA const& A, std::tuple<Args...> const& args, axis_set<Ns...>, axis_set<Is...>
	){
		// get the next axis to slice
		constexpr unsigned ax = axis_set<Ns...>::min_element; //axis of original A to slice next
		constexpr std::size_t ax_pos = axis_set<Ns...>::template index_of_v<ax>; //index of axis to slice next in the axis_set
		constexpr std::size_t args_pos = axis_set<Is...>::template element_v<ax_pos>; //Argument to use next
		constexpr std::size_t slice_ax = TensorA::axis::template index_of_v<ax-Nsliced>;
		auto sliced_Ns = typename axis_set<Ns...>::template remove_t<ax_pos>(); //remove the index from the set
		auto sliced_Is = typename axis_set<Is...>::template remove_t<ax_pos>(); //remove the index from the set
		// slice tensor
		auto Asliced = slice_helper<slice_ax>(A, std::get<args_pos>(args));
		
		//check if we sliced away a dimension
		constexpr unsigned axDiff = TensorA::axis::num_dims - decltype(Asliced)::axis::num_dims;
		return slice_dispatcher<Nsliced + axDiff>(Asliced, args, sliced_Ns, sliced_Is);
	}
	
}



template<std::size_t Dim, class TensorA, class Device, class... Args>
auto slice(tensor_expression<Dim, TensorA, Device>& A, Args... args){
	typename TensorA::closure_type Aclosure = A;
	constexpr std::size_t num_slice = sizeof...(Args);
	auto axis = typename TensorA::axis:: template front_t<num_slice>();
	auto inds = default_axis<num_slice>();
	return detail::slice_dispatcher<0>(Aclosure, std::make_tuple(args...), axis, inds);
}

template<std::size_t Dim, class TensorA, class Device, class... Args>
auto slice(tensor_expression<Dim, TensorA, Device> const& A, Args... args){
	typename TensorA::const_closure_type Aclosure = A;
	return slice(Aclosure, args...);
}

template<std::size_t Dim, class TensorA, class Device, class... Args>
auto slice(tensor_expression<Dim, TensorA, Device> && A, Args... args){
	static_assert(!std::is_base_of<tensor_container<Dim, TensorA, Device>,TensorA>::value, "It is unsafe to create a proxy from a temporary container");
	return slice(A, args...);
}

/*

////////////////////////////////////
//// Matrix Diagonal
////////////////////////////////////

///\brief Returns the diagonal of a constant square matrix as vector.
///
/// given a matrix 
/// A = (1 2 3)
///     (4 5 6)
///     (7 8 9)
///
/// the diag operation results in
/// diag(A) = (1,5,9)
template<class M, class Device>
typename detail::matrix_diagonal_optimizer<typename M::closure_type>::type
diag(matrix_expression<M, Device>& mat){
	REMORA_SIZE_CHECK(mat().size1() == mat().size2());
	return detail::matrix_diagonal_optimizer<typename M::closure_type>::create(mat());
}

template<class M, class Device>
typename detail::matrix_diagonal_optimizer<typename M::const_closure_type>::type
diag(matrix_expression<M, Device> const& mat){
	REMORA_SIZE_CHECK(mat().size1() == mat().size2());
	return detail::matrix_diagonal_optimizer<typename M::const_closure_type>::create(mat());
}


template<class M, class Device>
typename detail::matrix_diagonal_optimizer<typename M::closure_type>::type
diag(matrix_expression<M, Device> && m){
	static_assert(!std::is_base_of<matrix_container<M, Device>,M>::value, "It is unsafe to create a proxy from a temporary container");
	return diag(m());
}
////////////////////////////////////////////////
//// Matrix to Triangular Matrix
////////////////////////////////////////////////

template <class M, class Device, class Tag>
typename detail::triangular_proxy_optimizer<typename M::closure_type, Tag>::type
to_triangular(matrix_expression<M, Device>& m, Tag){
	return detail::triangular_proxy_optimizer<typename M::closure_type, Tag>::create(m());
}

template <class M, class Device, class Tag>
typename detail::triangular_proxy_optimizer<typename M::const_closure_type, Tag>::type
to_triangular(matrix_expression<M, Device> const& m, Tag){
	return detail::triangular_proxy_optimizer<typename M::const_closure_type, Tag>::create(m());
}

template <class M, class Device, class Tag>
typename detail::triangular_proxy_optimizer<typename M::closure_type, Tag>::type
to_triangular(matrix_expression<M, Device>&& m, Tag){
	static_assert(!std::is_base_of<matrix_container<M, Device>,M>::value, "It is unsafe to create a proxy from a temporary container");
	return detail::triangular_proxy_optimizer<typename M::closure_type, Tag>::create(m());
}


////////////////////////////////////
//// Matrix to vector set
////////////////////////////////////

template <class O, class M, class Device>
vector_set<typename M::const_closure_type, O >
as_set(matrix_expression<M, Device> const& m, O){
	return vector_set<typename M::const_closure_type, O >(m());
}

template <class O, class M, class Device>
vector_set<typename M::closure_type, O >
as_set(matrix_expression<M, Device>& m, O){
	return vector_set<typename M::closure_type, O >(m());
}

template <class O, class M, class Device>
vector_set<typename M::closure_type, O > 
as_set(matrix_expression<M, Device>&& m, O){
	static_assert(!std::is_base_of<matrix_container<M, Device>,M>::value, "It is unsafe to create a proxy from a temporary container");
	return vector_set<typename M::closure_type, O >(m());
}

/// \brief Transforms the matrix m to a set of points where each point is one row of m
template <class M>
auto as_rows(M&& m)-> decltype(as_set(std::forward<M>(m), row_major())){
	return as_set(std::forward<M>(m), row_major());
}

/// \brief Transforms the matrix m to a set of points where each point is one column of m
template <class M>
auto as_columns(M&& m)-> decltype(as_set(std::forward<M>(m), column_major())){
	return as_set(std::forward<M>(m), column_major());
}
*/
}

#endif
