/*!
 * \brief       Expression Optimizations
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
 #ifndef REMORA_EXPRESSION_OPTIMIZERS_HPP
#define REMORA_EXPRESSION_OPTIMIZERS_HPP

#include "proxy_optimizers_fwd.hpp"
#include "expression_classes.hpp"

namespace remora{namespace detail{
	
//forward declarations
// template<class Tensor, class V>
// struct matrix_vector_prod_optimizer;
// template<class TensorA, class TensorB>
// struct matrix_matrix_prod_optimizer;
	
// template<class TensorA, class TensorB, class Tag, class Side>
// struct matrix_matrix_solve_optimizer;
// template<class Tensor, class V, class Tag, class Side>
// struct matrix_vector_solve_optimizer;
	
// template<class Tensor, class Tag>
// struct matrix_inverse_optimizer;

template<class Tensor,  class F>
struct tensor_unary_optimizer;


////////////////////////////////////
//// Permute
////////////////////////////////////

//(alpha A)^T = alpha A^T
template<class Tensor, class Axis>
struct axis_permute_optimizer<scalar_multiply<Tensor>, Axis >{
	typedef axis_permute_optimizer<typename Tensor::const_closure_type, Axis> opt;
	typedef scalar_multiply<typename opt::type> type;
	
	static type create(scalar_multiply<Tensor> const& E){
		return {opt::create(E.expression()), E.scalar()};
	}
};

//(A+B)^T=A^T+B^T
template<class TensorA, class TensorB, class Axis>
struct axis_permute_optimizer<tensor_addition<TensorA,TensorB>, Axis >{
	typedef axis_permute_optimizer<typename TensorA::const_closure_type, Axis > left_opt;
	typedef axis_permute_optimizer<typename TensorB::const_closure_type, Axis > right_opt;
	typedef tensor_addition<typename left_opt::type,typename right_opt::type > type;
	
	static type create(tensor_addition<TensorA,TensorB> const& E){
		return {left_opt::create(E.lhs()),right_opt::create(E.rhs())};
	}
};

//trans(constant)  -> constant (swapped sizes)
template<class T, class Device, class OAxis, unsigned... Ns>
struct axis_permute_optimizer<scalar_tensor<T, OAxis, Device>, axis<Ns...> >{
	typedef scalar_tensor<T, typename OAxis::template permute_t<Ns...>, Device > type;
	static type create(scalar_tensor<T, OAxis, Device> const& E){
		return {axis<Ns...>::to_axis(E.shape()), E.scalar()};
	}
};

//f(A)^T = f(A^T) for f(A)_ij=f(A_ij)
template<class Tensor, class F, class Axis>
struct axis_permute_optimizer<tensor_unary<Tensor,F>, Axis >{
	typedef axis_permute_optimizer<typename Tensor::const_closure_type, Axis> opt;
	typedef tensor_unary<typename opt::type, F> type;
	
	static type create(tensor_unary<Tensor,F> const& E){
		return {opt::create(E.expression()),E.functor()};
	}
};

//f(A,B)^T=f(A^T,B^T)
template<class TensorA, class TensorB, class F, class Axis>
struct axis_permute_optimizer<tensor_binary<TensorA,TensorB, F>, Axis>{
	typedef axis_permute_optimizer<typename TensorA::const_closure_type, Axis> left_opt;
	typedef axis_permute_optimizer<typename TensorB::const_closure_type, Axis> right_opt;
	typedef tensor_binary<typename left_opt::type,typename right_opt::type, F > type;
	
	static type create(tensor_binary<TensorA,TensorB,F> const& E){
		return {left_opt::create(E.lhs()),right_opt::create(E.rhs()),E.functor()};
	}
};

//broadcasting
template<class Tensor, class OAxis, class DropList, unsigned... Permutation>
struct axis_permute_optimizer<tensor_broadcast<Tensor, OAxis, DropList >, axis<Permutation...> >{
	typedef axis<Permutation...> Axis;
	typedef typename OAxis::template permute_t<Permutation...> permuted_axis;
	typedef typename DropList::template select_t<Permutation...> permuted_drops;
	typedef typename detail::filter_slice<Axis, DropList >::type sliced_permutation;
	
	typedef axis_permute_optimizer<typename Tensor::const_closure_type, sliced_permutation> opt;
	typedef tensor_broadcast<typename opt::type, permuted_axis, permuted_drops> type;
	
	static type create(tensor_broadcast<Tensor, OAxis, DropList > const& E){
		auto shape = Axis::to_axis(E.shape());
		return {opt::create(E.expression()), shape};
	}
};

/*
//vector repeater behaves as outer product to: (v 1^T)^T = (1 v^T)
template<class V, class OAxis>
struct axis_permute_optimizer<vector_repeater<V,OAxis> >{
	typedef vector_repeater<V,typename OAxis::transposed_orientation> type;
	
	static type create(vector_repeater<V,OAxis> const& E){
		return type(E.expression(),E.num_repetitions());
	}
};

//(v1 v2^T)^T = v2 v1^T
template<class V1, class V2>
struct axis_permute_optimizer<outer_product<V1,V2> >{
	typedef outer_product<V2,V1> type;
	
	static type create(outer_product<V1,V2> const& E){
		return type(E.rhs(),E.lhs());
	}
};

//(A B)^T = A^T B^T
template<class TensorA, class TensorB>
struct axis_permute_optimizer<matrix_matrix_prod<TensorA,TensorB> >{
	typedef axis_permute_optimizer<typename TensorB::const_closure_type> left_opt;
	typedef axis_permute_optimizer<typename TensorA::const_closure_type> right_opt;
	typedef matrix_matrix_prod_optimizer<typename left_opt::type,typename right_opt::type> opt;
	typedef typename opt::type type;
	
	static type create(matrix_matrix_prod<TensorA,TensorB> const& E){
		return opt::create(left_opt::create(E.rhs()),right_opt::create(E.lhs()));
	}
};

//trans(diagonal)  =  diagonal
template<class V>
struct axis_permute_optimizer<diagonal_matrix<V> >{
	typedef diagonal_matrix<V> type;
	static type const& create(type const& E){
		return E;
	}
};

//(A | B)^T= (A^T & B^T)
//(A & B)^T= (A^T | B^T)
template<class TensorA, class TensorB, bool B>
struct axis_permute_optimizer<matrix_concat<TensorA,TensorB,B> >{
	typedef axis_permute_optimizer<typename TensorB::const_closure_type> right_opt;
	typedef axis_permute_optimizer<typename TensorA::const_closure_type> left_opt;
	typedef matrix_concat<typename left_opt::type,typename right_opt::type,!B > type;
	
	static type create(matrix_concat<TensorA,TensorB,B> const& E){
		return type(left_opt::create(E.lhs()),right_opt::create(E.rhs()));
	}
};*/



////////////////////////////////////
//// Merge
////////////////////////////////////

//slice(alpha A,i) = alpha slice(A,i)
template<class Tensor, std::size_t N>
struct axis_merge_optimizer<scalar_multiply<Tensor>, N >{
	typedef axis_merge_optimizer<typename Tensor::const_closure_type, N> opt;
	typedef scalar_multiply<typename opt::type> type;
	
	static type create(scalar_multiply<Tensor> const& E){
		return type(opt::create(E.expression()), E.scalar());
	}
};

// slice(A+B,i) = slice(A,i) + slice(B,i)
template<class TensorA, class TensorB, std::size_t N>
struct axis_merge_optimizer<tensor_addition<TensorA,TensorB>, N>{
	typedef axis_merge_optimizer<typename TensorA::const_closure_type, N> left_opt;
	typedef axis_merge_optimizer<typename TensorB::const_closure_type, N> right_opt;
	typedef tensor_addition<typename left_opt::type,typename right_opt::type > type;
	
	static type create(tensor_addition<TensorA,TensorB> const& E){
		return type(left_opt::create(E.lhs()),right_opt::create(E.rhs()));
	}
};

//slice(constant,i) = constant
template<class T, class Axis, class Device, std::size_t N>
struct axis_merge_optimizer<scalar_tensor<T, Axis, Device>, N>{
	typedef scalar_tensor<T, typename Axis::template slice_t<N>, Device> type;
	
	static type create(scalar_tensor<T, Axis, Device> const& E){
		auto shape = E.shape();
		auto new_shape = shape.slice(N);
		new_shape[N] *= shape[N];
		return type(new_shape,E.scalar());
	}
};

//slice(f(A),i) = f(slice(A,i))
template<class Tensor, class F, std::size_t N>
struct axis_merge_optimizer<tensor_unary<Tensor,F>, N>{
	typedef axis_merge_optimizer<typename Tensor::const_closure_type, N> opt;
	typedef tensor_unary<typename opt::type, F> type;
	
	static type create(tensor_unary<Tensor,F> const& E){
		return type(opt::create(E.expression()),E.functor());
	}
};

//slice(f(A,B),i)=f(slice(A,i),slice(B,i))
template<class TensorA, class TensorB, class F, std::size_t N>
struct axis_merge_optimizer<tensor_binary<TensorA,TensorB, F>, N>{
	typedef axis_merge_optimizer<typename TensorA::const_closure_type, N> left_opt;
	typedef axis_merge_optimizer<typename TensorB::const_closure_type, N> right_opt;
	typedef tensor_binary<typename left_opt::type,typename right_opt::type, F > type;
	
	static type create(tensor_binary<TensorA,TensorB,F> const& E){
		return type(left_opt::create(E.lhs()),right_opt::create(E.rhs()),E.functor());
	}
};

////////////////////////////////////
//// Split
////////////////////////////////////

//slice(alpha A,i) = alpha slice(A,i)
template<class Tensor, std::size_t N>
struct axis_split_optimizer<scalar_multiply<Tensor>, N >{
	typedef axis_split_optimizer<typename Tensor::const_closure_type, N> opt;
	typedef scalar_multiply<typename opt::type> type;
	
	static type create(scalar_multiply<Tensor> const& E, std::size_t size1, std::size_t size2){
		return type(opt::create(E.expression(), size1, size2), E.scalar());
	}
};

// slice(A+B,i) = slice(A,i) + slice(B,i)
template<class TensorA, class TensorB, std::size_t N>
struct axis_split_optimizer<tensor_addition<TensorA,TensorB>, N>{
	typedef axis_split_optimizer<typename TensorA::const_closure_type, N> left_opt;
	typedef axis_split_optimizer<typename TensorB::const_closure_type, N> right_opt;
	typedef tensor_addition<typename left_opt::type,typename right_opt::type > type;
	
	static type create(tensor_addition<TensorA,TensorB> const& E, std::size_t size1, std::size_t size2){
		return type(left_opt::create(E.lhs(), size1, size2),right_opt::create(E.rhs(), size1, size2));
	}
};

//slice(constant,i) = constant
template<class T, class Axis, class Device, std::size_t N>
struct axis_split_optimizer<scalar_tensor<T, Axis, Device>, N>{
	typedef scalar_tensor<T, typename Axis::template slice_t<N>, Device> type;
	
	static type create(scalar_tensor<T, Axis, Device> const& E, std::size_t size1, std::size_t size2){
		auto shape = E.shape();
		tensor_shape<Axis::num_dims+1> new_shape;
		for(unsigned i = 0; i != N; ++i){
			new_shape[i] = shape[i];
		}
		new_shape[N] = size1;
		new_shape[N+1] = size2;
		for(unsigned i = N + 1; i != Axis::num_dims; ++i){
			new_shape[i + 1] = shape[i];
		}
		return type(new_shape,E.scalar());
	}
};

//slice(f(A),i) = f(slice(A,i))
template<class Tensor, class F, std::size_t N>
struct axis_split_optimizer<tensor_unary<Tensor,F>, N>{
	typedef axis_split_optimizer<typename Tensor::const_closure_type, N> opt;
	typedef tensor_unary<typename opt::type, F> type;
	
	static type create(tensor_unary<Tensor,F> const& E, std::size_t size1, std::size_t size2){
		return type(opt::create(E.expression(), size1, size2),E.functor());
	}
};

//slice(f(A,B),i)=f(slice(A,i),slice(B,i))
template<class TensorA, class TensorB, class F, std::size_t N>
struct axis_split_optimizer<tensor_binary<TensorA,TensorB, F>, N>{
	typedef axis_split_optimizer<typename TensorA::const_closure_type, N> left_opt;
	typedef axis_split_optimizer<typename TensorB::const_closure_type, N> right_opt;
	typedef tensor_binary<typename left_opt::type,typename right_opt::type, F > type;
	
	static type create(tensor_binary<TensorA,TensorB,F> const& E, std::size_t size1, std::size_t size2){
		return type(left_opt::create(E.lhs(), size1, size2),right_opt::create(E.rhs(), size1, size2),E.functor());
	}
};

////////////////////////////////////
//// Slice
////////////////////////////////////

//slice(alpha A,i) = alpha slice(A,i)
template<class Tensor, std::size_t N>
struct slice_optimizer<scalar_multiply<Tensor>, N >{
	typedef slice_optimizer<typename Tensor::const_closure_type, N> opt;
	typedef scalar_multiply<typename opt::type> type;
	
	static type create(scalar_multiply<Tensor> const& E, std::size_t i){
		return type(opt::create(E.expression(),i), E.scalar());
	}
};

// slice(A+B,i) = slice(A,i) + slice(B,i)
template<class TensorA, class TensorB, std::size_t N>
struct slice_optimizer<tensor_addition<TensorA,TensorB>, N>{
	typedef slice_optimizer<typename TensorA::const_closure_type, N> left_opt;
	typedef slice_optimizer<typename TensorB::const_closure_type, N> right_opt;
	typedef tensor_addition<typename left_opt::type,typename right_opt::type > type;
	
	static type create(tensor_addition<TensorA,TensorB> const& E, std::size_t i){
		return type(left_opt::create(E.lhs(),i),right_opt::create(E.rhs(),i));
	}
};

//slice(constant,i) = constant
template<class T, class Axis, class Device, std::size_t N>
struct slice_optimizer<scalar_tensor<T, Axis, Device>, N>{
	typedef scalar_tensor<T, typename Axis::template slice_t<N>, Device> type;
	
	static type create(scalar_tensor<T, Axis, Device> const& E, std::size_t){
		return type(E.shape().slice(N), E.scalar());
	}
};

//slice(f(A),i) = f(slice(A,i))
template<class Tensor, class F, std::size_t N>
struct slice_optimizer<tensor_unary<Tensor,F>, N>{
	typedef slice_optimizer<typename Tensor::const_closure_type, N> opt;
	typedef tensor_unary<typename opt::type, F> type;
	
	static type create(tensor_unary<Tensor,F> const& E, std::size_t i){
		return type(opt::create(E.expression(),i),E.functor());
	}
};

//slice(f(A,B),i)=f(slice(A,i),slice(B,i))
template<class TensorA, class TensorB, class F, std::size_t N>
struct slice_optimizer<tensor_binary<TensorA,TensorB, F>, N>{
	typedef slice_optimizer<typename TensorA::const_closure_type, N> left_opt;
	typedef slice_optimizer<typename TensorB::const_closure_type, N> right_opt;
	typedef tensor_binary<typename left_opt::type,typename right_opt::type, F > type;
	
	static type create(tensor_binary<TensorA,TensorB,F> const& E, std::size_t i){
		return type(left_opt::create(E.lhs(),i),right_opt::create(E.rhs(),i),E.functor());
	}
};

template<int /*0*/, class Tensor, class AxisTrans, class DropTrans, std::size_t N>
struct slice_optimizer_broadcast{
	//case where N is not dropped
	//count the number of axis dropped before N to get the proper axis to slice
	static constexpr unsigned num_smaller(){
		auto arr = DropTrans::to_array();
		std::size_t count = 0;
		for(std::size_t i = 0; i != N; ++i){
			count += arr[i];
		}
		return count;
	}
	typedef slice_optimizer<Tensor, N - num_smaller()> slice_opt;
	typedef tensor_broadcast<typename slice_opt::type, AxisTrans, DropTrans > type;
	
	template<class TensorE>
	static type create(TensorE const& E, std::size_t i){
		return {slice_opt::create(E.expression(), i), E.shape().slice(N)};
	}
};

template<class Tensor, class AxisTrans, class DropTrans, std::size_t N>
struct slice_optimizer_broadcast<1,Tensor, AxisTrans, DropTrans, N>{
	//case where N is dropped
	typedef tensor_broadcast<Tensor, AxisTrans, DropTrans > type;
	
	template<class TensorE>
	static type create(TensorE const& E, std::size_t i){
		return {E.expression(), E.shape().slice(N)};
	}
};

template<class Tensor, class AxisTrans, std::size_t N>
struct slice_optimizer_broadcast<1, Tensor, AxisTrans, constant_integer_list<bool, false, AxisTrans::num_dims>, N>{
	//case where N is dropped and the resulting drop list is empty
	//here we just return the inner tensor.
	typedef typename Tensor::const_closure_type type;
	
	template<class TensorE>
	static type create(TensorE const& E, std::size_t){
		return E.expression();
	}
};


//case where we remove the last axis of Tensor
//here the result is a scalar_tensor
template<class Tensor, class AxisTrans, class DropTrans, std::size_t N>
struct slice_optimizer_broadcast<2,Tensor, AxisTrans, DropTrans, N>{
	typedef slice_optimizer<Tensor, 0> slice_opt;
	typedef scalar_tensor<typename Tensor::value_type, AxisTrans, typename Tensor::device_type> type;
	
	template<class TensorE>
	static type create(TensorE const& E, std::size_t i){
		return {E.shape().slice(N), slice_opt::create(E.expression(), i)};
	}
};


//dispatcher for slice implementation of broadcast
template<class Tensor, class Axis, class DropList, std::size_t N>
struct slice_optimizer<tensor_broadcast<Tensor, Axis, DropList >, N>{
	//check whether N is in the droplist
	static constexpr std::size_t droppedN = DropList::template element_v<N>;
	// check whether we are removing the last axis of the Tensor
	static constexpr std::size_t lastAxis = (Tensor::num_dims == (1-droppedN));
	
	//remove the Nth element from droplist and axis
	typedef typename DropList::template remove_t<N> drop_transformed;
	typedef typename Axis::template slice_t<N> axis_transformed;
	
	//reference the optimizer implementation for normal/droppedN/lastAxis
	typedef slice_optimizer_broadcast<droppedN+2*lastAxis, Tensor, axis_transformed, drop_transformed, N> opt;
	typedef typename opt::type type;
	
	static type create(tensor_broadcast<Tensor, Axis, DropList > const& E, std::size_t i){
		return opt::create(E, i);
	}
};

/*
//slice(repeat(v),i) = v if repeat is row_major
template<class V>
struct slice_optimizer<vector_repeater<V, row_major> >{
	typedef typename V::const_closure_type type;
	
	static type create(vector_repeater<V, row_major> const& E, std::size_t){
		return E.expression();
	}
};
//slice(repeat(v),i) = v(i) 1^T if repeat is column_major
template<class V>
struct slice_optimizer<vector_repeater<V, column_major> >{
	typedef scalar_tensor<typename V::value_type, typename V::device_type> type;
	
	static type create(vector_repeater<V, column_major> const& E, std::size_t i){
		return type(E.num_repetitions(), E.expression().elements()(i));
	}
};

//slice(v1 v2^T,i)^T = v(i) v2 
template<class V1, class V2>
struct slice_optimizer<outer_product<V1,V2> >{
	typedef scalar_multiply<V2> type;
	
	static type create(outer_product<V1,V2> const& E, std::size_t i){
		return type(E.rhs(),E.lhs().elements()(i));
	}
};

//slice(prod(A,B),i) = prod(slice(A),B) = prod(trans(B),slice(A)) 
template<class TensorA, class TensorB>
struct slice_optimizer<matrix_matrix_prod<TensorA,TensorB> >{
	typedef slice_optimizer<typename TensorA::const_closure_type> left_opt;
	typedef axis_permute_optimizer<typename TensorB::const_closure_type> right_opt;
	typedef matrix_vector_prod_optimizer<typename right_opt::type, typename left_opt::type> opt;
	typedef typename opt::type type;
	
	static type create(matrix_matrix_prod<TensorA,TensorB> const& E, std::size_t i){
		return opt::create(
			right_opt::create(E.rhs()),
			left_opt::create(E.lhs(),i)
		);
	}
};

//slice(diagonal(V),i)  =  (0,,...,0,v_i,0,...,0)
template<class V>
struct slice_optimizer<diagonal_matrix<V> >{
	typedef unit_vector<typename V::value_type, typename V::device_type> type;
	
	static type create(diagonal_matrix<V> const& E, std::size_t i){
		return type(E.size2(),i,E.expression().elements()(i));
	}
};*/



////////////////////////////////////
//// Subrange
////////////////////////////////////

//range(alpha * A) = alpha * range(A)
template<class Tensor, std::size_t N>
struct subrange_optimizer<scalar_multiply<Tensor>, N>{
	typedef subrange_optimizer<typename Tensor::const_closure_type, N> opt;
	typedef scalar_multiply<typename opt::type > type;
	
	static type create(scalar_multiply<Tensor> const& E,
		std::size_t start, std::size_t end
	){
		return type(opt::create(E.expression(),start,end), E.scalar());
	}
};

//range(A+B) = range(A) + range(B)
template<class TensorA, class TensorB, std::size_t N>
struct subrange_optimizer<tensor_addition<TensorA,TensorB>, N>{
	typedef subrange_optimizer<typename TensorA::const_closure_type, N> left_opt;
	typedef subrange_optimizer<typename TensorB::const_closure_type, N> right_opt;
	typedef tensor_addition<typename left_opt::type, typename right_opt::type > type;
	
	static type create(tensor_addition<TensorA,TensorB> const& E,
		std::size_t start, std::size_t end
	){
		return type(
			left_opt::create(E.lhs(),start,end),
			right_opt::create(E.rhs(),start,end)
		);
	}
};

//range(constant)  -> constant (changed sizes)
template<class T, class Device, class OAxis, std::size_t N>
struct subrange_optimizer<scalar_tensor<T,OAxis, Device>, N>{
	typedef scalar_tensor<T,Device, OAxis> type;
	static type create(type const& E,
		std::size_t start, std::size_t end
	){
		auto shape = E().shape();
		shape[N] = end - start;
		return type(shape, E.scalar());
	}
};

//range(f(A)) = f(range(A))
template<class Tensor, class F, std::size_t N>
struct subrange_optimizer<tensor_unary<Tensor, F>, N>{
	typedef subrange_optimizer<typename Tensor::const_closure_type, N> opt;
	typedef tensor_unary<typename opt::type, F > type;
	
	static type create(tensor_unary<Tensor, F> const& E,
		std::size_t start, std::size_t end
	){
		return type(opt::create(E.expression(),start,end), E.functor());
	}
};

//range(f(A,B)) = f(range(A),range(B))
template<class TensorA, class TensorB, class F, std::size_t N>
struct subrange_optimizer<tensor_binary<TensorA,TensorB, F>, N>{
	typedef subrange_optimizer<typename TensorA::const_closure_type, N> left_opt;
	typedef subrange_optimizer<typename TensorB::const_closure_type, N> right_opt;
	typedef tensor_binary<typename left_opt::type, typename right_opt::type, F > type;
	
	static type create(tensor_binary<TensorA,TensorB,F> const& E,
		std::size_t start, std::size_t end
	){
		return type(
			left_opt::create(E.lhs(),start,end),
			right_opt::create(E.rhs(),start,end),
			E.functor()
		);
	}
};
/*
//repeater behaves like outer_product
template<class V, class OAxis, std::size_t N>
struct subrange_optimizer<vector_repeater<V, OAxis> >{
	typedef vector_range_optimizer<typename V::const_closure_type > vector_opt;
	typedef vector_repeater<typename vector_opt::type, OAxis> type;
	
	static type create(
		vector_repeater<V, OAxis> const& E,
		std::size_t start1, std::size_t end1, std::size_t start2, std::size_t end2
	){
		return type( 
			vector_opt::create(E.expression(),OAxis::index_m(start1,start2),OAxis::index_m(end1,end2)), 
			OAxis::index_M(end1,end2) - OAxis::index_M(start1,start2)
		);
	}
};


//range( u v^T) = range(u) range(v)^T
template<class V1, class V2>
struct subrange_optimizer<outer_product<V1,V2> >{
	typedef vector_range_optimizer<typename V1::const_closure_type > left_opt;
	typedef vector_range_optimizer<typename V2::const_closure_type> right_opt;
	typedef outer_product<typename left_opt::type, typename right_opt::type> type;
	
	static type create(outer_product<V1,V2> const& E,
		std::size_t start1, std::size_t end1, std::size_t start2, std::size_t end2
	){
		return type( left_opt::create(E.lhs(),start1,end1), right_opt::create(E.rhs(),start2,end2));
	}
};

//range(prod(A,B),i) = prod(range(B),range(A)) 
template<class TensorA, class TensorB>
struct subrange_optimizer<matrix_matrix_prod<TensorA,TensorB> >{
	typedef subrange_optimizer<typename TensorA::const_closure_type> left_opt;
	typedef subrange_optimizer<typename TensorB::const_closure_type> right_opt;
	typedef matrix_matrix_prod_optimizer<typename left_opt::type, typename right_opt::type> opt;
	typedef typename opt::type type;
	
	static type create(matrix_matrix_prod<TensorA,TensorB> const& E,
		std::size_t start1, std::size_t end1, std::size_t start2, std::size_t end2
	){
		return opt::create(
			left_opt::create(E.lhs(),start1,end1,0,E.lhs().size2()),
			right_opt::create(E.rhs(),0,E.rhs().size1(),start2,end2)
		);
	}
};


//range(diagonal  -> diagonal padded with 0
template<class V>
struct subrange_optimizer<diagonal_matrix<V> >{
    typedef vector_range_optimizer<typename V::const_closure_type > opt;
	typedef diagonal_matrix<typename opt::type> type;
	static type create(diagonal_matrix<V> const& E,
		std::size_t start1, std::size_t end1, std::size_t start2, std::size_t end2
	){
        REMORA_RANGE_CHECK(start1 == start2);// "unimplemented: non-diagonal subranges of diagonal matrix"
        REMORA_RANGE_CHECK(end1 == end2); //"unimplemented: non-diagonal subranges of diagonal matrix"
        std::size_t startV = std::max(start1,start2);
        std::size_t endV = std::min(end1,end2);
		return type(opt::create(E.expression(),startV, endV));
	}
};*/


////////////////////////////////////
//// Broadcasting
////////////////////////////////////
template<class Tensor, std::size_t N>
struct broadcast_optimizer{
	typedef typename Tensor::axis::template split_t<(N == Tensor::axis::num_dims)? N - 1: N> Axis;
	typedef typename constant_integer_list<bool, false, Tensor::axis::num_dims>::template insert_t<N, true> drop_list;
	typedef tensor_broadcast<Tensor, Axis, drop_list > type;
	
	static type create(typename Tensor::const_closure_type const& E, std::size_t size){
		auto shape = E.shape();
		tensor_shape<Axis::num_dims> new_shape;
		for(unsigned i = 0; i != N; ++i){
			new_shape[i] = shape[i];
		}
		new_shape[N] = size;
		for(unsigned i = N; i != Axis::num_dims - 1; ++i){
			new_shape[i + 1] = shape[i];
		}
		return {E, new_shape};
	}
};


template<class Tensor,class OAxis, class DropList, std::size_t N >
struct broadcast_optimizer<tensor_broadcast<Tensor, OAxis, DropList >, N >{
	typedef typename OAxis::template split_t<(N == OAxis::num_dims)? N - 1: N> Axis;
	typedef tensor_broadcast<Tensor, Axis, typename DropList::template insert_t<N, true> > type;
	
	static type create(tensor_broadcast<Tensor, OAxis, DropList > const& E, std::size_t size){
		auto shape = E.shape();
		tensor_shape<Axis::num_dims> new_shape;
		for(std::size_t i = 0; i != N; ++i){
			new_shape[i] = shape[i];
		}
		new_shape[N] = size;
		for(std::size_t i = N; i != Axis::num_dims; ++i){
			new_shape[i + 1] = shape[i];
		}
		return {E.expression(), new_shape};
	}
};



////////////////////////////////////
//// Matrix-Diagonal
////////////////////////////////////
/*
//range(alpha * Tensor) = alpha * diag(Tensor)
template<class Tensor>
struct matrix_diagonal_optimizer<scalar_multiply<Tensor> >{
	typedef matrix_diagonal_optimizer<typename Tensor::const_closure_type > opt;
	typedef scalar_multiply<typename opt::type > type;
	
	static type create(scalar_multiply<Tensor> const& E){
		return type(opt::create(E.expression()), E.scalar());
	}
};

//diag(TensorA+TensorB) = diag(TensorA) + diag(TensorB)
template<class TensorA, class TensorB>
struct matrix_diagonal_optimizer<tensor_addition<TensorA,TensorB> >{
	typedef matrix_diagonal_optimizer<typename TensorA::const_closure_type > left_opt;
	typedef matrix_diagonal_optimizer<typename TensorB::const_closure_type > right_opt;
	typedef tensor_addition<typename left_opt::type, typename right_opt::type > type;
	
	static type create(tensor_addition<TensorA,TensorB> const& E){
		return type(left_opt::create(E.lhs()),right_opt::create(E.rhs()));
	}
};

//diag(constant)  -> constant (vector)
template<class T, class Device, class OAxis>
struct matrix_diagonal_optimizer<scalar_tensor<T,Device, OAxis> >{
	typedef scalar_tensor<T,Device> type;
	
	static type create(scalar_tensor<T,Device, OAxis> const& E){
		return type(E().size(), E.scalar());
	}
};

//diag(repeat(v,j)) -> range(v,0,min(v.size,j))
template<class V, class OAxis>
struct matrix_diagonal_optimizer<vector_repeater<V, OAxis> >{
	typedef vector_range_optimizer<typename V::const_closure_type > opt;
	typedef typename opt::type type;
	
	static type create(vector_repeater<V, OAxis> const& E){
		return opt::create(E.expression(),0, std::min(E.size1(),E.size2())); 

	}
};

// diag(f(Tensor)) -> f(diag(Tensor))
template<class Tensor, class F>
struct matrix_diagonal_optimizer<tensor_unary<Tensor, F> >{
	typedef matrix_diagonal_optimizer<typename Tensor::const_closure_type > opt;
	typedef tensor_unary<typename opt::type, F > type;
	
	static type create(tensor_unary<Tensor, F> const& E){
		return type(opt::create(E.expression()), E.functor());
	}
};
// diag(f(Tensor,TensorB)) -> f(diag(TensorA),diag(TensorB))
template<class TensorA, class TensorB, class F>
struct matrix_diagonal_optimizer<tensor_binary<TensorA,TensorB, F> >{
	typedef matrix_diagonal_optimizer<typename TensorA::const_closure_type > left_opt;
	typedef matrix_diagonal_optimizer<typename TensorB::const_closure_type > right_opt;
	typedef tensor_binary<typename left_opt::type, typename right_opt::type, F > type;
	
	static type create(tensor_binary<TensorA,TensorB,F> const& E){
		return type(left_opt::create(E.lhs()),right_opt::create(E.rhs()),E.functor());
	}
};

//diag( u v^T) -> range(u,size) range(v,size)^T, where size=min(u.size,v.size)
template<class V1, class V2>
struct matrix_diagonal_optimizer<outer_product<V1,V2> >{
	typedef vector_range_optimizer<typename V1::const_closure_type > left_opt;
	typedef vector_range_optimizer<typename V2::const_closure_type> right_opt;
	typedef typename common_value_type<V1,V2>::type value_type;
	typedef typename device_traits<typename V1::device_type>:: template multiply<value_type> functor;
	typedef tensor_binary<typename left_opt::type, typename right_opt::type, functor> type;
	
	static type create(outer_product<V1,V2> const& E){
		auto size = std::min(E.size1(),E.size2());
		return type( left_opt::create(E.lhs(),0,size), right_opt::create(E.rhs(),0,size), functor());
	}
};

//diag(diagonal(v))  -> v
template<class V>
struct matrix_diagonal_optimizer<diagonal_matrix<V> >{
	typedef typename V::const_closure_type type;
	static type create(diagonal_matrix<V> const& E){
		return E.expression();
	}
};*/

////////////////////////////////////
//// Tensor - Scalar Product
////////////////////////////////////


//default impl for alpha * A, creates just the expression
// handles all Tensor that can not be blockwise, e.g. : all containers, proxies, scalar_tensor
template<class Tensor>
struct scalar_multiply_optimizer{
	typedef scalar_multiply<Tensor> type;
	
	static type create(typename Tensor::const_closure_type const& E, typename Tensor::value_type alpha){
		return type(E, alpha);
	}
};


// alpha * (beta * A) = (alpha * beta) * A
template<class Tensor>
struct scalar_multiply_optimizer<scalar_multiply<Tensor> >{
	typedef scalar_multiply<Tensor> type;
	
	static type create(scalar_multiply<Tensor> const& E, typename Tensor::value_type alpha){
		return type(E.expression(), alpha * E.scalar());
	}
};


// alpha * (A + B) = alpha * A + alpha * B
template<class E1, class E2>
struct scalar_multiply_optimizer<tensor_addition<E1, E2> >{
	typedef typename tensor_addition<E1, E2>::value_type value_type;
	typedef scalar_multiply_optimizer<E1> opt1;
	typedef scalar_multiply_optimizer<E2> opt2;
	typedef tensor_addition<typename opt1::type, typename opt2::type> type;
	static type create(tensor_addition<E1, E2> const& E, value_type alpha){
		return type(opt1::create(E.lhs(), alpha), opt2::create(E.rhs(), alpha));
	}
};

// alpha * f(A) = (alpha * f)(A)
template<class Tensor, class F>
struct scalar_multiply_optimizer<tensor_unary<Tensor, F> >{
	typedef typename Tensor::device_type device_type;
	typedef typename F::result_type value_type;
	typedef typename device_traits<device_type>::template multiply_scalar<value_type> Multiplier;
	typedef tensor_unary_optimizer <tensor_unary<Tensor, F>, Multiplier > opt;
	typedef typename opt::type type;
	static type create(tensor_unary<Tensor, F> const& E, value_type alpha){
		return opt::create(E, Multiplier(alpha));
	}
};

// alpha * f(A, B) = (alpha * f)(A, B)
template<class TensorA, class TensorB, class F>
struct scalar_multiply_optimizer<tensor_binary<TensorA, TensorB, F> >{
	typedef typename TensorA::device_type device_type;
	typedef typename F::result_type value_type;
	typedef typename device_traits<device_type>::template multiply_scalar<value_type> Multiplier;
	typedef tensor_unary_optimizer <tensor_binary<TensorA, TensorB, F>, Multiplier > opt;
	typedef typename opt::type type;
	static type create(tensor_binary<TensorA, TensorB, F> const& E, value_type alpha){
		return opt::create(E, Multiplier(alpha));
	}
};


////////////////////////////////////
//// Matrix Unary
////////////////////////////////////

template<class Tensor, class F>
struct tensor_unary_optimizer{
	typedef tensor_unary<Tensor,F> type;
	
	static type create(typename Tensor::const_closure_type const& E, F const& f){
		return type(E,f);
	}
};

//f(g(x)) = (f o g)(x)
template<class Tensor, class F1, class F2>
struct tensor_unary_optimizer<tensor_unary<Tensor,F1>, F2 >{
	typedef typename device_traits<typename Tensor::device_type>::template compose<F1, F2> composed_type;
	typedef tensor_unary<Tensor,composed_type> type;
	
	static type create(tensor_unary<Tensor,F1> const& E, F2 const& f){
		return type(E.expression(),composed_type(E.functor(),f));
	}
};

//f(g(x,y)) = (f o g)(x,y)
template<class TensorA, class TensorB, class F1, class F2>
struct tensor_unary_optimizer<tensor_binary<TensorA,TensorB, F1>, F2 >{
	typedef typename device_traits<typename TensorA::device_type>::template compose<F1, F2> composed_type;
	typedef tensor_binary<TensorA, TensorB,composed_type> type;
	
	static type create(tensor_binary<TensorA, TensorB, F1> const& E, F2 const& f){
		return type(E.lhs(), E.rhs(), composed_type(E.functor(),f));
	}
};

/*
// alpha * repeat(v,n) = repeat(alpha * v, n)
template<class V, class O>
struct scalar_multiply_optimizer<vector_repeater<V, O> >{
	typedef vector_scalar_multiply_optimizer<V> opt;
	typedef vector_repeater<typename opt::type, O> type;
	static type create(vector_repeater<V, O> const& E, typename V::value_type alpha){
		return type(opt::create(E.expression(), alpha), E.num_repetitions());
	}
};

//alpha * v * u^T = (alpha * v) * u^T 
template<class V1, class V2>
struct scalar_multiply_optimizer<outer_product<V1, V2> >{
	typedef vector_scalar_multiply_optimizer<V1> opt;
	typedef outer_product<typename opt::type, V2> type;
	typedef typename type::value_type value_type;
	static type create(outer_product<V1, V2> const& E, value_type alpha){
		return type(opt::create(E.lhs(), alpha),E.rhs());
	}
};

//alpha * (A * B) can be folded into matrix_vector_prod 
template<class TensorA, class TensorB>
struct scalar_multiply_optimizer<matrix_matrix_prod<TensorA, TensorB> >{
	typedef matrix_matrix_prod<TensorA, TensorB> type;
	typedef typename type::value_type value_type;
	static type create(matrix_matrix_prod<TensorA, TensorB> const& E, value_type alpha){
		return type(E.lhs(), E.rhs(), alpha * E.alpha());
	}
};

// alpha*(A | B) = (alpha * A) | (alpha * B) 
template<class TensorA, class TensorB, bool b>
struct scalar_multiply_optimizer<matrix_concat<TensorA, TensorB, b> >{
	typedef scalar_multiply_optimizer<TensorA> opt1;
	typedef scalar_multiply_optimizer<TensorB> opt2;
	typedef matrix_concat<typename opt1::type, typename opt2::type, b> type;
	typedef typename type::value_type value_type;
	static type create(matrix_concat<TensorA, TensorB, b> const& E, value_type alpha){
		return type(opt1::create(E.lhs(), alpha), opt2::create(E.rhs(), alpha));
	}
};*/

/*
////////////////////////////////////
//// Matrix Vector Product
////////////////////////////////////
	
//matrix-vector multiplications
template<class Tensor, class V>
struct matrix_vector_prod_optimizer{
	typedef matrix_vector_prod<Tensor,V> type;
	
	static type create(typename Tensor::const_closure_type const& E, typename V::const_closure_type const& v){
		return type(E, v, typename type::value_type(1));
	}
};

//(alpha Tensor)*v = alpha * (Tensor * v)
template<class Tensor, class V>
struct matrix_vector_prod_optimizer<scalar_multiply<Tensor>,V >{
	typedef matrix_vector_prod_optimizer<Tensor, V> inner_opt;
	typedef vector_scalar_multiply_optimizer<typename inner_opt::type> opt;
	typedef typename opt::type type;
	
	static type create(scalar_multiply<Tensor> const& E, typename V::const_closure_type const& v){
		return opt::create(inner_opt::create(E.expression(), v), E.scalar());
	}
};

//Tensor*(alpha*v) = alpha * (Tensor * v)
template<class Tensor, class V>
struct matrix_vector_prod_optimizer<Tensor,scalar_multiply<V> >{
	typedef matrix_vector_prod_optimizer<Tensor, V> inner_opt;
	typedef vector_scalar_multiply_optimizer<typename inner_opt::type> opt;
	typedef typename opt::type type;
	
	static type create(typename Tensor::const_closure_type const& E, scalar_multiply<V> const& v){
		return opt::create(inner_opt::create(E, v.expression()), v.scalar());
	}
};

//(alpha Tensor)*(beta*v) = (alpha*beta) (Tensor*v) can be folded into matrix-vector product
template<class Tensor, class V>
struct matrix_vector_prod_optimizer<scalar_multiply<Tensor>,scalar_multiply<V> >{
	typedef matrix_vector_prod_optimizer<Tensor, V> inner_opt;
	typedef vector_scalar_multiply_optimizer<typename inner_opt::type> opt;
	typedef typename opt::type type;
	
	static type create(scalar_multiply<Tensor> const& E, scalar_multiply<V>const& v){
		return opt::create(inner_opt::create(E.expression(), v.expression()), v.scalar() * E.scalar());
	}
};

//(TensorA*TensorB)*V=TensorA*(TensorB*V)
template<class TensorA,class TensorB, class V>
struct matrix_vector_prod_optimizer<matrix_matrix_prod<TensorA,TensorB>,V>{
private:
	typedef matrix_vector_prod_optimizer<TensorB,V> inner_opt;
	typedef matrix_vector_prod_optimizer<TensorA, typename inner_opt::type> outer_opt;
	typedef vector_scalar_multiply_optimizer<typename outer_opt::type> scalar_opt;
public:
	typedef typename scalar_opt::type type;
	
	static type create(matrix_matrix_prod<TensorA,TensorB> const& E, typename V::const_closure_type const& v){
		auto inner_result = inner_opt::create(E.rhs(),v);
		auto outer_result = outer_opt::create(E.lhs(),inner_result);
		return scalar_opt::create(outer_result, E.alpha());
	}
};

//(TensorA+TensorB)*V=TensorA*V+TensorB*V
template<class TensorA,class TensorB, class V>
struct matrix_vector_prod_optimizer<tensor_addition<TensorA,TensorB>,V>{
private:
	typedef matrix_vector_prod_optimizer<TensorA,V> left_opt;
	typedef matrix_vector_prod_optimizer<TensorB,V> right_opt;
public:
	typedef tensor_addition<typename left_opt::type ,typename right_opt::type> type;
	
	static type create(tensor_addition<TensorA,TensorB> const& E, typename V::const_closure_type const& v){
		auto lhs = left_opt::create(E.lhs(),v);
		auto rhs = right_opt::create(E.rhs(),v);
		return type(lhs,rhs);
	}
};

//(v1*v2^T)*v3= v1*(v2^T*v3)
template<class V1,class V2, class V3>
struct matrix_vector_prod_optimizer<outer_product<V1,V2>,V3>{
	typedef scalar_multiply<V1> type;
	
	static type create(outer_product<V1,V2> const& E, typename V3::const_closure_type const& v){
		auto alpha = inner_prod(E.rhs(),v);
		return type(E.lhs(), alpha);
	}
};

template<class V1, class V2>
struct matrix_vector_prod_optimizer<vector_repeater<V1, row_major>, V2 >{
	typedef scalar_tensor<typename common_value_type<V1,V2>::type, typename V1::device_type> type;
	
	static type create(vector_repeater<V1, row_major> const& E, typename V2::const_closure_type const& v){
		auto alpha = inner_prod(E.expression(),v);
		return type(alpha,E.num_repetitions());
	}
};

template<class V1, class V2>
struct matrix_vector_prod_optimizer<vector_repeater<V1, column_major>, V2 >{
	typedef scalar_multiply<V1> type;
	
	static type create(vector_repeater<V1, row_major> const& E, typename V2::const_closure_type const& v){
		auto alpha = sum(E.expression(),v);
		return type(E.expression(),alpha);
	}
};

//diag(v1) * v2 = v1 .* v2
template<class V1,class V2>
struct matrix_vector_prod_optimizer<diagonal_matrix<V1>,V2>{
	typedef typename common_value_type<V1,V2>::type value_type;
	typedef typename device_traits<typename V1::device_type>:: template multiply<value_type> functor;
	typedef tensor_binary<V1, V2, functor> type;
	static type create(diagonal_matrix<V1> const& E, typename V2::const_closure_type const& v){
		return type(E.expression(),v, functor());
	}
};


////////////////////////////////////
//// Matrix Product
////////////////////////////////////

template<class TensorA, class TensorB>
struct matrix_matrix_prod_optimizer{
	typedef matrix_matrix_prod<TensorA,TensorB> type;
	
	static type create(typename TensorA::const_closure_type const& lhs, typename TensorB::const_closure_type const& rhs){
		return type(lhs, rhs, typename type::value_type(1));
	}
};


//(alpha TensorA)*B = alpha (TensorA*B)
template<class TensorA, class TensorB>
struct matrix_matrix_prod_optimizer<scalar_multiply<TensorA>,TensorB >{
	typedef matrix_matrix_prod_optimizer<TensorA, TensorB> inner_opt;
	typedef scalar_multiply<typename inner_opt::type> opt;
	typedef typename opt::type type;
	
	static type create(scalar_multiply<TensorA> const& A, typename TensorB::const_closure_type const& B){
		return opt::create(inner_opt::create(A.expression(), B), A.scalar());
	}
};

//TensorA*(alpha*B) = alpha (TensorA*B)
template<class TensorA, class TensorB>
struct matrix_matrix_prod_optimizer<TensorA,scalar_multiply<TensorB> >{
	typedef matrix_matrix_prod_optimizer<TensorA, TensorB> inner_opt;
	typedef scalar_multiply<typename inner_opt::type> opt;
	typedef typename opt::type type;
	
	static type create(typename TensorA::const_closure_type const& A, scalar_multiply<TensorB> const& B){
		return opt::create(inner_opt::create(A, B.expression()), B.scalar());
	}
};

//(alpha TensorA)*(beta*B) = (alpha*beta) (TensorA*B)
template<class TensorA, class TensorB>
struct matrix_matrix_prod_optimizer<scalar_multiply<TensorA>,scalar_multiply<TensorB> >{
	typedef matrix_matrix_prod_optimizer<TensorA, TensorB> inner_opt;
	typedef scalar_multiply<typename inner_opt::type> opt;
	typedef typename opt::type type;
	
	static type create(scalar_multiply<TensorA> const& A, scalar_multiply<TensorB>const& B){
		return opt::create(inner_opt::create(A.expression(), B.expression()), A.scalar() * B.scalar());
	}
};*/

/*
////////////////////////////////////
//// Vector-Set Fold
////////////////////////////////////

template<class S, class F, class G>
struct fold_vector_set_optimizer;

template<class Tensor, class F, class G>
struct fold_vector_set_optimizer<vector_set<Tensor, row_major>, F, G>{
	typedef matrix_row_transform<Tensor, F, G> type;
	static type create(vector_set<Tensor, row_major> const& set, F const& f, G const& g){
		return type(set.expression(), f, g);
	}
};

template<class Tensor, class F, class G>
struct fold_vector_set_optimizer<vector_set<Tensor, column_major>, F, G>{
	typedef axis_permute_optimizer<Tensor> opt;
	typedef matrix_row_transform<typename opt::type, F, G> type;
	static type create(vector_set<Tensor, column_major> const& set, F const& f, G const& g){
		return type(opt::create(set.expression()), f, g);
	}
};

//~ template<class S, class Tensor>
//~ struct vector_set_matrix_prod_optimizer;

//~ template<class TensorA, class TensorB>
//~ struct vector_set_matrix_prod_optimizer<vector_set<TensorA, row_major>, TensorB>{
	//~ typedef matrix_matrix_prod_optimizer<TensorA, TensorB> opt;
	//~ typedef vector_set<typename opt::type, row_major> type;
	//~ static type create(vector_set<TensorA, row_major> const& set, typename TensorB::const_closure_type const& m2){
		//~ return as_set(opt::create(set.expression(), m2), row_major());
	//~ }
//~ };

//~ template<class TensorA, class TensorB>
//~ struct vector_set_matrix_prod_optimizer<vector_set<TensorA, column_major>, TensorB>{
	//~ typedef axis_permute_optimizer<TensorB> trans_opt;
	//~ typedef matrix_matrix_prod_optimizer<typename trans_opt::type, TensorA> opt;
	//~ typedef vector_set<typename opt::type, column_major> type;
	//~ static type create(vector_set<TensorA, column_major> const& set, typename TensorB::const_closure_type const& m2){
		//~ return as_set(opt::create(trans_opt::create(m2),set.expression()), column_major());
	//~ }
//~ };

//~ template<class S, class V>
//~ struct vector_set_inner_prod_optimizer;

//~ template<class Tensor, class V>
//~ struct vector_set_inner_prod_optimizer<vector_set<Tensor, row_major>, V>{
	//~ typedef matrix_vector_prod_optimizer<Tensor, V> opt;
	//~ typedef typename opt::type type;
	//~ static type create(vector_set<Tensor, row_major> const& set, typename V::const_closure_type const& v){
		//~ return opt::create(set.expression(), v);
	//~ }
//~ };
//~ template<class Tensor, class V>
//~ struct vector_set_inner_prod_optimizer<vector_set<Tensor, column_major>, V>{
	//~ typedef axis_permute_optimizer<Tensor> trans_opt;
	//~ typedef matrix_vector_prod_optimizer<typename trans_opt::type, V> opt;
	//~ typedef typename opt::type type;
	//~ static type create(vector_set<Tensor, column_major> const& set, typename V::const_closure_type const& v){
		//~ return opt::create(trans_opt::create(set.expression()), v);
	//~ }
//~ };

*/
}}
#endif
