/*!
 * \brief       Implements the Dense storage vector and matrices
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
#ifndef REMORA_DENSE_HPP
#define REMORA_DENSE_HPP

#include "expression_types.hpp"
#include "detail/traits.hpp"
#include "detail/proxy_optimizers_fwd.hpp"
#include "detail/merge_proxy.hpp"
namespace remora{
	
	
/** \brief A dense tensor of values of type \c T.
 *
 * A tensor with shape s of dimension N represents an n-dimensional array with elements of type T where dimension i has size s[i].
 * The whole tensor thus has s[0]*s[1]*...*s[N-1] elements. A 1-d tensor is equivalent to a vector and a 2d tensor
 * is equivalent to a matrix.
 * In a dense tensor all elements are represented as a contiguous chunk of memory by definition.
 * Axis is a permutation of the numbers 0,1,N-1. Each number must occur once and the amount of entries
 * define the dimensionality of the Tensor.
 * The permutation defines the axis of indices when accessing elements in memory.
 * E.g. axis<0,1> defines a matrix with standard c-layour, also called row-major. 
 * axis<1,0> defines column-major layout (compatible to fortran).
 * axis<2,3,0,1> defines a 4-dimensional tensor where the indices (i,j,k,l) that define
 * an entry are mapped to the internal data storage as data[k][l][i][j].
 *
 * \tparam T the type of object stored in the tensor (like double, float, complex, etc...)
 * \tparam Axis the storage organization.
 * \tparam Device the device the tensor is located at
 */
template<class T, class Axis, class Device>
class tensor;


/** \brief A dense tensor of values of type \c T.
 *
 * A tensor with shape s of dimension N represents an n-dimensional array with elements of type T where dimension i has size s[i].
 * The whole tensor thus has s[0]*s[1]*...*s[N-1] elements. A 1-d tensor is equivalent to a vector and a 2d tensor
 * is equivalent to a matrix.
 * In a dense tensor all elements are represented as a contiguous chunk of memory by definition.
 * Axis is a permutation of the numbers 0,1,N-1. Each number must occur once and the amount of entries
 * define the dimensionality of the Tensor.
 * The permutation defines the axis of indices when accessing elements in memory.
 * E.g. axis<0,1> defines a matrix with standard c-layour, also called row-major. 
 * axis<1,0> defines column-major layout (compatible to fortran).
 * axis<2,3,0,1> defines a 4-dimensional tensor where the indices (i,j,k,l) that define
 * an entry are mapped to the internal data storage as data[k][l][i][j].
 *
 * \tparam T the type of object stored in the tensor (like double, float, complex, etc...)
 * \tparam Axis the storage organization.
 * \tparam Device the device the tensor is located at
 */
template<class T, unsigned N, class Device = cpu_tag>
using tensorN = tensor<T, default_axis<N>, Device>;
	
/// \brief A dense vector of values of type \c T.
///
/// For a \f$n\f$-dimensional vector \f$v\f$ and \f$0\leq i < n\f$ every element \f$v_i\f$ is mapped
/// to the \f$i\f$-th element of the container.
/// The tag descripes on which device the vector is located
///
/// \tparam T the type of object stored in the vector (like double, float, complex, etc...)
/// \tparam Device the device this vector lives on, the default is cpu_tag for a cpu vector
template<class T, class Device = cpu_tag>
using vector = tensorN<T, 1, Device>;


/// \brief A dense matrix of values of type \c T.
///
/// For a \f$(E \times n)\f$-dimensional matrix and \f$ 0 \leq i < E, 0 \leq j < n\f$, every element \f$ m_{i,j} \f$ is mapped to
/// the \f$(i*n + j)\f$-th element of the container for row major orientation or the \f$ (i + j*E) \f$-th element of
/// the container for column major orientation. In a dense matrix all elements are represented in memory in a
/// contiguous chunk of memory by definition.
///
/// Orientation can also be specified, otherwise a \c row_major is used.
///
/// \tparam T the type of object stored in the matrix (like double, float, complex, etc...)
/// \tparam Orientation the storage organization. It can be either \c row_major or \c column_major. Default is \c row_major
/// \tparam Device the device this matrix lives on, the default is cpu_tag for a cpu matrix
template<class T, class Orientation=row_major, class Device = cpu_tag>
using matrix = tensor<T,Orientation, Device>;



/// \brief A proxy to a  dense tensor of values of type \c T.
///
/// Using external memory provided by another tensor, references a subrange of the tensor.
/// The referenced region is not required to be consecutive, i.e. a subregion of a tensor can be used.
///
/// \tparam T the type of object stored in the tensor (like double, float, complex, etc...)
/// \tparam Orientation the storage organization. It can be either \c row_major or \c column_major. Default is \c row_major
/// \tparam Tag the storage tag. dense_tag by default and continuous_dense_tag if the memory region referenced is continuous.
/// \tparam Device the device this vector lives on, the default is cpu_tag for a cpu vector
template<class T,class Axis, class Tag = dense_tag, class Device = cpu_tag>
class dense_tensor_adaptor;


// template<class T, class Orientation, class TriangularType, class Device>
// class dense_triangular_proxy;

///////////////////////////////////
// Adapt memory as vector
///////////////////////////////////

/// \brief Converts a chunk of memory into a vector of a given size.
// template <class T>
// dense_tensor_adaptor<T, axis<0>, continuous_dense_tag, cpu_tag> adapt_vector(std::size_t size, T * v){
	// return dense_tensor_adaptor<T, axis<0>, continuous_dense_tag, cpu_tag>(v,{size});
// }

// template <class T>
// dense_tensor_adaptor<T, axis<0>, dense_tag, cpu_tag> adapt_vector(std::size_t size, T * v, std::size_t stride){
	// return dense_tensor_adaptor<T, axis<0>, dense_tag, cpu_tag>(v,{size}, {stride});
// }

/// \brief Converts a C-style array into a vector.
// template <class T, std::size_t N>
// dense_tensor_adaptor<T, axis<0>, continuous_dense_tag, cpu_tag> adapt_vector(T (&array)[N]){
	// return dense_tensor_adaptor<T, axis<0>, continuous_dense_tag, cpu_tag>(array,{N}, {1});
// }

/// \brief Converts a chunk of memory into a matrix of given size and row-major storage layout
// template <class T>
// dense_tensor_adaptor<T, row_major, continuous_dense_tag, cpu_tag> adapt_matrix(std::size_t size1, std::size_t size2, T* data){
	// return dense_tensor_adaptor<T, row_major, continuous_dense_tag, cpu_tag>(data,{size1, size2});
// }

/// \brief Converts a 2D C-style array into a matrix with row-major storage layout
// template <class T, std::size_t M, std::size_t N>
// dense_tensor_adaptor<T, row_major, continuous_dense_tag, cpu_tag> adapt_matrix(T (&array)[M][N]){
	// return dense_tensor_adaptor<T, row_major, continuous_dense_tag, cpu_tag>(&(array[0][0]),{M,N});
// }

///////////////////////////////////
// Traits
///////////////////////////////////

template<class T, class Axis, class Device>
struct tensor_temporary_type<T, Axis, dense_tag, Device>{
	typedef tensor<T, Axis, Device> type;
};

//////////////////////////////////
//////Expression Traits
///////////////////////////////////

namespace detail{
	
	
///////////////////////////////////////////////////
//////Traits For Proxy Expressions
///////////////////////////////////////////////////


////////////////////////TENSOR SUBRANGE//////////////////////
template<class T, class Axis, class TagList, class Device, std::size_t N>
struct subrange_optimizer<dense_tensor_adaptor<T, Axis, TagList, Device>, N >{
	//slicing an axis will cause the next axis in memory to not be dense any more
	struct storage_tag_transform{
		template<class Seq>
		static constexpr Seq apply(Seq seq){
			auto axes = Axis::to_array();
			constexpr unsigned next_ax = Axis::template element_v<N> - 1;
			//fint the position of the next axis in memory and set it to not-dense
			for(std::size_t i = 0; i != Axis::num_dims; ++i){
				if( axes[i] == next_ax){
					seq.values[i] = 0;
					break;
				}
			}
			return seq;
		}
	};
	typedef typename TagList::template transform_t<storage_tag_transform> proxy_tag_list;
	typedef dense_tensor_adaptor<T, Axis, proxy_tag_list, Device> type;
	
	static type create(
		dense_tensor_adaptor<T, Axis, TagList, Device> const& E,
		std::size_t start, std::size_t end
	){
		REMORA_SIZE_CHECK(start < E.shape()[N]);
		REMORA_SIZE_CHECK(end < E.shape()[N]);
		//compute new shape
		auto new_shape = E.shape();
		new_shape[N] = end - start;
		//get the proper point in storage
		auto storage = E.raw_storage();
		T* new_values = storage.values + storage.strides[N] * start;
		//return the proxy
		return type({new_values, storage.strides}, E.queue(), new_shape);
	}
};


////////////////////////TENSOR SLICE//////////////////////

template<class T, class Axis, class TagList, class Device, std::size_t N>
struct slice_optimizer<dense_tensor_adaptor<T, Axis, TagList, Device>, N>{
	struct storage_tag_transform{
		template<class Seq>
		static constexpr Seq apply(Seq seq){
			auto axes = Axis::to_array();
			//update the next axis as it is not continuous anymore
			//underflow of -1 is defined -> changes in the major axis don't affect any other axis
			constexpr unsigned next_ax = unsigned(Axis::template element_v<N>) - 1;
			for(std::size_t i = 0; i != Axis::num_dims; ++i){
				if( axes[i] == next_ax){
					seq.values[i] = 0;
				}
			}
			//remove the nth axis
			for(std::size_t i = N; i < Axis::num_dims - 1; ++i){
				seq.values[i] = seq.values[i + 1];
			}
			//this will still give N elements, so after transform we have to cut off the last element
			return seq;
		}
	};
	typedef typename TagList::template transform_t<storage_tag_transform>::template front_t<Axis::num_dims - 1> proxy_tag_list;
	typedef dense_tensor_adaptor<T, typename Axis::template slice_t<N>, proxy_tag_list, Device> type;
	
	static type create(
		dense_tensor_adaptor<T, Axis, TagList, Device> const& E,
		std::size_t index
	){
		REMORA_SIZE_CHECK(index < E.shape()[N]);
		//compute new shape by cutting out the selected Axis
		auto strides = E.raw_storage().strides;
		auto shape = E.shape();
		std::array<std::size_t, Axis::num_dims-1> new_strides;
		for(unsigned i = 0; i != N; ++i){
			new_strides[i] = strides[i];
		}
		for(unsigned i = N + 1; i != Axis::num_dims; ++i){
			new_strides[i - 1] = strides[i];
		}
		T* values = E.raw_storage().values + index * strides[N];
		//return the proxy
		return type({values, new_strides}, E.queue(), shape.slice(N));
	}
};

//specialization for the (rare?) case that we split off the last dimension
// note: this might have to be device-dependent! Treat it as a stub!
template<class T, class Tag, class Device, std::size_t N>
struct slice_optimizer<dense_tensor_adaptor<T, axis<0>, Tag, Device>, N>{
	typedef decltype(std::declval<dense_tensor_adaptor<T, axis<0>, Tag, Device> >()(0)) type;
	
	static type create(
		dense_tensor_adaptor<T, axis<0>, Tag, Device> const& E,
		std::size_t index
	){
		return E(index);
	}
};

////////////////////////TENSOR AXIS SPLIT//////////////////////
template<class T, class Axis, class TagList, class Device, std::size_t N>
struct axis_split_optimizer<dense_tensor_adaptor<T, Axis, TagList, Device>, N>{
	//the newly created axis is always dense
	typedef typename TagList::template insert_t<N, 1u > proxy_tag_list;
	typedef dense_tensor_adaptor<T, typename Axis::template split_t<N>, proxy_tag_list, Device> type;
	
	static type create(
		dense_tensor_adaptor<T, Axis, TagList, Device> const& E,
		std::size_t size1, std::size_t size2
	){
		//compute new shape by cutting out the selected Axis
		auto strides = E.raw_storage().strides;
		std::array<std::size_t, Axis::num_dims+1> new_strides;
		for(unsigned i = 0; i != N; ++i){
			new_strides[i] = strides[i];
		}
		new_strides[N] = strides[N]* size2;
		new_strides[N + 1] = strides[N];
		for(unsigned i = N + 1; i != Axis::num_dims; ++i){
			new_strides[i + 1] = strides[i];
		}
		//return the proxy
		return type({E.raw_storage().values, new_strides}, E.queue(), E.shape().split(N, size1, size2));
	}
};

////////////////////////TENSOR AXIS MERGE//////////////////////
template<class T, class Axis, class TagList, class Device, std::size_t N>
struct axis_merge_optimizer<dense_tensor_adaptor<T, Axis, TagList, Device>, N>{
	static_assert(N < Axis::num_dims - 1);
	//check whether we need a proxy or not.
	static constexpr std::size_t needs_proxy = !TagList::template element_v<N>
		||(Axis::template element_v<N> + 1 != Axis::template element_v<N + 1>);
	//case 1: needs_proxy = false
	//the new axis has the same tag as the second of the two merged axis. 
	//so we only have to remove the Nth tag
	typedef typename TagList::template remove_t<N> proxy_tag_list;
	typedef dense_tensor_adaptor<T, typename Axis::template slice_t<N>, proxy_tag_list, Device> no_proxy_type;
	
	static no_proxy_type create(dense_tensor_adaptor<T, Axis, TagList, Device> const& E, integer_list<std::size_t, 0>){
		//compute new shape by cutting out the selected Axis
		auto strides = E.raw_storage().strides;
		auto shape = E.shape();
		REMORA_SIZE_CHECK(strides[N] == strides[N + 1] * shape[N+1]);
		
		std::array<std::size_t, Axis::num_dims-1> new_strides;
		for(unsigned i = 0; i != N; ++i){
			new_strides[i] = strides[i];
		}
		new_strides[N] = strides[N + 1];
		for(unsigned i = N + 2; i != Axis::num_dims; ++i){
			new_strides[i - 1] = strides[i];
		}
		//return the proxy
		return {{E.raw_storage().values, new_strides}, E.queue(), shape.merge(N)};
	}
	
	//case 2: we need a proxy :-(
	typedef merge_proxy<dense_tensor_adaptor<T, Axis, TagList, Device>, N> proxy_type;
	static proxy_type create(dense_tensor_adaptor<T, Axis, TagList, Device> const& E, integer_list<std::size_t, 1>){
		return E;
	}
	//dispatcher
	typedef typename std::conditional<needs_proxy, proxy_type, no_proxy_type>::type type;
	static type create(dense_tensor_adaptor<T, Axis, TagList, Device> const& E){
		return create(E, integer_list<std::size_t, needs_proxy>());
	}
};

////////////////////////TENSOR PERMUTE//////////////////////
template<class T, class Axis, class TagList, class Device, unsigned... Permutation>
struct axis_permute_optimizer<dense_tensor_adaptor<T,Axis, TagList, Device>, axis<Permutation...> >{
	typedef typename TagList::template select_t<Permutation...> proxy_tag_list;
	typedef dense_tensor_adaptor<T, typename Axis:: template permute_t<Permutation...>, proxy_tag_list, Device> type;
	
	static type create(dense_tensor_adaptor<T,Axis, TagList, Device> const& E){
		auto storage = E.raw_storage();
		auto permuted_strides = axis<Permutation...>::to_axis(storage.strides);
		auto shape = axis<Permutation...>::to_axis(E.shape());
		return type({storage.values, permuted_strides}, E.queue(), shape);
	}
};

// template<class T, class Axis, class Triangular, class Device>
// struct axis_permute_optimizer<dense_triangular_proxy<T, Axis, Triangular, Device> >{
	// typedef dense_triangular_proxy<T, typename Axis::transposed_orientation, Triangular, Device> type;
	
	// static type create(dense_triangular_proxy<T, Axis, Triangular, Device> const& E){
		// return type(E.raw_storage(), E.queue(), E.size2(), E.size1());
	// }
// };


////////////////////////MATRIX DIAGONAL//////////////////////
// template<class T, class Orientation, class Device, class Device>
// struct matrix_diagonal_optimizer<dense_matrix_adaptor<T,Orientation, Device, Device> >{
	// typedef dense_vector_adaptor<T, dense_tag, Device> type;
	
	// static type create(dense_matrix_adaptor<T,Orientation, Device, Device> const& E){
		// return type(E.raw_storage().diag(), E.queue(), std::min(E.size1(), E.size2()));
	// }
// };


////////////////////////TO TRIANGULAR//////////////////////

// template<class T, class Orientation, class Device, class Device, class Triangular>
// struct triangular_proxy_optimizer<dense_matrix_adaptor<T,Orientation, Device, Device>, Triangular >{
	// typedef dense_triangular_proxy<T, Orientation, Triangular, Device> type;
	
	// static type create(dense_matrix_adaptor<T,Orientation, Device, Device> const& E){
		// return type(E.raw_storage(), E.queue(), E.size1(), E.size2());
	// }
// };


}

}

//include device dependent implementations
#include "cpu/dense.hpp"
// #if defined(__HCC__) || defined(__NVCC__)
// #include "hip/dense.hpp"
// #endif
#endif
