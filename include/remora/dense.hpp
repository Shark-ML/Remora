/*!
 * \brief       Implements the Dense storage vector and matrices
 * 
 * \author      O. Krause
 * \date        2014
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
namespace remora{
	
/// \brief A dense vector of values of type \c T.
///
/// For a \f$n\f$-dimensional vector \f$v\f$ and \f$0\leq i < n\f$ every element \f$v_i\f$ is mapped
/// to the \f$i\f$-th element of the container.
/// The tag descripes whether the vector is residing on a cpu or gpu which change its semantics.
///
/// \tparam T the type of object stored in the matrix (like double, float, complex, etc...)
/// \tparam Device the device this vector lives on, the default is cpu_tag for a cpu vector
template<class T, class Device = cpu_tag>
class vector;


/// \brief A dense matrix of values of type \c T.
///
/// For a \f$(m \times n)\f$-dimensional matrix and \f$ 0 \leq i < m, 0 \leq j < n\f$, every element \f$ m_{i,j} \f$ is mapped to
/// the \f$(i*n + j)\f$-th element of the container for row major orientation or the \f$ (i + j*m) \f$-th element of
/// the container for column major orientation. In a dense matrix all elements are represented in memory in a
/// contiguous chunk of memory by definition.
///
/// Orientation can also be specified, otherwise a \c row_major is used.
///
/// \tparam T the type of object stored in the matrix (like double, float, complex, etc...)
/// \tparam L the storage organization. It can be either \c row_major or \c column_major. Default is \c row_major
/// \tparam Device the device this matrix lives on, the default is cpu_tag for a cpu matrix
template<class T, class L=row_major, class Device = cpu_tag>
class matrix;

template<class T, class Tag = dense_tag, class Device = cpu_tag>
class dense_vector_adaptor;

template<class T,class Orientation = row_major, class Tag = dense_tag, class Device = cpu_tag>
class dense_matrix_adaptor;

///////////////////////////////////
// Adapt memory as vector
///////////////////////////////////

/// \brief Converts a chunk of memory into a vector of a given size.
template <class T>
dense_vector_adaptor<T, continuous_dense_tag, cpu_tag> adapt_vector(std::size_t size, T * v){
	return dense_vector_adaptor<T, continuous_dense_tag, cpu_tag>(v,size);
}

/// \brief Converts a C-style array into a vector.
template <class T, std::size_t N>
dense_vector_adaptor<T, continuous_dense_tag, cpu_tag> adapt_vector(T (&array)[N]){
	return dense_vector_adaptor<T, continuous_dense_tag, cpu_tag>(array,N);
}

/// \brief Converts a chunk of memory into a matrix of given size.
template <class T>
dense_matrix_adaptor<T, row_major, continuous_dense_tag, cpu_tag> adapt_matrix(std::size_t size1, std::size_t size2, T* data){
	return dense_matrix_adaptor<T, row_major, continuous_dense_tag, cpu_tag>(data,size1, size2);
}

/// \brief Converts a 2D C-style array into a matrix of given size.
template <class T, std::size_t M, std::size_t N>
dense_matrix_adaptor<T, row_major, continuous_dense_tag, cpu_tag> adapt_matrix(T (&array)[M][N]){
	return dense_matrix_adaptor<T, row_major, continuous_dense_tag, cpu_tag>(&(array[0][0]),M,N);
}


///////////////////////////////////
// Traits
///////////////////////////////////

template<class T, class Device>
struct vector_temporary_type<T,dense_tag, Device>{
	typedef vector<T, Device> type;
};

template<class T, class Device>
struct vector_temporary_type<T,continuous_dense_tag, Device>{
	typedef vector<T, Device> type;
};

template<class T, class L, class Device>
struct matrix_temporary_type<T,L,dense_tag, Device>{
	typedef matrix<T,L, Device> type;
};
template<class T, class L, class Device>
struct matrix_temporary_type<T,L,continuous_dense_tag, Device>{
	typedef matrix<T,L, Device> type;
};

template<class T, class Device>
struct matrix_temporary_type<T,unknown_orientation,dense_tag, Device>{
	typedef matrix<T,row_major, Device> type;
};

template<class T, class Device>
struct matrix_temporary_type<T,unknown_orientation,continuous_dense_tag, Device>{
	typedef matrix<T,row_major, Device> type;
};

//////////////////////////////////
//////Expression Traits
///////////////////////////////////


template<class T, class Tag, class Device>
struct vector_range_optimizer<dense_vector_adaptor<T, Tag, Device> >{
	typedef dense_vector_adaptor<T, Tag, Device> type;
	
	static type create(dense_vector_adaptor<T, Tag, Device> const& m, 
		std::size_t start, std::size_t end,
	){
		auto const& storage = m.raw_storage();
		return type(storage.sub_region(start), m.queue(), end - start);
	}
};
template<class T, class Device>
struct vector_range_optimizer<vector<T, Device> >{
	typedef dense_vector_adaptor<T, continuous_dense_tag, Device> type;
	
	static type create(vector<T, Device> const& m, 
		std::size_t start, std::size_t end,
	){
		auto const& storage = m.raw_storage();
		return type(storage.sub_region(start), m.queue(), end - start);
	}
};

template<class T, class Orientation, class Tag, class Device>
struct matrix_transpose_optimizer<dense_matrix_adaptor<T,Orientation, Tag, Device> >{
	typedef dense_matrix_adaptor<T,typename Orientation::transposed_orientation, Tag, Device> type;
	
	static type create(dense_matrix_adaptor<T,Orientation, Tag, Device> const& m){
		auto const& storage = m.raw_storage();
		return type(m.raw_storage(), m.queue(), m.size2(), m.size1());
	}
};

template<class T, class Orientation, class Device>
struct matrix_transpose_optimizer<matrix<T,Orientation, Device> >{
	typedef dense_matrix_adaptor<T,typename Orientation::transposed_orientation, continuous_dense_tag, Device> type;
	
	static type create(matrix<T,Orientation, Device> const& m){
		auto const& storage = m.raw_storage();
		return type(m.raw_storage(), m.queue(), m.size2(), m.size1());
	}
};

template<class T, class Orientation, class Tag, class Device>
struct matrix_row_optimizer<dense_matrix_adaptor<T,Orientation, Tag, Device> >{
	typedef std::conditional<std::is_same<Orientation, row_major>::value, Tag, dense_tag>::type proxy_tag;
	typedef dense_vector_adaptor<T, proxy_tag, Device> type;
	
	static type create(dense_matrix_adaptor<T,Orientation, Tag, Device> const& m, std::size_t i){
		auto const& storage = m.raw_storage();
		return type(storage.row(i, Orientation()), m.queue(), m.size2());
	}
};

template<class T, class Orientation, class Device>
struct matrix_row_optimizer<matrix<T,Orientation, Device> >{
	typedef typename continuous_dense_tag::storage_type::row_storage<Orientation>::type storage_type; 
	typedef dense_vector_adaptor<T, storage_type, Device> type;
	
	static type create(matrix<T,Orientation, Device> const& m){
		auto const& storage = m.raw_storage();
		return type(storage.row(i, Orientation()), m.queue(), m.size2());
	}
};


template<class T, class Orientation, class Tag, class Device>
struct matrix_range_optimizer<dense_matrix_adaptor<T,Orientation, Tag, Device> >{
	typedef dense_matrix_adaptor<T, Orientation, dense_tag, Device> type;
	
	static type create(dense_matrix_adaptor<T,Orientation, Tag, Device> const& m, 
		std::size_t start1, std::size_t end1,
		std::size_t start2, std::size_t end2
	){
		auto const& storage = m.raw_storage();
		return type(storage.sub_region(start1, end1, Orientation()), m.queue(), end1-start1, end2-start2);
	}
};



template<class T, class Orientation, class Device>
struct matrix_range_optimizer<matrix<T,Orientation, Device> >{
	typedef dense_matrix_adaptor<T, Orientation, dense_tag, Device> type;
	
	static type create(dense_matrix_adaptor<T,Orientation, Tag, Device> const& m, 
		std::size_t start1, std::size_t end1,
		std::size_t start2, std::size_t end2
	){
		auto const& storage = m.raw_storage();
		return type(storage.sub_region(start1, start2, Orientation()), m.queue(), end1-start1, end2-start2);
	}
};


}

//include device dependent implementations
#include "cpu/dense.hpp"
#ifdef REMORA_USE_GPU
#include "gpu/dense.hpp"
#endif

#endif
