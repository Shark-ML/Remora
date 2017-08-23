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

template<class T, class Device>
struct vector_temporary_type<T,dense_tag, Device>{
	typedef vector<T, Device> type;
};

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

template<class T, class L, class Device>
struct matrix_temporary_type<T,L,dense_tag, Device>{
	typedef matrix<T,L, Device> type;
};

template<class T, class Device>
struct matrix_temporary_type<T,unknown_orientation,dense_tag, Device>{
	typedef matrix<T,row_major, Device> type;
};

template<class T, class Tag = dense_tag, class Device = cpu_tag>
class dense_vector_adaptor;

template<class T,class Orientation = row_major, class Tag = dense_tag, class Device = cpu_tag>
class dense_matrix_adaptor;

// ------------------
// Adapt memory as vector
// ------------------

/// \brief Converts a chunk of memory into a vector of a given size.
template <class T>
dense_vector_adaptor<T> adapt_vector(std::size_t size, T * v){
	return dense_vector_adaptor<T>(v,size);
}

/// \brief Converts a C-style array into a vector.
template <class T, std::size_t N>
dense_vector_adaptor<T> adapt_vector(T (&array)[N]){
	return dense_vector_adaptor<T>(array,N);
}

/// \brief Converts a chunk of memory into a matrix of given size.
template <class T>
dense_matrix_adaptor<T> adapt_matrix(std::size_t size1, std::size_t size2, T* data){
	return dense_matrix_adaptor<T>(data,size1, size2);
}

/// \brief Converts a 2D C-style array into a matrix of given size.
template <class T, std::size_t M, std::size_t N>
dense_matrix_adaptor<T> adapt_matrix(T (&array)[M][N]){
	return dense_matrix_adaptor<T>(&(array[0][0]),M,N);
}


}

//include device dependent implementations
#include "cpu/dense.hpp"
#ifdef REMORA_USE_GPU
#include "gpu/dense.hpp"
#endif

#endif
