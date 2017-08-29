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
namespace remora{
	
	
// ------------------
// Vector subrange
// ------------------

/// \brief Return a subrange of a specified vector, forming a vector for the specified indices between start and stop index.
///
/// The vector starts with first index being 0 for the element that is indexed with start in the original vector.
template<class V, class Device>
typename detail::vector_range_optimizer<V>::type
subrange(vector_expression<V, Device>& v, std::size_t start, std::size_t stop){
	return detail::vector_range_optimizer<V>::create(v(), start, stop);
}

template<class V, class Device>
typename detail::vector_range_optimizer<typename const_expression<V>::type>::type
subrange(vector_expression<V, Device> const& v, std::size_t start, std::size_t stop){
	return detail::vector_range_optimizer<typename const_expression<V>::type>::create(v(), start, stop);
}

template<class V, class Device>
typename detail::vector_range_optimizer<V>::type
subrange(vector_expression<V,Device>&& v, std::size_t start, std::size_t stop){
	return subrange(v(),start,stop);
}
	
	
////////////////////////////////////
//// Matrix Transpose
////////////////////////////////////

/// \brief Returns a proxy which transposes the matrix
///
/// given a matrix 
/// A = (1 2 3)
///     (4 5 6)
///     (7 8 9)
///
/// the trans(A) operation results in
/// trans(A) = (1 4 7)
///            (2 5 8)
///            (3 6 9)
template<class M, class Device>
typename detail::matrix_transpose_optimizer<M>::type
trans(matrix_expression<M, Device> & m){
	return detail::matrix_transpose_optimizer<M>::create(m());
}
template<class M, class Device>
typename detail::matrix_transpose_optimizer<typename const_expression<M>::type >::type
trans(matrix_expression<M, Device> const& m){
	typedef typename const_expression<M>::type closure_type;
	return detail::matrix_transpose_optimizer<closure_type>::create(m());
}

template<class M, class Device>
typename detail::matrix_transpose_optimizer<M>::type
trans(matrix_expression<M, Device> && m){
	static_assert(!std::is_base_of<matrix_container<M, Device>,M>::value, "It is unsafe to create a proxy from a temporary container");
	return trans(m());
}

////////////////////////////////////
//// Matrix Row and Column
////////////////////////////////////

/// \brief Returns a vector-proxy representing the i-th row of the Matrix
///
/// given a matrix 
/// A = (1 2 3)
///     (4 5 6)
///     (7 8 9)
///
/// the row(A,1) operation results in
/// row(A,1) = (4,5,6)
template<class M, class Device>
typename detail::matrix_row_optimizer<M>::type
row(matrix_expression<M, Device>& m, typename M::size_type i){
	return detail::matrix_row_optimizer<M>::create(m(), i);
}
template<class M, class Device>
typename detail::matrix_row_optimizer<typename const_expression<M>::type>::type
row(matrix_expression<M, Device> const& m, typename M::size_type i){
	return detail::matrix_row_optimizer<typename const_expression<M>::type>::create(m(), i);
}

template<class M>
typename detail::matrix_row_optimizer<M>::type
row(M && m, typename M::size_type i){
	static_assert(!std::is_base_of<matrix_container<M, typename M::device_type>,M>::value, "It is unsafe to create a proxy from a temporary container");
	return row(m,i);
}

/// \brief Returns a vector-proxy representing the j-th column of the Matrix
///
/// given a matrix 
/// A = (1 2 3)
///     (4 5 6)
///     (7 8 9)
///
/// the column(A,1) operation results in
/// column(A,1) = (2,5,8)
template<class M, class Device>
auto column(matrix_expression<M, Device>& m, typename M::size_type j) -> decltype(row(trans(m),j)){
	return row(trans(m),j);
}
template<class M, class Device>
auto column(matrix_expression<M, Device> const& m, typename M::size_type j) -> decltype(row(trans(m),j)){
	return row(trans(m),j);
}

template<class M, class Device>
auto column(matrix_expression<M, Device> && m, typename M::size_type j) -> decltype(row(trans(m),j)){
	static_assert(!std::is_base_of<matrix_container<M, Device>,M>::value, "It is unsafe to create a proxy from a temporary container");
	return column(m());
}

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
typename detail::matrix_vector_range_optimizer<M>::type
diag(matrix_expression<M, Device>& mat){
	REMORA_SIZE_CHECK(mat().size1() == mat().size2());
	return detail::matrix_vector_range_optimizer<M>::create(mat(),0,mat().size1(),0,mat().size1());
}

template<class M, class Device>
typename detail::matrix_vector_range_optimizer<typename const_expression<M>::type>::type
diag(matrix_expression<M, Device> const& mat){
	REMORA_SIZE_CHECK(mat().size1() == mat().size2());
	return detail::matrix_vector_range_optimizer<typename const_expression<M>::type>::create(mat(),0,mat().size1(),0,mat().size1());
}


template<class M, class Device>
typename detail::matrix_vector_range_optimizer<M>::type
diag(matrix_expression<M, Device> && m){
	static_assert(!std::is_base_of<matrix_container<M, Device>,M>::value, "It is unsafe to create a proxy from a temporary container");
	return diag(m());
}

////////////////////////////////////
//// Matrix Subranges
////////////////////////////////////


///\brief Returns a submatrix of a given matrix.
///
/// given a matrix 
/// A = (1 2 3)
///     (4 5 6)
///     (7 8 9)
///
/// the subrange(A,0,2,1,3) operation results in
/// subrange(A,0,2,1,3) = (4 5)
///                       (7 8)
template<class M, class Device>
typename detail::matrix_range_optimizer<M>::type subrange(
	matrix_expression<M, Device>& m, 
	std::size_t start1, std::size_t stop1,
	std::size_t start2, std::size_t stop2
){
	REMORA_RANGE_CHECK(start1 <= stop1);
	REMORA_RANGE_CHECK(start2 <= stop2);
	REMORA_SIZE_CHECK(stop1 <= m().size1());
	REMORA_SIZE_CHECK(stop2 <= m().size2());
	return detail::matrix_range_optimizer<M>::create(m(), start1, stop1, start2, stop2);
}
template<class M, class Device>
typename detail::matrix_range_optimizer<typename const_expression<M>::type>::type subrange(
	matrix_expression<M, Device> const& m, 
	std::size_t start1, std::size_t stop1,
	std::size_t start2, std::size_t stop2
){
	REMORA_RANGE_CHECK(start1 <= stop1);
	REMORA_RANGE_CHECK(start2 <= stop2);
	REMORA_SIZE_CHECK(stop1 <= m().size1());
	REMORA_SIZE_CHECK(stop2 <= m().size2());
	return detail::matrix_range_optimizer<typename const_expression<M>::type>::create(m(), start1, stop1, start2, stop2);
}

template<class M, class Device>
typename detail::matrix_range_optimizer<M>::type subrange(
	matrix_expression<M, Device> && m, 
	std::size_t start1, std::size_t stop1,
	std::size_t start2, std::size_t stop2
){
	static_assert(!std::is_base_of<matrix_container<M, Device>,M>::value, "It is unsafe to create a proxy from a temporary container");
	return subrange(m(),start1,stop1,start2,stop2);
}

template<class M, class Device>
auto rows(
	matrix_expression<M, Device>& m, 
	std::size_t start, std::size_t stop
) -> decltype(subrange(m, start, stop, 0,m().size2())){
	REMORA_RANGE_CHECK(start <= stop);
	REMORA_SIZE_CHECK(stop <= m().size1());
	return subrange(m, start, stop, 0,m().size2());
}

template<class M, class Device>
auto rows(
	matrix_expression<M, Device> const& m, 
	std::size_t start, std::size_t stop
) -> decltype(subrange(m, start, stop, 0,m().size2())){
	REMORA_RANGE_CHECK(start <= stop);
	REMORA_SIZE_CHECK(stop <= m().size1());
	return subrange(m, start, stop, 0,m().size2());
}

template<class M, class Device>
auto rows(
	matrix_expression<M, Device> && m, 
	std::size_t start, std::size_t stop
) -> decltype(subrange(m, start, stop, 0,m().size2())){
	static_assert(!std::is_base_of<matrix_container<M, Device>,M>::value, "It is unsafe to create a proxy from a temporary container");
	return rows(m(),start,stop);
}

template<class M, class Device>
auto columns(
	matrix_expression<M, Device>& m, 
	typename M::size_type start, typename M::size_type stop
) -> decltype(subrange(m, 0,m().size1(), start, stop)){
	REMORA_RANGE_CHECK(start <= stop);
	REMORA_SIZE_CHECK(stop <= m().size2());
	return subrange(m, 0,m().size1(), start, stop);
}

template<class M, class Device>
auto columns(
	matrix_expression<M, Device> const& m, 
	typename M::size_type start, typename M::size_type stop
) -> decltype(subrange(m, 0,m().size1(), start, stop)){
	REMORA_RANGE_CHECK(start <= stop);
	REMORA_SIZE_CHECK(stop <= m().size2());
	return subrange(m, 0,m().size1(), start, stop);
}

template<class M, class Device>
auto columns(
	matrix_expression<M, Device> && m, 
	std::size_t start, std::size_t stop
) -> decltype(subrange(m, 0,m().size1(), start, stop)){
	static_assert(!std::is_base_of<matrix_container<M, Device>,M>::value, "It is unsafe to create a proxy from a temporary container");
	return columns(m(),start,stop);
}

////////////////////////////////////
//// Matrix to Vector
////////////////////////////////////

/// \brief Converts a dense matrix to a vector
///
/// The matrix is linearized along its fast index as indicated by the orientation.
/// m.g. a row-major matrix is lienarized by concatenating its rows to one large vector.
template<class M, class Device>
typename detail::linearized_matrix_optimizer<M>::type
to_vector(matrix_expression<M, Device> const& m){
	return detail::linearized_matrix_optimizer<M>::create(m());
}

template<class M, class Device>
typename detail::linearized_matrix_optimizer<typename const_expression<M>::type>::type
to_vector(matrix_expression<M, Device> const& m){
	return detail::linearized_matrix_optimizer<typename const_expression<M>::type>::create(m());
}

template<class M, class Device>
typename detail::linearized_matrix_optimizer<M>::type
to_vector(matrix_expression<M, Device> && m){
	static_assert(!std::is_base_of<vector_container<M, Device>,M>::value, "It is unsafe to create a proxy from a temporary container");
	return to_vector(m());
}


//~ ////////////////////////////////////
//~ //// Matrix Adaptor
//~ ////////////////////////////////////

//~ /// \brief Converts a dense vector to a matrix of a given size
//~ template <class V, class Device>
//~ typename std::enable_if<
	//~ std::is_same<typename V::storage_type::storage_tag,continuous_dense_tag>::value,
	//~ dense_matrix_adaptor<
		//~ typename std::remove_reference<typename V::reference>::type,
		//~ row_major, Device
	//~ >
//~ >::type
//~ to_matrix(
	//~ vector_expression<V, Device>& v,
	//~ std::size_t size1, std::size_t size2
//~ ){
	//~ REMORA_SIZE_CHECK(size1 * size2 == v().size());
	//~ typedef typename std::remove_reference<typename V::reference>::type ElementType;
	//~ return dense_matrix_adaptor<ElementType, row_major, Device>(v, size1, size2);
//~ }

//~ /// \brief Converts a dense vector to a matrix of a given size
//~ template <class V, class Device>
//~ typename std::enable_if<
	//~ std::is_same<typename V::storage_type::storage_tag,continuous_dense_tag>::value,
	//~ dense_matrix_adaptor<typename V::value_type const,row_major, Device>
//~ >::type 
//~ to_matrix(
	//~ vector_expression<V, Device> const& v,
	//~ std::size_t size1, std::size_t size2
//~ ){
	//~ REMORA_SIZE_CHECK(size1 * size2 == v().size());
	//~ return dense_matrix_adaptor<typename V::value_type const, row_major, Device>(v, size1, size2);
//~ }

//~ template<class V>
//~ auto to_matrix(vector_expression<V,Device>&& v, std::size_t size1, std::size_t size2)->decltype(to_matrix(static_cast<M&>(m()),size1, size2)){
	//~ static_assert(!std::is_base_of<vector_container<V>,V>::value, "It is unsafe to create a proxy from a temporary container");
	//~ return to_matrix(v(), size1, size2);
//~ }
}

#endif
