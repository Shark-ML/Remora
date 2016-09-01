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

#ifndef SHARK_LINALG_BLAS_MATRIX_PROXY_HPP
#define SHARK_LINALG_BLAS_MATRIX_PROXY_HPP

#include "detail/matrix_proxy_classes.hpp"
#include "detail/matrix_proxy_optimizers.hpp"

namespace shark {
namespace blas {
	
	
////////////////////////////////////
//// Matrix Transpose
////////////////////////////////////

// (trans m) [i] [j] = m [j] [i]
template<class M>
temporary_proxy<typename detail::matrix_transpose_optimizer<M>::type>
trans(matrix_expression<M> & m){
	return detail::matrix_transpose_optimizer<M>::create(m());
}
template<class M>
typename detail::matrix_transpose_optimizer<typename const_expression<M>::type >::type
trans(matrix_expression<M> const& m){
	typedef typename const_expression<M>::type closure_type;
	return detail::matrix_transpose_optimizer<closure_type>::create(m());
}

template<class M>
temporary_proxy<typename detail::matrix_transpose_optimizer<M>::type>
trans(temporary_proxy<M> m){
	return trans(static_cast<M&>(m));
}

////////////////////////////////////
//// Matrix Row and Column
////////////////////////////////////

/// \brief Returns a vector-proxy representing the i-th row of the Matrix
template<class M>
temporary_proxy< matrix_row<M> > row(matrix_expression<M>& expression, typename M::index_type i){
	return matrix_row<M> (expression(), i);
}
template<class M>
matrix_row<typename const_expression<M>::type>
row(matrix_expression<M> const& expression, typename M::index_type i){
	return matrix_row<typename const_expression<M>::type> (expression(), i);
}

template<class M>
temporary_proxy<matrix_row<M> > row(temporary_proxy<M> expression, typename M::index_type i){
	return row(static_cast<M&>(expression), i);
}

/// \brief Returns a vector-proxy representing the j-th column of the Matrix
template<class M>
temporary_proxy<matrix_row<typename detail::matrix_transpose_optimizer<M>::type> >
column(matrix_expression<M>& expression, typename M::index_type j){
	return row(trans(expression),j);
}
template<class M>
matrix_row<typename detail::matrix_transpose_optimizer<typename const_expression<M>::type >::type>
column(matrix_expression<M> const& expression, typename M::index_type j){
	return row(trans(expression),j);
}

template<class M>
temporary_proxy<matrix_row<typename detail::matrix_transpose_optimizer<M>::type> >
column(temporary_proxy<M> expression, typename M::index_type j){
	return row(trans(static_cast<M&>(expression)),j);
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
template<class M>
matrix_vector_range<typename const_expression<M>::type > diag(matrix_expression<M> const& mat){
	SIZE_CHECK(mat().size1() == mat().size2());
	matrix_vector_range<typename const_expression<M>::type > diagonal(mat(),range(0,mat().size1()),range(0,mat().size1()));
	return diagonal;
}

///\brief Returns the diagonal of a square matrix as vector.
///
/// given a matrix 
/// A = (1 2 3)
///     (4 5 6)
///     (7 8 9)
///
/// the diag operation results in
/// diag(A) = (1,5,9)
template<class M>
temporary_proxy< matrix_vector_range<M> > diag(matrix_expression<M>& mat){
	SIZE_CHECK(mat().size1() == mat().size2());
	matrix_vector_range<M> diagonal(mat(),range(0,mat().size1()),range(0,mat().size1()));
	return diagonal;
}

template<class M>
temporary_proxy< matrix_vector_range<M> > diag(temporary_proxy<M> mat){
	return diag(static_cast<M&>(mat));
}


////////////////////////////////////
//// Matrix Subranges
////////////////////////////////////

template<class M>
temporary_proxy< matrix_range<M> > subrange(
	matrix_expression<M>& expression, 
	std::size_t start1, std::size_t stop1,
	std::size_t start2, std::size_t stop2
){
	RANGE_CHECK(start1 <= stop1);
	RANGE_CHECK(start2 <= stop2);
	SIZE_CHECK(stop1 <= expression().size1());
	SIZE_CHECK(stop2 <= expression().size2());
	return matrix_range<M> (expression(), range(start1, stop1), range(start2, stop2));
}
template<class M>
matrix_range<typename const_expression<M>::type> subrange(
	matrix_expression<M> const& expression, 
	std::size_t start1, std::size_t stop1,
	std::size_t start2, std::size_t stop2
){
	RANGE_CHECK(start1 <= stop1);
	RANGE_CHECK(start2 <= stop2);
	SIZE_CHECK(stop1 <= expression().size1());
	SIZE_CHECK(stop2 <= expression().size2());
	return matrix_range<typename const_expression<M>::type> (expression(), range(start1, stop1), range(start2, stop2));
}

template<class M>
temporary_proxy< matrix_range<M> > subrange(
	temporary_proxy<M> expression, 
	std::size_t start1, std::size_t stop1,
	std::size_t start2, std::size_t stop2
){
	return subrange(static_cast<M&>(expression),start1,stop1,start2,stop2);
}

template<class M>
temporary_proxy<matrix_range<M> > rows(
	matrix_expression<M>& expression, 
	std::size_t start, std::size_t stop
){
	RANGE_CHECK(start <= stop);
	SIZE_CHECK(stop <= expression().size1());
	return subrange(expression, start, stop, 0,expression().size2());
}

template<class M>
matrix_range<typename const_expression<M>::type> rows(
	matrix_expression<M> const& expression, 
	std::size_t start, std::size_t stop
){
	RANGE_CHECK(start <= stop);
	SIZE_CHECK(stop <= expression().size1());
	return subrange(expression, start, stop, 0,expression().size2());
}

template<class M>
temporary_proxy<matrix_range<M> > rows(
	temporary_proxy<M> expression, 
	std::size_t start, std::size_t stop
){
	return rows(static_cast<M&>(expression),start,stop);
}

template<class M>
temporary_proxy< matrix_range<M> > columns(
	matrix_expression<M>& expression, 
	typename M::index_type start, typename M::index_type stop
){
	RANGE_CHECK(start <= stop);
	SIZE_CHECK(stop <= expression().size2());
	return subrange(expression, 0,expression().size1(), start, stop);
}

template<class M>
matrix_range<typename const_expression<M>::type> columns(
	matrix_expression<M> const& expression, 
	typename M::index_type start, typename M::index_type stop
){
	RANGE_CHECK(start <= stop);
	SIZE_CHECK(stop <= expression().size2());
	return subrange(expression, 0,expression().size1(), start, stop);
}

template<class M>
temporary_proxy<matrix_range<M> > columns(
	temporary_proxy<M> expression, 
	std::size_t start, std::size_t stop
){
	return columns(static_cast<M&>(expression),start,stop);
}

////////////////////////////////////
//// Matrix Adaptor
////////////////////////////////////

/// \brief Converts a chunk of memory into a matrix of given size.
template <class T>
temporary_proxy< dense_matrix_adaptor<T> > adapt_matrix(std::size_t size1, std::size_t size2, T* data){
	return dense_matrix_adaptor<T>(data,size1, size2);
}

/// \brief Converts a 2D C-style array into a matrix of given size.
template <class T, std::size_t M, std::size_t N>
temporary_proxy<dense_matrix_adaptor<T> > adapt_matrix(T (&array)[M][N]){
	return dense_matrix_adaptor<T>(&(array[0][0]),M,N);
}

/// \brief Converts a dense vector to a matrix of a given size
template <class V>
typename boost::enable_if<
	boost::is_same<typename V::storage_category,dense_tag>,
	temporary_proxy< dense_matrix_adaptor<
		typename boost::remove_reference<typename V::reference>::type
	> >
>::type
to_matrix(
	vector_expression<V>& v,
	std::size_t size1, std::size_t size2
){
	typedef typename boost::remove_reference<typename V::reference>::type ElementType;
	return dense_matrix_adaptor<ElementType>(v().storage(), size1, size2);
}

/// \brief Converts a dense vector to a matrix of a given size
template <class V>
typename boost::enable_if<
	boost::is_same<typename V::storage_category,dense_tag>,
	temporary_proxy< dense_matrix_adaptor<typename V::value_type const> >
>::type 
to_matrix(
	vector_expression<V> const& v,
	std::size_t size1, std::size_t size2
){
	return dense_matrix_adaptor<typename V::value_type const>(v().storage(), size1, size2);
}

template <class E>
typename boost::enable_if<
	boost::is_same<typename E::storage_category,dense_tag>,
	temporary_proxy< dense_matrix_adaptor<
		typename boost::remove_reference<typename E::reference>::type
	> >
>::type 
to_matrix(
	temporary_proxy<E> v,
	std::size_t size1, std::size_t size2
){
	return to_matrix(static_cast<E&>(v),size1,size2);
}

}}
#endif
