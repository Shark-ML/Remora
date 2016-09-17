//===========================================================================
/*!
 *
 * \brief       Implements the triangular packed matrix-vector multiplication
 *
 * \author      O. Krause
 * \date        2010
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
//===========================================================================
#ifndef SHARK_LINALG_BLAS_KERNELS_DEFAULT_tpmv_HPP
#define SHARK_LINALG_BLAS_KERNELS_DEFAULT_tpmv_HPP

#include "../../expression_types.hpp"
#include <boost/mpl/bool.hpp>

namespace shark{ namespace blas{ namespace bindings{
	
//Lower triangular(row-major) - vector product
// computes the row-wise inner product of the matrix
// starting with row 0 and stores the result in b(i)
//this does not interfere with the next products as 
// the element b(i) is not needed for iterations j > i
template<class MatA, class V>
void tpmv_impl(
	matrix_expression<MatA, cpu_tag> const& A,
	vector_expression<V, cpu_tag>& b,
	row_major,
	upper
){
	typedef typename V::value_type value_type;
	typedef typename V::index_type index_type;
	typedef typename MatA::const_row_iterator row_iterator;
	index_type size = A().size1();
	
	for(index_type i = 0; i != size; ++i){
		value_type sum(0);
		row_iterator end = A().row_end(i);
		for(row_iterator pos = A().row_begin(i); pos != end; ++pos){
			sum += *pos*b()(pos.index());
		}
		b()(i) = sum;
	}
}

//Upper triangular(row-major) - vector product
// computes the row-wise inner product of the matrix
// starting with the last row and stores the result in b(i)
// this does not interfere with the next products as 
// the element b(i) is not needed for row products j < i
template<class MatA, class V>
void tpmv_impl(
	matrix_expression<MatA, cpu_tag> const& A,
	vector_expression<V, cpu_tag>& b,
	row_major,
	lower
){
	typedef typename V::value_type value_type;
	typedef typename V::index_type index_type;
	typedef typename MatA::const_row_iterator row_iterator;
	index_type size = A().size1();
	
	for(index_type irev = size; irev != 0; --irev){
		index_type i= irev-1;
		value_type sum(0);
		row_iterator end = A().row_end(i);
		for(row_iterator pos = A().row_begin(i); pos != end; ++pos){
			sum += *pos*b()(pos.index());
		}
		b()(i) = sum;
	}
}

//Upper triangular(column-major) - vector product
//computes the product as a series of vector-additions
//on b starting with the last column of A.
template<class MatA, class V>
void tpmv_impl(
	matrix_expression<MatA, cpu_tag> const& A,
	vector_expression<V, cpu_tag>& b,
	column_major,
	upper
){
	typedef typename MatA::const_column_iterator column_iterator;
	typedef typename V::index_type index_type;
	typedef typename V::value_type value_type;
	index_type size = A().size1();
	for(index_type i = 0; i != size; ++i){
		value_type bi = b()(i);
		b()(i)= value_type/*zero*/();
		column_iterator end = A().column_end(i);
		for(column_iterator pos = A().column_begin(i); pos != end; ++pos){
			b()(pos.index()) += *pos*bi;
		}
	}
	
}

//Lower triangular(column-major) - vector product
// computes the product as a series of vector-additions
// on b starting with the first column of A.
template<class MatA, class V>
void tpmv_impl(
	matrix_expression<MatA, cpu_tag> const& A,
	vector_expression<V, cpu_tag>& b,
	column_major,
	lower
){
	typedef typename MatA::const_column_iterator column_iterator;
	typedef typename V::index_type index_type;
	typedef typename V::value_type value_type;
	index_type size = A().size1();
	
	for(index_type irev = size; irev != 0; --irev){
		index_type i= irev-1;
		value_type bi = b()(i);
		b()(i)= value_type/*zero*/();
		column_iterator end = A().column_end(i);
		for(column_iterator pos = A().column_begin(i); pos != end; ++pos){
			b()(pos.index()) += *pos*bi;
		}
	}
}

//dispatcher
template <typename MatA, typename V>
void tpmv(
	matrix_expression<MatA, cpu_tag> const& A, 
	vector_expression<V, cpu_tag>& b,
	boost::mpl::false_//unoptimized
){
	SIZE_CHECK(A().size1() == A().size2());
	SIZE_CHECK(A().size2() == b().size());	
	tpmv_impl(
		A, b,
		typename MatA::orientation::orientation(),
		typename MatA::orientation::triangular_type()
	);
}

}}}
#endif
