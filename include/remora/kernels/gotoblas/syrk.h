/*!
 * 
 *
 * \brief       -
 *
 * \author      O. Krause
 * \date        2012
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

#ifndef SHARK_LINALG_BLAS_KERNELS_GOTOBLAS_SYRK_H
#define SHARK_LINALG_BLAS_KERNELS_GOTOBLAS_SYRK_H

#include "cblas_inc.h"

namespace shark {namespace detail {namespace bindings {
// C <- alpha * A * A^T + beta * C
// C <- alpha * A^T * A + beta * C

inline void syrk(
	CBLAS_ORDER const Order, CBLAS_UPLO const Uplo,
	 CBLAS_TRANSPOSE const Trans, int const N, int const K,
	 float const alpha, float const *A, int const lda,
	 float const beta, float *C, int const ldc
) {
	cblas_ssyrk(Order, Uplo, Trans, N, K, alpha, const_cast<float*>(A), lda, beta, C, ldc);
}

inline void syrk(
	CBLAS_ORDER const Order, CBLAS_UPLO const Uplo,
	 CBLAS_TRANSPOSE const Trans, int const N, int const K,
	 double const alpha, double const *A, int const lda,
	 double const beta, double *C, int const ldc
) {
	cblas_dsyrk(Order, Uplo, Trans, N, K, alpha, const_cast<double*>(A), lda, beta, C, ldc);
}

inline void syrk(
	CBLAS_ORDER const Order, CBLAS_UPLO const Uplo,
	 CBLAS_TRANSPOSE const Trans, int const N, int const K,
	 std::complex<float> const &alpha,
	 std::complex<float> const *A, int const lda,
	 std::complex<float> const &beta,
	 std::complex<float> *C, int const ldc
) {
	cblas_csyrk(Order, Uplo, Trans, N, K,
		 (float*)(&alpha),
		 (float*)(A), lda,
		 (float*)(&beta),
		 (float*)(C), ldc);
}

inline void syrk(
	CBLAS_ORDER const Order, CBLAS_UPLO const Uplo,
	 CBLAS_TRANSPOSE const Trans, int const N, int const K,
	 std::complex<double> const &alpha,
	 std::complex<double> const *A, int const lda,
	 std::complex<double> const &beta,
	 std::complex<double> *C, int const ldc
) {
	cblas_zsyrk(Order, Uplo, Trans, N, K,
		 (double*)(&alpha),
		 (double*)(A), lda,
		 (double*)(&beta),
		 (double*)(C), ldc);
}

template <typename T, typename MatrA, typename SymmC>
inline void syrk (
	CBLAS_UPLO const uplo, CBLAS_TRANSPOSE trans, 
	T const& alpha, blas::matrix_expression<MatrA> const& a, 
	T const& beta, blas::matrix_expression<SymmC>& c
){

	std::size_t const n = c().size1();
	SIZE_CHECK (n == c().size2()); 
	 
	std::size_t k = 0;
	if( trans == CblasNoTrans ){
		k = a().size2();
		SIZE_CHECK(n == a().size1());
	}
	else{
		k = a().size1();
		SIZE_CHECK(n == a().size2());
	}

	 CBLAS_ORDER const stor_ord
		= (CBLAS_ORDER)storage_order<typename MatrA::orientation>::value;

	 syrk (stor_ord, uplo, trans, 
		(int)n, (int)k, alpha, 
		traits::matrix_storage (a()), 
		traits::leadingDimension (a()),
		beta, 
		traits::matrix_storage (c()), 
		traits::leadingDimension (c())
	); 
}
template <bool upper,typename T, typename MatrA, typename SymmC>
inline void syrk (
	T const& alpha, blas::matrix_expression<MatrA> const& matA, 
	T const& beta, blas::matrix_expression<SymmC>& matC
){
	if(traits::isTransposed(matA))
		syrk(upper?CblasLower:CblasUpper,CblasTrans,alpha,trans(matA),beta,matC);
	else
		syrk(upper?CblasLower:CblasUpper,CblasNoTrans,alpha,matA,beta,matC);
}

}}}

#endif
