/*!
 *  \author O. Krause
 *  \date 2011
 *
 *  \par Copyright (c) 1998-2011:
 *      Institut f&uuml;r Neuroinformatik<BR>
 *      Ruhr-Universit&auml;t Bochum<BR>
 *      D-44780 Bochum, Germany<BR>
 *      Phone: +49-234-32-25558<BR>
 *      Fax:   +49-234-32-14209<BR>
 *      eMail: Shark-admin@neuroinformatik.ruhr-uni-bochum.de<BR>
 *      www:   http://www.neuroinformatik.ruhr-uni-bochum.de<BR>
 *      <BR>
 *
 *
 *  <BR><HR>
 *  This file is part of Shark. This library is free software;
 *  you can redistribute it and/or modify it under the terms of the
 *  GNU General Public License as published by the Free Software
 *  Foundation; either version 3, or (at your option) any later version.
 *
 *  This library is distributed in the hope that it will be useful,
 *  but WITHOUT ANY WARRANTY; without even the implied warranty of
 *  MERCHANTABILITY or FITNESS FOR matA PARTICULAR PURPOSE. See the
 *  GNU General Public License for more details.
 *
 *  You should have received matA copy of the GNU General Public License
 *  along with this library; if not, see <http://www.gnu.org/licenses/>.
 *
 */

#ifndef SHARK_LINALG_IMPL_NUMERIC_BINDINGS_ATLAS_TRSM_HPP
#define SHARK_LINALG_IMPL_NUMERIC_BINDINGS_ATLAS_TRSM_HPP

#include "cblas_inc.h"

///solves systems of triangular matrices

namespace shark {namespace detail {namespace bindings {
inline void trsm(
	CBLAS_ORDER order, CBLAS_UPLO uplo,CBLAS_TRANSPOSE transA, 
	CBLAS_SIDE side, CBLAS_DIAG unit,
	int n, int nRHS,
	float const *matA, int lda, float *matB, int ldb
) {
	cblas_strsm(order, side, uplo, transA, unit,n, nRHS, 1.0, matA, lda, matB, ldb);
}

inline void trsm(
	CBLAS_ORDER order, CBLAS_UPLO uplo,CBLAS_TRANSPOSE transA, 
	CBLAS_SIDE side, CBLAS_DIAG unit,
	int n, int nRHS,
	double const *matA, int lda, double *matB, int ldb
) {
	cblas_dtrsm(order, side, uplo, transA, unit,n, nRHS, 1.0, matA, lda, matB, ldb);
}

inline void trsm(
	CBLAS_ORDER order, CBLAS_UPLO uplo,CBLAS_TRANSPOSE transA, 
	CBLAS_SIDE side, CBLAS_DIAG unit,
	int n, int nRHS,
	std::complex<float> const *matA, int lda, std::complex<float> *matB, int ldb
) {
	std::complex<float> alpha(1.0,0);
	cblas_ctrsm(order, side, uplo, transA, unit,n, nRHS,
		static_cast<void const *>(&alpha),
	        static_cast<void const *>(matA), lda,
	        static_cast<void *>(matB), ldb);
}
inline void trsm(
	CBLAS_ORDER order, CBLAS_UPLO uplo,CBLAS_TRANSPOSE transA, 
	CBLAS_SIDE side, CBLAS_DIAG unit,
	int n, int nRHS,
	std::complex<double> const *matA, int lda, std::complex<double> *matB, int ldb
) {
	std::complex<double> alpha(1.0,0);
	cblas_ztrsm(order, side, uplo, transA, unit,n, nRHS,
		static_cast<void const *>(&alpha),
	        static_cast<void const *>(matA), lda,
	        static_cast<void *>(matB), ldb);
}


// trsm(): solves matA system of linear equations matA * X = matB
//             when matA is matA triangular matrix
template <typename SymmA, typename MatrB>
void trsm(
	CBLAS_UPLO uplo, CBLAS_TRANSPOSE transA,
	CBLAS_SIDE side, CBLAS_DIAG unit,
	blas::matrix_expression<SymmA> const &matA, 
	blas::matrix_expression<MatrB> &matB
){
	CBLAS_ORDER const storOrd = (CBLAS_ORDER)storage_order<typename SymmA::orientation_category>::value;

	SIZE_CHECK(matA().size1() == matA().size2());

	int m = matB().size1();
	int nrhs = matB().size2();
	
	trsm(storOrd, uplo, transA, side,unit, m, nrhs,
		traits::matrix_storage(matA()),
		traits::leadingDimension(matA()),
		traits::matrix_storage(matB()),
		traits::leadingDimension(matB())
	);
}
template <bool upper, bool left, bool unit,typename SymmA, typename MatrB>
void trsm(
	blas::matrix_expression<SymmA> const &matA,
	blas::matrix_expression<MatrB> &matB
){
	CBLAS_DIAG cblasUnit = unit?CblasUnit:CblasNonUnit;
	CBLAS_SIDE cblasLeft = left?CblasLeft:CblasRight;
	if(traits::isTransposed(matA))
		trsm(upper?CblasLower:CblasUpper,CblasTrans,cblasLeft,cblasUnit,trans(matA),matB);
	else
		trsm(upper?CblasUpper:CblasLower,CblasNoTrans,cblasLeft,cblasUnit,matA,matB);
}

}}}
#endif
