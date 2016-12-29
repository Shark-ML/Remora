/*!
 * 
 *
 * \brief       -
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

#ifndef SHARK_LINALG_BLAS_KERNELS_DEFAULT_GEMM_HPP
#define SHARK_LINALG_BLAS_KERNELS_DEFAULT_GEMM_HPP

#include "../gemv.hpp"
#include "../../vector.hpp"
#include "../../detail/matrix_proxy_classes.hpp"
#include <boost/mpl/bool.hpp>
#include <boost/align/aligned_allocator.hpp>
#include <boost/align/assume_aligned.hpp>
#include <type_traits>
#include <vector>


#define SHARK_BLAS_VECTOR_LENGTH 16

namespace shark {namespace blas {namespace bindings {
	
//  Block-GEMM implementation based on boost.ublas
//  written by:
//  Copyright (c) 2016
//  Michael Lehn, Imre Palik
//
//  Distributed under the Boost Software License, Version 1.0. (See
//  accompanying file LICENSE_1_0.txt or copy at
//  http://www.boost.org/LICENSE_1_0.txt)
	

template <typename T>
struct prod_block_size {
	static const unsigned vector_length = SHARK_BLAS_VECTOR_LENGTH/sizeof(T); // Number of elements in a vector register
	static const unsigned mr = 4; // stripe width for lhs
	static const unsigned nr = 3 * vector_length; // stripe width for rhs
	static const unsigned mc = 128;
	static const unsigned kc = 512; // stripe length
	static const unsigned nc = (1024/nr) * nr;
	static const unsigned align = 64; // align temporary arrays to this boundary
};

template <>
struct prod_block_size<float> {
	static const unsigned vector_length = SHARK_BLAS_VECTOR_LENGTH/sizeof(float); 
	static const unsigned mc = 256;
	static const unsigned kc = 512; // stripe length
	static const unsigned nc = 4096;
	static const unsigned mr = 4; // stripe width for lhs
	static const unsigned nr = 16; // stripe width for rhs
	static const unsigned align = 64; // align temporary arrays to this boundary
};

template <>
struct prod_block_size<long double> {
	static const unsigned vector_length = SHARK_BLAS_VECTOR_LENGTH/sizeof(long double); 
	static const unsigned mc = 256;
	static const unsigned kc = 512; // stripe length
	static const unsigned nc = 4096;
	static const unsigned mr = 1; // stripe width for lhs
	static const unsigned nr = 4; // stripe width for rhs
	static const unsigned align = 64; // align temporary arrays to this boundary
};

//-- Micro Kernel For Dense operations----------------------------------------------------------
template <class block_size, class T, class TC>
void ugemm(
	std::size_t kc, TC alpha, T const* A, T const* B,
	TC* C, std::size_t ldc
){
	BOOST_ALIGN_ASSUME_ALIGNED(A, block_size::align);
	BOOST_ALIGN_ASSUME_ALIGNED(B, block_size::align);
	
#ifdef SHARK_USE_SIMD
	static const std::size_t vecNR = block_size::nr/block_size::vector_length;
#ifdef BOOST_COMP_CLANG_DETECTION
	typedef T vx __attribute__((ext_vector_type (vector_length)));
#else
        typedef T vx __attribute__((vector_size (SHARK_BLAS_VECTOR_LENGTH)));
#endif
	vx P[block_size::mr * vecNR] = {};
#else
typedef T vx;
static const std::size_t vecNR = block_size::nr;
typename std::aligned_storage<sizeof(T[block_size::mr*vecNR]),block_size::align>::type Pa;
	T* P = reinterpret_cast<T*>(&Pa);
	for (std::size_t c = 0; c < block_size::mr*vecNR; c++)
		P[c] = 0;
#endif
	
	
	// perform the matrix-matrix product as outer product 
	// of rows of A and B
	vx const* b = (vx const*)B;
	for (std::size_t l=0; l<kc; ++l) {
		for (std::size_t i=0; i<block_size::mr; ++i) {
			for (std::size_t j=0; j<vecNR; ++j) {
				P[i * vecNR+j] += A[i]*b[j];
			}
		}
		A += block_size::mr;
		b += vecNR;
	}
	//multiply with alpha if necessary
	if (alpha!=TC(1)) {
		for (std::size_t i=0; i<block_size::mr; ++i) {
			for (std::size_t j=0; j< vecNR; ++j) {
				P[i*vecNR+j] *= alpha;
			}
		}
	}
	
	//add result to C
	T const* p = (T const*) P;
	for (std::size_t i=0; i<block_size::mr; ++i) {
		for (std::size_t j=0; j<block_size::nr; ++j) {
			C[i*ldc+j] += p[i*block_size::nr+j];
		}
	}
}


// Macro Kernel for two densly packed Blocks
template <class T, class TC, class block_size>
void mgemm(
	std::size_t mc, std::size_t nc, std::size_t kc, TC alpha,
	T const* A, T const* B, TC *C,
	std::size_t ldc, block_size
){
	static std::size_t const MR = block_size::mr;
	static std::size_t const NR = block_size::nr;
	std::size_t const mp  = (mc+MR-1) / MR;
	std::size_t const np  = (nc+NR-1) / NR;
	
	for (std::size_t j=0; j<np; ++j) {
		std::size_t const nr = std::min(NR, nc - j*NR);

		for (std::size_t i=0; i<mp; ++i) {
			std::size_t const mr = std::min(MR, mc - i*MR);
			auto CBlockStart = C+i*MR*ldc+j*NR;
			if (mr==MR && nr==NR) {
				ugemm<block_size>(
					kc, alpha,
					&A[i*kc*MR], &B[j*kc*NR],
					CBlockStart, ldc
				);
			} else {
				TC CTempBlock[MR*NR];
				std::fill_n(CTempBlock, MR*NR, T(0));
				ugemm<block_size>(
					kc, alpha,
					&A[i*kc*MR], &B[j*kc*NR],
					CTempBlock, NR
				);
				
				for (std::size_t i0=0; i0<mr; ++i0){	
					for (std::size_t j0=0; j0<nr; ++j0) {
						CBlockStart[i0*ldc+j0] += CTempBlock[i0*NR+j0];
					}
				}
			}
		}
	}
}

//-- Packing blocks ------------------------------------------------------------
template <class E, class T, class block_size>
void pack_A(matrix_expression<E, cpu_tag> const& A, T* p, block_size)
{
	BOOST_ALIGN_ASSUME_ALIGNED(p, block_size::align);

	std::size_t const mc = A().size1();
	std::size_t const kc = A().size2();
	static std::size_t const MR = block_size::mr;
	const std::size_t mp = (mc+MR-1) / MR;

	std::size_t nu = 0;
	for (std::size_t l=0; l<mp; ++l) {
		for (std::size_t j=0; j<kc; ++j) {
			for (std::size_t i = l*MR; i < l*MR + MR; ++i,++nu) {
				p[nu] = (i<mc) ? A()(i,j) : T(0);
			}
		}
	}
}


template <class E, class T, class block_size>
void pack_B(matrix_expression<E, cpu_tag> const& B, T* p, block_size)
{
        BOOST_ALIGN_ASSUME_ALIGNED(p, block_size::align);

        std::size_t const kc = B ().size1();
        std::size_t const nc = B ().size2();
        static std::size_t const NR = block_size::nr;
        std::size_t const np = (nc+NR-1) / NR;

	std::size_t nu = 0;
        for (std::size_t l=0; l<np; ++l) {
		for (std::size_t i=0; i<kc; ++i) {
			for (std::size_t j = l*NR; j < l*NR + NR; ++j,++nu){
				p[nu] = (j<nc) ? B()(i,j) : T(0);
			}
		}
        }
}

//-- Dense gemm
template <class E1, class E2, class Mat, class Orientation1, class Orientation2>
void gemm_impl(
	matrix_expression<E1, cpu_tag> const& e1,
	matrix_expression<E2, cpu_tag> const& e2,
	matrix_expression<Mat, cpu_tag>& m,
	typename Mat::value_type alpha,
	row_major, Orientation1, Orientation2,
	dense_tag, dense_tag
){
	typedef typename std::common_type<
		typename E1::value_type, typename E2::value_type, typename Mat::value_type
	>::type value_type;
	
	typedef prod_block_size<
		typename std::common_type<typename E1::value_type, typename E2::value_type>::type
	> block_size;
	
	static const std::size_t MC = block_size::mc;
        static const std::size_t NC = block_size::nc;
	static const std::size_t KC = block_size::kc;
	
	//obtain uninitialized aligned storage
	boost::alignment::aligned_allocator<value_type,block_size::align> allocator;
	value_type* A = allocator.allocate(MC * KC);
	value_type* B = allocator.allocate(NC * KC);

        const std::size_t M = m().size1();
        const std::size_t N = m().size2();
        const std::size_t K = e1().size2 ();
        const std::size_t mb = (M+MC-1) / MC;
        const std::size_t nb = (N+NC-1) / NC;
        const std::size_t kb = (K+KC-1) / KC;
	
	auto storageM = m().raw_storage();
        auto C_ = storageM.values;
        const std::size_t ldc = storageM.leading_dimension;
        for (std::size_t j=0; j<nb; ++j) {
		std::size_t nc = std::min(NC, N - j*NC);

		for (std::size_t l=0; l<kb; ++l) {
			std::size_t kc = std::min(KC, K - l*KC);
			matrix_range<typename const_expression<E2>::type> Bs(e2(), l*KC, l*KC+kc, j*NC, j*NC+nc);
			pack_B(Bs, B, block_size());

			for (std::size_t i=0; i<mb; ++i) {
				std::size_t mc = std::min(MC, M - i*MC);
				matrix_range<typename const_expression<E1>::type> As(e1(), i*MC, i*MC+mc, l*KC, l*KC+kc);
				pack_A (As, A, block_size());

				mgemm(
					mc, nc, kc, alpha, A, B,
					&C_[i*MC*ldc+j*NC], ldc , block_size()
				);
			}
		}
	}
	//free storage
	allocator.deallocate(A,MC * KC);
	allocator.deallocate(B,NC * KC);
}


// Dense-Sparse gemm
template <class E1, class E2, class M, class Orientation>
void gemm_impl(
	matrix_expression<E1, cpu_tag> const& e1,
	matrix_expression<E2, cpu_tag> const& e2,
	matrix_expression<M, cpu_tag>& m,
	typename M::value_type alpha,
	row_major, row_major, Orientation,
	dense_tag, sparse_tag
){
	for (std::size_t i = 0; i != e1().size1(); ++i) {
		matrix_row<M> row_m(m(),i);
		matrix_row<typename const_expression<E1>::type> row_e1(e1(),i);
		matrix_transpose<typename const_expression<E2>::type> trans_e2(e2());
		kernels::gemv(trans_e2,row_e1,row_m,alpha);
	}
}

template <class E1, class E2, class M>
void gemm_impl(
	matrix_expression<E1, cpu_tag> const& e1,
	matrix_expression<E2, cpu_tag> const& e2,
	matrix_expression<M, cpu_tag>& m,
	typename M::value_type alpha,
	row_major, column_major, column_major,
	dense_tag, sparse_tag
){
	typedef matrix_transpose<M> Trans_M;
	typedef matrix_transpose<typename const_expression<E2>::type> Trans_E2;
	Trans_M trans_m(m());
	Trans_E2 trans_e2(e2());
	for (std::size_t j = 0; j != e2().size2(); ++j) {
		matrix_row<Trans_M> column_m(trans_m,j);
		matrix_row<Trans_E2> column_e2(trans_e2,j);
		kernels::gemv(e1,column_e2,column_m,alpha);
	}
}

template <class E1, class E2, class M>
void gemm_impl(
	matrix_expression<E1, cpu_tag> const& e1,
	matrix_expression<E2, cpu_tag> const& e2,
	matrix_expression<M, cpu_tag>& m,
	typename M::value_type alpha,
	row_major, column_major, row_major,
	dense_tag, sparse_tag
){
	for (std::size_t k = 0; k != e1().size2(); ++k) {
		for(std::size_t i = 0; i != e1().size1(); ++i){
			noalias(row(m,i)) += alpha * e1()(i,k) * row(e2,k);
		}
	}
}

// Sparse-Dense gemm
template <class E1, class E2, class M, class Orientation>
void gemm_impl(
	matrix_expression<E1, cpu_tag> const& e1,
	matrix_expression<E2, cpu_tag> const& e2,
	matrix_expression<M, cpu_tag>& m,
	typename M::value_type alpha,
	row_major, row_major, Orientation,
	sparse_tag, dense_tag
){
	for (std::size_t i = 0; i != e1().size1(); ++i) {
		matrix_row<M> row_m(m(),i);
		matrix_row<E1> row_e1(e1(),i);
		matrix_transpose<E2> trans_e2(e2());
		kernels::gemv(trans_e2,row_e1,row_m,alpha);
	}
}

template <class E1, class E2, class M>
void gemm_impl(
	matrix_expression<E1, cpu_tag> const& e1,
	matrix_expression<E2, cpu_tag> const& e2,
	matrix_expression<M, cpu_tag>& m,
	typename M::value_type alpha,
	row_major, column_major, column_major,
	sparse_tag, dense_tag
){
	typedef matrix_transpose<M> Trans_M;
	typedef matrix_transpose<typename const_expression<E2>::type> Trans_E2;
	Trans_M trans_m(m());
	Trans_E2 trans_e2(e2());
	for (std::size_t j = 0; j != e2().size2(); ++j) {
		matrix_row<Trans_M> column_m(trans_m,j);
		matrix_row<Trans_E2> column_e2(trans_e2,j);
		kernels::gemv(e1,column_e2,column_m,alpha);
	}
}

template <class E1, class E2, class M>
void gemm_impl(
	matrix_expression<E1, cpu_tag> const& e1,
	matrix_expression<E2, cpu_tag> const& e2,
	matrix_expression<M, cpu_tag>& m,
	typename M::value_type alpha,
	row_major, column_major, row_major,
	sparse_tag, dense_tag
){
	for (std::size_t k = 0; k != e1().size2(); ++k) {
		auto e1end = e1().column_end(k);
		for(auto e1pos = e1().column_begin(k); e1pos != e1end; ++e1pos){
			std::size_t i = e1pos.index();
			auto val = alpha * (*e1pos);
			noalias(row(m,i)) += val * row(e2,k);
		}
	}
}

// Sparse-Sparse gemm
template<class M, class E1, class E2>
void gemm_impl(
	matrix_expression<E1, cpu_tag> const& e1,
	matrix_expression<E2, cpu_tag> const& e2,
	matrix_expression<M, cpu_tag>& m,
	typename M::value_type alpha,
	row_major, row_major, row_major, 
	sparse_tag, sparse_tag
) {
	typedef typename M::value_type value_type;
	value_type zero = value_type();
	typename vector_temporary<E1>::type temporary(e2().size2(), zero);
	for (std::size_t i = 0; i != e1().size1(); ++i) {
		kernels::gemv(trans(e2),row(e1,i),temporary,alpha);
		for (std::size_t j = 0; j != temporary.size(); ++ j) {
			if (temporary(j) != zero) {
				m()(i, j) += temporary(j);//fixme: better use something like insert
				temporary(j) = zero;
			}
		}
	}
}

template<class M, class E1, class E2>
void gemm_impl(
	matrix_expression<E1, cpu_tag> const& e1,
	matrix_expression<E2, cpu_tag> const& e2,
	matrix_expression<M, cpu_tag>& m,
	typename M::value_type alpha,
	row_major, row_major, column_major, 
	sparse_tag, sparse_tag
) {
	typedef matrix_transpose<M> Trans_M;
	typedef matrix_transpose<typename const_expression<E2>::type> Trans_E2;
	Trans_M trans_m(m());
	Trans_E2 trans_e2(e2());
	for (std::size_t j = 0; j != e2().size2(); ++j) {
		matrix_row<Trans_M> column_m(trans_m,j);
		matrix_row<Trans_E2> column_e2(trans_e2,j);
		kernels::gemv(e1,column_e2,column_m,alpha);
	}
}

template <class E1, class E2, class M, class Orientation>
void gemm_impl(
	matrix_expression<E1, cpu_tag> const& e1,
	matrix_expression<E2, cpu_tag> const& e2,
	matrix_expression<M, cpu_tag>& m,
	typename M::value_type alpha,
	row_major, column_major, Orientation o,
	sparse_tag t1, sparse_tag t2
){
	typename transposed_matrix_temporary<E1>::type e1_trans(e1);
	gemm_impl(e1_trans,e2,m,alpha,row_major(),row_major(),o,t1,t2);
}

//column major result is transformed to row_major using A=B*C <=> A^T = C^T B^T
template<class M, class E1, class E2, class Orientation1, class Orientation2, class Tag1, class Tag2>
void gemm_impl(
	matrix_expression<E1, cpu_tag> const& e1,
	matrix_expression<E2, cpu_tag> const& e2,
	matrix_expression<M, cpu_tag>& m,
	typename M::value_type alpha,
	column_major, Orientation1, Orientation2,
	Tag1, Tag2
){
	matrix_transpose<M> transposedM(m());
	typedef typename Orientation1::transposed_orientation transpO1;
	typedef typename Orientation2::transposed_orientation transpO2;
	gemm_impl(trans(e2),trans(e1),transposedM,alpha,row_major(),transpO2(),transpO1(), Tag2(),Tag1());
}

//dispatcher
template<class M, class E1, class E2>
void gemm(
	matrix_expression<E1, cpu_tag> const& e1,
	matrix_expression<E2, cpu_tag> const& e2,
	matrix_expression<M, cpu_tag>& m,
	typename M::value_type alpha,
	boost::mpl::false_
) {
	SIZE_CHECK(m().size1() == e1().size1());
	SIZE_CHECK(m().size2() == e2().size2());
	
	typedef typename M::orientation ResultOrientation;
	typedef typename E1::orientation E1Orientation;
	typedef typename E2::orientation E2Orientation;
	typedef typename E1::evaluation_category::tag E1Tag;
	typedef typename E2::evaluation_category::tag E2Tag;
	
	gemm_impl(e1, e2, m,alpha,
		ResultOrientation(),E1Orientation(),E2Orientation(),
		E1Tag(),E2Tag()
	);
}

}}}

#endif
