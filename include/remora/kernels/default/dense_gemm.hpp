/*!
 *
 *
 * \brief       dense matrix matrix multiplication implementation
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

#ifndef REMORA_KERNELS_DEFAULT_DENSE_GEMM_HPP
#define REMORA_KERNELS_DEFAULT_DENSE_GEMM_HPP

#include "../../detail/proxy_optimizers_fwd.hpp" //for subrange
#include <type_traits> //std::common_type
#include "simd.hpp"
#include <algorithm>//std::fill

namespace remora{namespace bindings{

//  Block-GEMM implementation based on boost.ublas
//  written by:
//  Copyright (c) 2016
//  Michael Lehn, Imre Palik
//
//  Distributed under the Boost Software License, Version 1.0. (See
//  copy at http://www.boost.org/LICENSE_1_0.txt)

//-- Micro Kernel For Dense operations----------------------------------------------------------
template <class block_size, class T, class TC>
void ugemm(
	std::size_t kc, TC alpha, T const* A, T const* B,
	TC* C, std::size_t stride1, std::size_t stride2
){
	BOOST_ALIGN_ASSUME_ALIGNED(A, block_size::block::align);
	BOOST_ALIGN_ASSUME_ALIGNED(B, block_size::block::align);

	typedef typename block_size::block::type vx;
	static const std::size_t vector_length = block_size::block::vector_elements;
	static const std::size_t vecNR = block_size::nr/vector_length;
#ifdef REMORA_USE_SIMD
	vx P[block_size::mr * vecNR] = {};
#else
	typename std::aligned_storage<sizeof(vx[block_size::mr*vecNR]),block_size::block::align>::type Pa;
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
			C[i * stride1 + j * stride2] += p[i*block_size::nr+j];
		}
	}
}


// Macro Kernel for two densly packed Blocks
template <class T, class TC, class block_size>
void mgemm(
	std::size_t mc, std::size_t nc, std::size_t kc, TC alpha,
	T const* A, T const* B, TC *C,
	std::size_t stride1, std::size_t stride2, block_size
){
	static std::size_t const MR = block_size::mr;
	static std::size_t const NR = block_size::nr;
	std::size_t const mp  = (mc+MR-1) / MR;
	std::size_t const np  = (nc+NR-1) / NR;

	for (std::size_t j=0; j<np; ++j) {
		std::size_t const nr = std::min(NR, nc - j*NR);

		for (std::size_t i=0; i<mp; ++i) {
			std::size_t const mr = std::min(MR, mc - i*MR);
			auto CBlockStart = C+i*MR*stride1+j*NR*stride2;
			if (mr==MR && nr==NR) {
				ugemm<block_size>(
					kc, alpha,
					&A[i*kc*MR], &B[j*kc*NR],
					CBlockStart, stride1, stride2
				);
			} else {
				TC CTempBlock[MR*NR];
				std::fill_n(CTempBlock, MR*NR, T(0));
				ugemm<block_size>(
					kc, alpha,
					&A[i*kc*MR], &B[j*kc*NR],
					CTempBlock, NR, 1
				);

				for (std::size_t i0=0; i0<mr; ++i0){
					for (std::size_t j0=0; j0<nr; ++j0) {
						CBlockStart[i0*stride1 + j0 * stride2] += CTempBlock[i0*NR+j0];
					}
				}
			}
		}
	}
}


//-- Packing blocks ------------------------------------------------------------
template <class E, class T, class block_size>
void pack_A_dense(matrix_expression<E, cpu_tag> const& A, T* p, block_size){
	BOOST_ALIGN_ASSUME_ALIGNED(p, block_size::block::align);
	auto A_elem = A().elements();

	std::size_t const mc = A().shape()[0];
	std::size_t const kc = A().shape()[1];
	static std::size_t const MR = block_size::mr;
	const std::size_t mp = (mc+MR-1) / MR;

	std::size_t nu = 0;
	for (std::size_t l=0; l<mp; ++l) {
		for (std::size_t j=0; j<kc; ++j) {
			for (std::size_t i = l*MR; i < l*MR + MR; ++i,++nu) {
				p[nu] = (i<mc) ? A_elem(i,j) : T(0);
			}
		}
	}
}


template <class E, class T, class block_size>
void pack_B_dense(matrix_expression<E, cpu_tag> const& B, T* p, block_size)
{
	BOOST_ALIGN_ASSUME_ALIGNED(p, block_size::block::align);
	auto B_elem = B().elements();

	std::size_t const kc = B ().shape()[0];
	std::size_t const nc = B ().shape()[1];
	static std::size_t const NR = block_size::nr;
	std::size_t const np = (nc+NR-1) / NR;

	std::size_t nu = 0;
	for (std::size_t l=0; l<np; ++l) {
		for (std::size_t i=0; i<kc; ++i) {
			for (std::size_t j = l*NR; j < l*NR + NR; ++j,++nu){
				p[nu] = (j<nc) ? B_elem(i,j) : T(0);
			}
		}
	}
}
template <typename T>
struct gemm_block_size {
	typedef bindings::block<T> block;
	static const unsigned mr = 4; // stripe width for lhs
	static const unsigned nr = 3 * block::max_vector_elements; // stripe width for rhs
	static const unsigned mc = 128;
	static const unsigned kc = 512; // stripe length
	static const unsigned nc = (1024/nr) * nr;
};

template <>
struct gemm_block_size<float> {
	typedef bindings::block<float> block;
	static const unsigned mr = 4; // stripe width for lhs
	static const unsigned nr = 16; // stripe width for rhs
	static const unsigned mc = 256;
	static const unsigned kc = 512; // stripe length
	static const unsigned nc = 4096;
	
};

template <>
struct gemm_block_size<long double> {
	typedef bindings::block<long double> block;
	static const unsigned mr = 1; // stripe width for lhs
	static const unsigned nr = 4; // stripe width for rhs
	static const unsigned mc = 256;
	static const unsigned kc = 512; // stripe length
	static const unsigned nc = 4096;
	
};

//-- Dense gemm
template <class E1, class E2, class Mat>
void dense_gemm(
	matrix_expression<E1, cpu_tag> const& e1,
	matrix_expression<E2, cpu_tag> const& e2,
	matrix_expression<Mat, cpu_tag>& m,
	typename Mat::value_type alpha
){
	static_assert(std::is_same<typename Mat::axis,row_major>::value,"target matrix must be row major");
	typedef typename std::common_type<
		typename E1::value_type, typename E2::value_type, typename Mat::value_type
	>::type value_type;

	typedef gemm_block_size<
		typename std::common_type<typename E1::value_type, typename E2::value_type>::type
	> block_size;

	static const std::size_t MC = block_size::mc;
	static const std::size_t NC = block_size::nc;
	static const std::size_t KC = block_size::kc;

	//obtain uninitialized aligned storage
	boost::alignment::aligned_allocator<value_type,block_size::block::align> allocator;
	value_type* A = allocator.allocate(MC * KC);
	value_type* B = allocator.allocate(NC * KC);

	const std::size_t M = m().shape()[0];
	const std::size_t N = m().shape()[1];
	const std::size_t K = e1().shape()[1];
	const std::size_t mb = (M+MC-1) / MC;
	const std::size_t nb = (N+NC-1) / NC;
	const std::size_t kb = (K+KC-1) / KC;

	auto storageM = m().raw_storage();
	auto C_ = storageM.values;
	auto strides = storageM.strides;
	for (std::size_t j=0; j<nb; ++j) {
		std::size_t nc = std::min(NC, N - j*NC);

		for (std::size_t l=0; l<kb; ++l) {
			std::size_t kc = std::min(KC, K - l*KC);
			auto Bs0 = detail::subrange_optimizer<typename E2::const_closure_type,0>::create(e2(), l * KC, l * KC + kc);
			auto Bs = detail::subrange_optimizer<decltype(Bs0),1>::create(Bs0, j * NC, j * NC + nc);
			pack_B_dense(Bs, B, block_size());

			for (std::size_t i=0; i<mb; ++i) {
				std::size_t mc = std::min(MC, M - i*MC);
				auto As0 = remora::detail::subrange_optimizer<typename E1::const_closure_type,0>::create(e1(), i * MC, i * MC + mc);
				auto As = remora::detail::subrange_optimizer<decltype(As0),1>::create(As0, l*KC, l*KC + kc);
				pack_A_dense(As, A, block_size());
				mgemm(
					mc, nc, kc, alpha, A, B,
					(C_ + i*MC*strides[0]+j*strides[1]*NC), strides[0] , strides[1], block_size()
				);
			}
		}
	}
	//free storage
	allocator.deallocate(A, MC * KC);
	allocator.deallocate(B, NC * KC);
}

}}
#endif
