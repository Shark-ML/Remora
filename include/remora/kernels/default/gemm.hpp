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

#ifndef REMORA_KERNELS_DEFAULT_GEMM_HPP
#define REMORA_KERNELS_DEFAULT_GEMM_HPP

#include "../gemv.hpp"//for dispatching to gemv
#include "../../assignment.hpp"//plus_assign
#include "../../vector.hpp"//sparse gemm needs temporary vector
#include "../../detail/matrix_proxy_classes.hpp"//matrix row,column,transpose,range
#include "mgemm.hpp" //block macro kernel for dense gemm
#include <boost/mpl/bool.hpp> //boost::mpl::false_ marker for unoptimized

#include <type_traits> //std::common_type


namespace remora{namespace bindings {

//  Dense Block-GEMM implementation based on boost.ublas
//  written by:
//  Copyright (c) 2016
//  Michael Lehn, Imre Palik
//
//  Distributed under the Boost Software License, Version 1.0. (See
//  accompanying file LICENSE_1_0.txt or copy at
//  http://www.boost.org/LICENSE_1_0.txt)

template <typename T>
struct gemm_block_size {
	typedef detail::block<T> block;
	static const unsigned mr = 4; // stripe width for lhs
	static const unsigned nr = 3 * block::max_vector_elements; // stripe width for rhs
	static const unsigned mc = 128;
	static const unsigned kc = 512; // stripe length
	static const unsigned nc = (1024/nr) * nr;
};

template <>
struct gemm_block_size<float> {
	typedef detail::block<float> block;
	static const unsigned mc = 256;
	static const unsigned kc = 512; // stripe length
	static const unsigned nc = 4096;
	static const unsigned mr = 4; // stripe width for lhs
	static const unsigned nr = 16; // stripe width for rhs
};

template <>
struct gemm_block_size<long double> {
	typedef detail::block<long double> block;
	static const unsigned mc = 256;
	static const unsigned kc = 512; // stripe length
	static const unsigned nc = 4096;
	static const unsigned mr = 1; // stripe width for lhs
	static const unsigned nr = 4; // stripe width for rhs
};

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
			pack_B_dense(Bs, B, block_size());

			for (std::size_t i=0; i<mb; ++i) {
				std::size_t mc = std::min(MC, M - i*MC);
				matrix_range<typename const_expression<E1>::type> As(e1(), i*MC, i*MC+mc, l*KC, l*KC+kc);
				pack_A_dense(As, A, block_size());

				mgemm(
					mc, nc, kc, alpha, A, B,
					&C_[i*MC*ldc+j*NC], ldc , 1, block_size()
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
			matrix_row<M> row_m(m(),i);
			matrix_row<typename const_expression<E2>::type> row_e2(e2(),k);
			plus_assign(row_m,row_e2,alpha * e1()(i,k));
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
			matrix_row<M> row_m(m(),i);
			matrix_row<typename const_expression<E2>::type> row_e2(e2(),k);
			plus_assign(row_m,row_e2,alpha * (*e1pos));
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
	vector<value_type> temporary(e2().size2(), zero);//dense vector for quick random access
	matrix_transpose<typename const_expression<E2>::type> e2trans(e2());
	for (std::size_t i = 0; i != e1().size1(); ++i) {
		matrix_row<typename const_expression<E1>::type> rowe1(e1(),i);
		kernels::gemv(e2trans,rowe1,temporary,alpha);
		auto insert_pos = m().row_begin(i);
		for (std::size_t j = 0; j != temporary.size(); ++ j) {
			if (temporary(j) != zero) {
				//find element with that index
				auto row_end = m().row_end(i);
				while(insert_pos != row_end && insert_pos.index() < j)
					++insert_pos;
				//check if element exists
				if(insert_pos != row_end && insert_pos.index() == j){
					*insert_pos += temporary(j);
				}else{//create new element
					insert_pos = m().set_element(insert_pos,j,temporary(j));
				}
				//~ m()(i,j) += temporary(j);
				temporary(j) = zero; // delete element
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
	//best way to compute this is to transpose e1 in memory. alternative would be
	// to compute outer products, which is a no-no.
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
	matrix_transpose<E1 const> e1trans(e1());
	matrix_transpose<E2 const> e2trans(e2());
	gemm_impl(e2trans,e1trans,transposedM,alpha,row_major(),transpO2(),transpO1(), Tag2(),Tag1());
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

}}

#endif
