/*!
 *
 *
 * \brief       matrix-matrix multiplication kernel
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

#ifndef REMORA_KERNELS_GEMM_HPP
#define REMORA_KERNELS_GEMM_HPP

//#include "default/gemm.hpp"

//#ifdef REMORA_USE_CBLAS
//#include "cblas/dense_gemm.hpp"
//#else
#include "default/dense_gemm.hpp"
//#endif*/

#include "../detail/proxy_optimizers_fwd.hpp"

namespace remora{
	
namespace bindings{
	//-- Dense gemm
	template <class E1, class E2, class Mat, class Axis1, class Axis2, class Device>
	void gemm(
		matrix_expression<E1, Device> const& e1,
		matrix_expression<E2, Device> const& e2,
		matrix_expression<Mat, Device>& m,
		typename Mat::value_type alpha,
		row_major, Axis1, Axis2,
		dense_tag, dense_tag
	){
		dense_gemm(e1,e2,m,alpha);
	}
	//column major result is transformed to row_major using A=B*C <=> A^T = C^T B^T
	template<class M, class E1, class E2, class Axis1, class Axis2, class Tag1, class Tag2, class Device>
	void gemm(
		matrix_expression<E1, Device> const& e1,
		matrix_expression<E2, Device> const& e2,
		matrix_expression<M, Device>& m,
		typename M::value_type alpha,
		column_major, Axis1, Axis2,
		Tag1, Tag2
	){
		auto transM = detail::axis_permute_optimizer<typename M::closure_type, axis<1,0> >::create(m());
		auto transE1 = detail::axis_permute_optimizer<typename E1::const_closure_type, axis<1,0> >::create(e1());
		auto transE2 = detail::axis_permute_optimizer<typename E2::const_closure_type, axis<1,0> >::create(e2());
		typedef typename Axis1::template swap_axes_t<1,0> trans_axis1;
		typedef typename Axis2::template swap_axes_t<1,0> trans_axis2;
		gemm(transE2,transE1,transM,alpha,row_major(),trans_axis2(),trans_axis1(), Tag2(),Tag1());
	}
}

namespace kernels{

///\brief Well known GEneral Matrix-Matrix product kernel M+=alpha*E1*E2.
///
/// If bindings are included and the matrix combination allow for a specific binding
/// to be applied, the binding is called automatically from {binding}/gemm.h
/// otherwise default/gemm.h is used which is fully implemented for all dense/sparse combinations.
/// if a combination is optimized, bindings::has_optimized_gemm<M,E1,E2>::type evaluates to std::true_type
/// The kernels themselves are implemented in bindings::gemm.
template<class M, class E1, class E2, class Device>
void gemm(
	matrix_expression<E1, Device> const& e1,
	matrix_expression<E2, Device> const& e2,
	matrix_expression<M, Device>& m,
	typename M::value_type alpha
) {
	REMORA_SIZE_CHECK(m().shape()[0] == e1().shape()[0]);
	REMORA_SIZE_CHECK(m().shape()[1] == e2().shape()[1]);
	REMORA_SIZE_CHECK(e1().shape()[1] == e2().shape()[0]);

	typedef typename M::axis ResultAxis;
	typedef typename E1::axis E1Axis;
	typedef typename E2::axis E2Axis;
	typedef typename E1::evaluation_category::tag E1Tag;
	typedef typename E2::evaluation_category::tag E2Tag;

	bindings::gemm(e1, e2, m ,alpha,
		ResultAxis(), E1Axis(), E2Axis(),
		E1Tag(),E2Tag()
	);
}

}}
//#if defined(__HCC__) || defined(__NVCC__)
//#include "hip/gemm.hpp"
//#endif

#endif
