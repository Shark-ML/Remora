/*!
 * 
 *
 * \brief       matrix-vector multiplication kernel
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
#ifndef REMORA_KERNELS_GEMV_HPP
#define REMORA_KERNELS_GEMV_HPP

#include "default/gemv.hpp"

// #ifdef REMORA_USE_CBLAS
// #include "cblas/gemv.hpp"
// #else
// if no bindings are included, we have to provide the default has_optimized_gemv 
// otherwise the binding will take care of this
namespace remora{ namespace bindings{
template<class M1, class M2, class M3>
struct  has_optimized_gemv
: public std::false_type{};
}}
// #endif
	
namespace remora{namespace kernels{
	
///\brief Well known GEneral Matrix-Vector product kernel v+=alpha*A*x.
///
/// If bindings are included and the matrix/vector combination allows for a specific binding
/// to be applied, the binding is called automatically from {binding}/gemv.h
/// otherwise default/gemv.h is used which is fully implemented for all dense/sparse combinations.

template<class VecV, class MatA, class VecX, class Device>
void gemv(
	matrix_expression<MatA, Device> const& A,
	vector_expression<VecX, Device> const& x,
	vector_expression<VecV, Device>& v,
	typename VecV::value_type alpha
) {
	assert(x().shape()[0] == A().shape()[1]);
	assert(v().shape()[0] == A().shape()[0]);
	/// if a combination is optimized, bindings::has_optimized_gemv<>::type evaluates to std::true_type
	/// The kernels themselves are implemented in bindings::gemv.
	bindings::gemv(
		A, x, v,alpha,
		typename MatA::axis(),
		typename bindings::has_optimized_gemv<VecV,MatA,VecX>::type()
	);
}

}}
#if defined(__HCC__) || defined(__NVCC__)
#include "hip/gemv.hpp"
#endif
#endif