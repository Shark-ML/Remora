/*!
 * 
 *
 * \brief       Implements tensor reductions
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
 * MERCHANTABILITY or FITNESS FOR E PARTICULAR PURPOSE.  See the
 * GNU Lesser General Public License for more details.
 * 
 * You should have received a copy of the GNU Lesser General Public License
 * along with Shark.  If not, see <http://www.gnu.org/licenses/>.
 *
 */

#ifndef REMORA_KERNELS_REDUCE_HPP
#define REMORA_KERNELS_REDUCE_HPP

#include "default/reduce.hpp"
#if defined(__HCC__) || defined(__NVCC__)
#include "hip/reduce.hpp"
#endif

namespace remora {


namespace kernels{
template <class TensorE, class TensorA, std::size_t NA, class F, class Device>
void reduce_last(
	tensor_expression<NA, TensorE, Device> const& E, 
	tensor_expression<NA - 1, TensorA, Device>& A,
	F f
){
	//permute A to standard layout, apply same permutation to E
	typedef typename TensorA::axis::inverse_t permute_A;
	auto A_permuted = permute(A, permute_A());
	auto E_permuted = permute(E, typename permute_A::expand_t());
	typedef typename TensorA::evaluation_category::tag ACategory;
	typedef typename TensorE::evaluation_category::tag ECategory;
	bindings::reduce_last(
		E_permuted, A_permuted, f, typename decltype(E_permuted)::axis(), ACategory(), ECategory()
	);
}

}}

#endif
