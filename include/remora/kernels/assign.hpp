/*!
 * \brief       Kernels for matrix-expression assignments
 * 
 * \author      O. Krause
 * \date        2013
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
#ifndef REMORA_KERNELS_ASSIGN_HPP
#define REMORA_KERNELS_ASSIGN_HPP

#include "default/assign.hpp"
#include "../proxy_expressions.hpp"
#ifdef REMORA_USE_OPENCL
#include "opencl/assign.hpp"
#endif
#if defined(__HCC__) || defined(__NVCC__)
#include "hip/assign.hpp"
#endif



#include <type_traits>

namespace remora {namespace kernels{
	
//////////////////////////////////////////////////////
////Apply a functor to all matrix-elements
/////////////////////////////////////////////////////
	
template<std::size_t N, class Functor, class TensorA, class Device>
void apply(
	tensor_expression<N, TensorA, Device>& A, 
	Functor const& f
){
	if(A().shape().num_elements() == 0) return;
	typedef typename TensorA::evaluation_category::tag ACategory;
	typedef typename TensorA::axis::inverse_t permutation;
	auto Apermuted = permute(A, permutation());
	bindings::apply(Apermuted, f, default_axis<N>(), ACategory() );
}

/////////////////////////////////////////////////////////////////
//////Matrix Assignment implementing op=
////////////////////////////////////////////////////////////////

// Dispatcher
template<std::size_t N, class TensorA, class TensorE, class Device>
void assign(
	tensor_expression<N, TensorA, Device>& A,
	tensor_expression<N, TensorE, Device> const& E
){
	REMORA_SIZE_CHECK(A().shape() == E().shape());
	if(A().shape().num_elements() == 0) return;
	
	typedef typename TensorA::evaluation_category::tag ACategory;
	typedef typename TensorE::evaluation_category::tag ECategory;
	
	//standardize axis components:
	//by default we want A to be standard-axis.
	//if E is not dense, we change this to E as standard-axis.
	typedef typename TensorA::axis::inverse_t permutation_A;
	typedef typename TensorE::axis::inverse_t permutation_E;
	typedef typename std::conditional<std::is_same<ECategory, dense_tag>::value, permutation_A, permutation_E>::type permutation;
	auto Apermuted = permute(A, permutation());
	auto Epermuted = permute(E, permutation());

	bindings::assign(Apermuted, Epermuted, default_axis<N>(), typename decltype(Epermuted)::axis(), ACategory(), ECategory());
}


///////////////////////////////////////////////////////////////////////////////////////////
//////Matrix Assignment With Functor implementing +=,-=...
///////////////////////////////////////////////////////////////////////////////////////////



//First Level Dispatcher, dispatches by orientation
template<std::size_t N, class Functor, class TensorA, class TensorE, class Device>
void assign(
	tensor_expression<N, TensorA, Device>& A, 
	tensor_expression<N, TensorE, Device> const& E,
	Functor const& f
){
	REMORA_SIZE_CHECK(A().shape() == E().shape());
	if(A().shape().num_elements() == 0) return;
	
	typedef typename TensorA::evaluation_category::tag ACategory;
	typedef typename TensorE::evaluation_category::tag ECategory;
	
	//standardize axis components:
	//by default we want A to be standard-axis.
	//if E is not dense, we change this to E as standard-axis.
	typedef typename TensorA::axis::inverse_t permutation_A;
	typedef typename TensorE::axis::inverse_t permutation_E;
	typedef typename std::conditional<std::is_same<ECategory, dense_tag>::value, permutation_A, permutation_E>::type permutation;
	auto Apermuted = permute(A, permutation());
	auto Epermuted = permute(E, permutation());
	
	bindings::assign_functor(Apermuted, Epermuted, f, default_axis<N>(), typename decltype(Epermuted)::axis(), ACategory(), ECategory());
}

}}

#endif
