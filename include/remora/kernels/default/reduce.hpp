/*!
 * 
 *
 * \brief       Reduces a tensor
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

#ifndef REMORA_KERNELS_DEFAULT_REDUCE_HPP
#define REMORA_KERNELS_DEFAULT_REDUCE_HPP

#include "../../proxy_expressions.hpp"//for slice
#include "../../detail/traits.hpp"

namespace remora{namespace bindings{


template <class TensorE, class TensorA, class F>
void reduce_last(
	tensor_expression<2, TensorE, cpu_tag> const& E, 
	tensor_expression<1, TensorA, cpu_tag>& A,
	F f,
	axis<0,1>, dense_tag, dense_tag
){
	auto shape = E().shape();
	auto E_elem = E().elements();
	for(std::size_t i = 0; i != shape[0]; ++i){
		typename TensorE::value_type s = E_elem(i,std::size_t(0));
		for(std::size_t j = 1; j != shape[1]; ++j){
			s = f(s,E_elem(i,j));
		}
		A()(i) += s;
	}
}

template <class TensorE, class TensorA, class F>
void reduce_last(
	tensor_expression<2, TensorE, cpu_tag> const& E, 
	tensor_expression<1, TensorA, cpu_tag>& A,
	F f,
	axis<1, 0>, dense_tag, dense_tag
){
	auto shape = E().shape();
	
	std::size_t n = shape[0];
	const std::size_t BLOCK_SIZE = 16;
	typename TensorA::value_type storage[BLOCK_SIZE];
	std::size_t numBlocks = (n + BLOCK_SIZE - 1) / BLOCK_SIZE; 
	auto E_elem = E().elements();
	for(std::size_t b = 0; b != numBlocks; ++b){
		std::size_t start = b * BLOCK_SIZE;
		std::size_t cur_size = std::min(BLOCK_SIZE, n - start);
		for(std::size_t i = 0; i != cur_size; ++i){
			storage[i] = E_elem(start + i, std::size_t(0));
		}
		for(std::size_t j = 1; j != shape[1]; ++j){
			for(std::size_t i = 0; i != cur_size; ++i){
				storage[i] = f(storage[i], E_elem(start + i, j));
			}
		}
		for(std::size_t i = 0; i != cur_size; ++i){
			A()(start + i) += storage[i];
		}
	}
}


template <class TensorE, class TensorA, std::size_t N, class F, class Axis, class Tag1, class Tag2>
void reduce_last(
	tensor_expression<N, TensorE, cpu_tag> const& E, 
	tensor_expression<N - 1, TensorA, cpu_tag>& A,
	F f,
	Axis, Tag1 t1, Tag2 t2
){
	std::size_t size = E().shape()[0];
	for(std::size_t i = 0; i != size; ++i){
		auto Aslice = slice(A,i);
		auto Eslice = slice(E,i);
		reduce_last(Eslice, Aslice, f, typename Axis::template slice_t<0>(),t1,t2);
	}
}

}}

#endif
