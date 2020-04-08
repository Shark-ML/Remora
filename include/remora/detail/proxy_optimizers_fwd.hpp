/*!
 * \brief       Proxy Optimizations
 * 
 * \author      O. Krause
 * \date        2016
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
#ifndef REMORA_DETAIL_PROXY_OPTIMIZERS_FWD_HPP
#define REMORA_DETAIL_PROXY_OPTIMIZERS_FWD_HPP

#include <cstddef> //std::size_t

namespace remora{namespace detail{
	
template<class TensorA, std::size_t N>
struct axis_split_optimizer;

template<class TensorA, std::size_t N>
struct axis_merge_optimizer;
	
template<class TensorA, std::size_t N>
struct subrange_optimizer;

template<class TensorA, std::size_t N>
struct slice_optimizer;
	
template<class TensorA, class Axis>
struct axis_permute_optimizer;
    
// template<class M>
// struct matrix_diagonal_optimizer;

template<class M, class Tag>
struct triangular_proxy_optimizer;

template<class M>
struct scalar_multiply_optimizer;

}}
#endif
