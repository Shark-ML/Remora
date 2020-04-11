//===========================================================================
/*!
 * 
 *
 * \brief       Storage Types of matrix expressions
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
//===========================================================================

#ifndef REMORA_DETAIL_STORAGE_HPP
#define REMORA_DETAIL_STORAGE_HPP

#include "structure.hpp"
#include "integer_list.hpp"
#include <type_traits>

namespace remora{
	
struct unknown_storage{
	typedef unknown_tag storage_tag;
};

namespace detail{
	template<bool... First, bool... Second>
	constexpr bool is_compatible_dense_storage_helper(integer_list<bool, First...>, integer_list<bool, Second...>){
		bool arr1[]={First...};
		bool arr2[]={Second...};
		std::size_t N = sizeof...(First);
		bool compatible = true;
		for(std::size_t i = 0; i != N; ++i){
			if(arr1[i] == true && arr2[i] == false){
				compatible = false;
				break;
			}
		}
		return compatible;
	};
}

template<class T, class DenseAxisTag>
struct dense_tensor_storage{
	typedef DenseAxisTag dense_axis_tag;
	
	T* values;
	std::array<std::size_t, dense_axis_tag::num_dims> strides;
	
	dense_tensor_storage(){}
	dense_tensor_storage(T* values, std::array<std::size_t, dense_axis_tag::num_dims> const& strides):values(values),strides(strides){
	}
	template<class U, class Tag, class = typename std::enable_if<detail::is_compatible_dense_storage_helper(dense_axis_tag(), Tag()) , void>::type >
	dense_tensor_storage(dense_tensor_storage<U, Tag> const& storage): values(storage.values), strides(storage.strides){}
};

namespace detail{
	template<unsigned... Ns>
	integer_list<bool, ((void)Ns,true)...> make_continuous_tensor_storage(std::integer_sequence<unsigned, Ns...>);
}

template<std::size_t N, class T>
using continuous_tensor_storage = dense_tensor_storage<T, constant_integer_list<bool, true, N> >;

}

#endif
