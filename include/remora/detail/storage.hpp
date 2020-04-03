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
#include <type_traits>

namespace remora{
	
struct unknown_storage{
	typedef unknown_tag storage_tag;
};

template<std::size_t Dim, class T, class Tag>
struct dense_tensor_storage{
	typedef Tag storage_tag;
	template<unsigned N>
	using sub_tag = typename std::conditional< N == 0, Tag, dense_tag>::type;
	
	T* values;
	std::array<std::size_t, Dim> strides;
	
	dense_tensor_storage(){}
	dense_tensor_storage(T* values, std::array<std::size_t, Dim> const& strides):values(values),strides(strides){}
	template<class U, class Tag2>
	dense_tensor_storage(dense_tensor_storage<Dim, U, Tag2> const& storage): values(storage.values), strides(storage.strides){
		static_assert(!(std::is_same<Tag,continuous_dense_tag>::value && std::is_same<Tag2,dense_tag>::value), "Trying to assign dense to continuous dense storage");
	}
};
}

#endif
