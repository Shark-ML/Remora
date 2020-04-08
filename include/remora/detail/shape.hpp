//===========================================================================
/*!
 * 
 *
 * \brief       Axis descriptor Traits-class
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
//===========================================================================

#ifndef REMORA_DETAIL_SHAPE_HPP
#define REMORA_DETAIL_SHAPE_HPP

#include <array>

namespace remora{

template<std::size_t Dim>
class tensor_shape{
public:
	tensor_shape(): shape_array(){}
	template<class ...Size, class = typename std::enable_if<sizeof...(Size) == Dim,void>::type >
	constexpr tensor_shape(Size... sizes): shape_array({std::size_t(sizes)...}){}
	std::size_t size()const{
		return shape_array.size();
	}
	std::size_t& operator[](std::size_t i){
		return shape_array[i];
	}
	
	std::size_t const& operator[](std::size_t i)const{
		return shape_array[i];
	}
	
	std::size_t num_elements()const{
		if (size() == 0)
			return 0;
		std::size_t num_elem = 1;
		for (std::size_t dim: shape_array){
			num_elem *= dim;
		}
		return num_elem;
	}

	
	bool operator==(tensor_shape<Dim> const& other)const{
		for (std::size_t i = 0; i != size(); ++i){
			if (shape_array[i] != other[i]){
				return false;
			}
		}
		return true;
	}
	bool operator!=(tensor_shape<Dim> const& other)const{
		return !(*this == other);
	}
	std::array<std::size_t, Dim> shape_array;
};

}
#endif