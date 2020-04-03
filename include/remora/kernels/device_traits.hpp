//===========================================================================
/*!
 * 
 *
 * \brief       Traits of matrix expressions
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

#ifndef REMORA_KERNELS_DEVICE_TRAITS_HPP
#define REMORA_KERNELS_DEVICE_TRAITS_HPP

#include "../expression_types.hpp"

namespace remora {
	

template<class Device>
struct device_traits;

//some devices do not need a queue but the interface still expects one.
struct no_queue{};
//for non-dense expression, the functor interface does not make sense but it is still expected to have elements()
struct no_functor{};

}

#include "default/device_traits.hpp"

#endif


