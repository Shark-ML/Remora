//===========================================================================
/*!
 * 
 *
 * \brief       Traits of matrix expressions
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

#ifndef REMORA_DETAIL_TRAITS_HPP
#define REMORA_DETAIL_TRAITS_HPP

#include "evaluation_tags.hpp"
#include "structure.hpp"
#include "storage.hpp"
#include "shape.hpp"
#include "axis.hpp"

#include <complex>
#include <type_traits>

namespace remora {
	
template<class T>
struct real_traits{
	typedef T type;
};

template<class T>
struct real_traits<std::complex<T> >{
	typedef T type;
};

template<class E1, class E2>
struct common_value_type
: public std::common_type<
	typename E1::value_type,
	typename E2::value_type
>{};

template<class E>
struct closure: public std::conditional<
	std::is_const<E>::value,
	typename E::const_closure_type,
	typename E::closure_type
>{};

template<class E>
struct reference: public std::conditional<
	std::is_const<E>::value,
	typename E::const_reference,
	typename E::reference
>{};

template<class E>
struct storage: public std::conditional<
	std::is_const<E>::value,
	typename E::const_storage_type,
	typename E::storage_type
>{};

///\brief Determines a good tensor type storing an expression returning values of type T having a certain evaluation category on a specific device.
template<class ValueType, class Axis, class Category, class Device>
struct tensor_temporary_type;

/// For the creation of temporary tensor in the assignment of proxies
template <class E>
struct tensor_temporary{
	typedef typename tensor_temporary_type<
		typename E::value_type,
		typename E::axis,
		typename E::evaluation_category::tag,
		typename E::device_type
	>::type type;
};

}

#endif


