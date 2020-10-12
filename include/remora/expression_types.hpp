/*!
 * \brief       Defines the basic types of CRTP base-classes
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
#ifndef REMORA_EXPRESSION_TYPE_HPP
#define REMORA_EXPRESSION_TYPE_HPP

#include <array>
namespace remora{
	
	
struct cpu_tag{};
struct hip_tag{};

template<std::size_t Dim, class T, class Device>
struct tensor_expression{
	typedef Device device_type;
	static constexpr std::size_t num_dims = Dim;
	T const& operator()() const {
		return *static_cast<T const*>(this);
	}

	T& operator()() {
		return *static_cast<T*>(this);
	}
};

template<std::size_t Dim, class T, class Device>
struct tensor_container: public tensor_expression<Dim, T, Device>{};

/// \brief Base class for Scalar Expressions
///
/// it does not model the Scalar Expression concept but all derived types should.
/// The class defines a common base type and some common interface for all
/// statically derived Scalar Expression classes.
template<class S, class Device>
using scalar_expression = tensor_expression<0, S, Device>;
	
/// \brief Base class for Vector Expression models
///
/// it does not model the Vector Expression concept but all derived types should.
/// The class defines a common base type and some common interface for all
/// statically derived Vector Expression classes.
template<class V, class Device>
using vector_expression = tensor_expression<1, V, Device>;

/// \brief Base class for Matrix Expression models
///
/// it does not model the Matrix Expression concept but all derived types should.
/// The class defines a common base type and some common interface for all
/// statically derived Matrix Expression classes.
template<class M, class Device>
using matrix_expression = tensor_expression<2, M, Device>;



}

#endif
