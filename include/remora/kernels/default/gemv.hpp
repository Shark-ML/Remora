/*!
 * 
 *
 * \brief       -
 *
 * \author      O. Krause
 * \date        2012
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
 * MatAERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU Lesser General Public License for more details.
 * 
 * You should have received a copy of the GNU Lesser General Public License
 * along with Shark.  If not, see <http://www.gnu.org/licenses/>.
 *
 */
#ifndef REMORA_KERNELS_DEFAULT_GEMatAVecV_HPP
#define REMORA_KERNELS_DEFAULT_GEMatAVecV_HPP

#include "../../expression_types.hpp" //matrix/vector_expression
#include "../../detail/proxy_optimizers_fwd.hpp"
#include "../../detail/traits.hpp" //matrix orientations
#include "dot.hpp" //inner product
#include "../assign.hpp" //assignment of vectors
#include <type_traits> //std::false_type marker for unoptimized

namespace remora{namespace bindings {
	
//row major can be further reduced to inner_prod()
template<class VecX, class MatA, class VecV>
void gemv(
	matrix_expression<MatA, cpu_tag> const& A,
	vector_expression<VecX, cpu_tag> const& x,
	vector_expression<VecV, cpu_tag>& v, 
	typename VecV::value_type alpha,
	row_major,
	std::false_type
){
	typedef typename VecX::value_type value_type;
	typedef typename VecV::size_type size_type;
	value_type value;
	size_type size = A().shape()[0];
	
	for(size_type i = 0; i != size;++i){
		auto A_i = detail::slice_optimizer<typename MatA::const_closure_type, 0>::create(A(), i);
		bindings::dot(A_i, x, value,  typename MatA::evaluation_category::tag(), typename VecV::evaluation_category::tag());
		if(value != value_type())//handling of sparse vs.
			v()(i) += alpha * value;
	}
}

//column major is implemented by computing a linear combination of matrix-rows 
template<class VecX, class MatA, class VecV>
void gemv(
	matrix_expression<MatA, cpu_tag> const& A,
	vector_expression<VecX, cpu_tag> const& x,
	vector_expression<VecV, cpu_tag>& v,
	typename VecV::value_type alpha,
	column_major,
	std::false_type
) {
	typedef typename VecV::size_type size_type;
	typedef typename VecX::value_type value_type;
	typedef device_traits<cpu_tag>::multiply_and_add<value_type> MultAdd;
	size_type size = A().shape()[1];
	for(size_type j = 0; j != size; ++j){
		auto A_j = detail::slice_optimizer<typename MatA::const_closure_type, 1>::create(A(), j);
		kernels::assign(v, A_j, MultAdd(alpha * x()(j)));
	}
}

}}
#endif
