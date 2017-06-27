/*!
 *  \brief Low level proxy generating functions
 *
 *
 *  \author  O. Krause
 *  \date    2016
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
#ifndef REMORA_KERNELS_DEFAULT_PROXIES_HPP
#define REMORA_KERNELS_DEFAULT_PROXIES_HPP

#include "../../detail/matrix_proxy_classes.hpp"
#include "../../detail/vector_proxy_classes.hpp"

// about the purpose of this file:
// expression templates are using an expression optimization mechansism
// which binds proxies with all kinds of block expressions
// therefore those proxy generators can not be used for defining
// kernels, as this would generate cyclic include dependencies.
// instead we generate "simple" versions of the proxies here

namespace remora{namespace bindings {

//vector subrange
template<class V, class Device>
vector_range<typename const_expression<V>::type>
simple_subrange(vector_expression<V, Device> const& expression, std::size_t start, std::size_t stop){
	return vector_range<typename const_expression<V>::type>(expression(), start, stop);
}

template<class V, class Device>
vector_range<V>
simple_subrange(vector_expression<V, Device>& expression, std::size_t start, std::size_t stop){
	return vector_range<V>(expression(), start, stop);
}

//matrix subrange
template<class M, class Device>
matrix_range<typename const_expression<M>::type>
simple_subrange(matrix_expression<M, Device> const& expression, 
	std::size_t start1, std::size_t stop1, std::size_t start2, std::size_t stop2 ){
	return matrix_range<typename const_expression<M>::type>(expression(), start1, stop1, start2, stop2);
}

template<class M, class Device>
matrix_range<M>
simple_subrange(matrix_expression<M, Device>& expression, 
	std::size_t start1, std::size_t stop1, std::size_t start2, std::size_t stop2 ){
	return matrix_range<M>(expression(), start1, stop1, start2, stop2);
}

//matrix row
template<class M, class Device>
matrix_row<typename const_expression<M>::type>
simple_row(matrix_expression<M, Device> const& expression, typename M::size_type i){
	return matrix_row<typename const_expression<M>::type>(expression(), i);
}
template<class M, class Device>
matrix_row<M>
simple_row(matrix_expression<M, Device>& expression, typename M::size_type i){
	return matrix_row<M>(expression(), i);
}

//matrix column
template<class M, class Device>
matrix_row< matrix_transpose< typename const_expression<M>::type > >
simple_column(matrix_expression<M, Device> const& expression, typename M::size_type i){
	typedef matrix_transpose< typename const_expression<M>::type > TransE;
	TransE transE(expression());
	return matrix_row< TransE >(transE, i);
}
template<class M, class Device>
matrix_row< matrix_transpose<M> >
simple_column(matrix_expression<M, Device>& expression, typename M::size_type i){
	typedef matrix_transpose<M> TransE;
	TransE transE(expression());
	return matrix_row< TransE >(transE, i);
}

//simple trans
template<class M, class Device>
matrix_transpose< typename const_expression<M>::type >
simple_trans(matrix_expression<M, Device>const& m){
	return matrix_transpose< typename const_expression<M>::type >(m());
}
template<class M, class Device>
matrix_transpose<M>
simple_trans(matrix_expression<M, Device>& m){
	return matrix_transpose<M>(m());
}

}}
#endif