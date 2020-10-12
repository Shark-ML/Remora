/*!
 * \brief       Kernels for matrix-expression assignments
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
#ifndef REMORA_KERNELS_DEFAULT_ASSIGN_HPP
#define REMORA_KERNELS_DEFAULT_ASSIGN_HPP

#include "../../proxy_expressions.hpp"
#include "../../detail/traits.hpp"
#include "../device_traits.hpp"
#include <algorithm>
#include <vector>
namespace remora{namespace bindings{
	
template<class F, class TensorA>
void apply(
	tensor_expression<0, TensorA, cpu_tag>& A,
	F const& f,
	axis<>,
	dense_tag
){
	A()() = f(A()());
}

template<class F, class TensorA>
void apply(
	tensor_expression<1, TensorA, cpu_tag>& A,
	F const& f,
	axis<0>,
	dense_tag
){
	auto size = A().shape()[0];
	for(std::size_t i = 0; i != size; ++i){
		A()(i) = f(A()(i));
	}
}

//todo: versions for sparse and packed!

template<std::size_t N, class F, class TensorA, unsigned N0, unsigned... Ns, class Tag>
void apply(
	tensor_expression<N, TensorA, cpu_tag>& A,
	F const& f,
	axis<N0, Ns...>,
	Tag t
){
	typedef typename TensorA::size_type size_type;
	unsigned map[] = {Ns...};
	auto shape = A().shape();
	std::array<size_type, N> index={0};
	size_type num_slices = shape.num_elements()/shape[N - 1];
	
	for(size_type s = 0; s != num_slices; ++s){
		// assign the current slice
		auto slice_A = slice(A,std::get<Ns>(index)...);
		remora::bindings::apply(slice_A, f, axis<0>(), t);
		// increment index
		for (size_type j = N - 1; j > 0; --j){
			size_type i = j - 1;
			index[map[i]] += 1;
			if(index[map[i]] == shape[i]){
				index[map[i]] = 0;
			}else{
				break;
			}
		}
	}
}


///////////////////////////////////////////////////////////////////////////////////////////
//////Matrix Assignment With Functor implementing +=,-=...
///////////////////////////////////////////////////////////////////////////////////////////


template<class TensorA, class TensorE, class Functor>
void assign_functor(
	tensor_expression<0, TensorA, cpu_tag>& A, 
	tensor_expression<0, TensorE, cpu_tag> const& E,
	Functor f,
	axis<>, axis<>, dense_tag, dense_tag
){
	auto e_elem = E().elements();
	A()() = f(A()(), e_elem());
}
template<class TensorA, class TensorE, class Functor>
void assign_functor(
	tensor_expression<1, TensorA, cpu_tag>& A, 
	tensor_expression<1, TensorE, cpu_tag> const& E,
	Functor f,
	axis<0>, axis<0>, dense_tag, dense_tag
){
	typedef typename TensorA::size_type size_type;
	size_type size = A().shape()[0];
	auto e_elem = E().elements();
	for(size_type i = 0; i != size; ++i){
		A()(i) = f(A()(i), e_elem(i));
	}
}

template<class TensorA, class TensorE, class Functor>
void assign_functor(
	tensor_expression<2, TensorA, cpu_tag>& A, 
	tensor_expression<2, TensorE, cpu_tag> const& E,
	Functor f,
	axis<0,1>, axis<0, 1>, dense_tag, dense_tag
){
	typedef typename TensorA::size_type size_type;
	size_type size1 = A().shape()[0];
	size_type size2 = A().shape()[1];
	auto e_elem = E().elements();
	for (size_type i = 0; i < size1; ++i){
		for (size_type j = 0; j < size2; ++j){
			A()(i,j) = f(A()(i,j), e_elem(i,j));
		}
	}
}

template<class TensorA, class TensorE, class Functor>
void assign_functor(
	tensor_expression<2, TensorA, cpu_tag>& A, 
	tensor_expression<2, TensorE, cpu_tag> const& E,
	Functor f,
	axis<0,1>, axis<1,0>, dense_tag, dense_tag
){
	//compute blockwise by reading the rhs transposed
	constexpr std::size_t blockSize = 16;
	typename TensorA::value_type blockStorage[blockSize][blockSize];

	typedef typename TensorA::size_type size_type;
	size_type size1 = A().shape()[0];
	size_type size2 = A().shape()[1];
	auto e_elem = E().elements();
	for (size_type iblock = 0; iblock < size1; iblock += blockSize){
		for (size_type jblock = 0; jblock < size2; jblock += blockSize){
			std::size_t blockSizei = std::min(blockSize,size1-iblock);
			std::size_t blockSizej = std::min(blockSize,size2-jblock);

			//fill the block with the values of E
			for (size_type j = 0; j < blockSizej; ++j){
				for (size_type i = 0; i < blockSizei; ++i){
					blockStorage[i][j] = e_elem(iblock+i,jblock+j);
				}
			}

			//compute block values and store in A
			for (size_type i = 0; i < blockSizei; ++i){
				for (size_type j = 0; j < blockSizej; ++j){
					A()(iblock+i,jblock+j) = f(A()(iblock+i,jblock+j), blockStorage[i][j]);
				}
			}
		}
	}
}

template<std::size_t N, class TensorA, class TensorE, class Functor, unsigned I0, unsigned I1, unsigned... Inds,  class AxisE, class Tag>
void assign_functor(
	tensor_expression<N, TensorA, cpu_tag>& A, 
	tensor_expression<N, TensorE, cpu_tag> const& E,
	Functor f,
	axis<I0, I1, Inds...>, AxisE, dense_tag t1, Tag t2
){
	typedef typename TensorA::size_type size_type;
	auto shape = A().shape();
	//we implement this by indexing over the first N-2 dimensions and than use the 2D version over the sliced matrix
	//issue: Inds... does not necessarily go 0,...,N-3 but is any subset of {0,N-1},
	//to correct for that, we store them in a full N-dim index-vector but use the Inds[i] to store index i.
	std::array<unsigned, N - 2> map={Inds...};
	std::array<size_type, N> index={0};
	size_type num_slices = shape.num_elements()/(shape[N - 2] * shape[N - 1]);
	
	for(size_type s = 0; s != num_slices; ++s){
		// assign the current slice
		auto slice_A = slice(A,std::get<Inds>(index)...);
		auto slice_E = slice(E,std::get<Inds>(index)...);
		assign_functor(slice_A, slice_E, f, typename decltype(slice_A)::axis(), typename decltype(slice_E)::axis(), t1, t2);
		// increment index
		for (size_type i = N - 2; i > 0; --i){
			auto j = map[i - 1];
			index[j] += 1;
			if(index[j] == shape[i - 1]){
				index[j] = 0;
			}else{
				break;
			}
		}
	}
}

/////////////////////////////////////////////////////////////////
//////Matrix Assignment implementing op=
////////////////////////////////////////////////////////////////

template<std::size_t N, class TensorA, class TensorE, class AxisA, class AxisE, class Tag>
void assign(
	tensor_expression<N, TensorA, cpu_tag>& A, 
	tensor_expression<N, TensorE, cpu_tag> const& E,
	AxisA ax, AxisE ex, dense_tag t1, Tag t2
){
	typedef typename TensorA::value_type value_type;
	assign_functor(A, E, device_traits<cpu_tag>::right_arg<value_type>(), ax, ex, t1, t2);
}


// direct assignment for sparse matrices with the same orientation
// template<class M, class E>
// void matrix_assign(
	// matrix_expression<M, cpu_tag>& m,
	// matrix_expression<E, cpu_tag> const& e,
	// row_major, row_major,sparse_tag, sparse_tag
// ) {
	// m().clear();
	// for(std::size_t i = 0; i != m().size1(); ++i){
		// auto m_pos = m().major_begin(i);
		// auto end = e().major_end(i);
		// for(auto it = e().major_begin(i); it != end; ++it){
			// m_pos = m().set_element(m_pos,it.index(),*it);
		// }
	// }
// }

//remain the versions where both arguments do not have the same orientation

//sparse-sparse
/*
template<class M, class E>
void matrix_assign(
	matrix_expression<M, cpu_tag>& m,
	matrix_expression<E, cpu_tag> const& e,
	row_major, column_major,sparse_tag,sparse_tag
) {
	//apply the transposition of e()
	//first evaluate e and fill the values into a vector which is then sorted by row_major order
	//this gives this algorithm a run time of  O(eval(e)+k*log(k))
	//where eval(e) is the time to evaluate and k*log(k) the number of non-zero elements
	typedef typename M::value_type value_type;
	typedef typename M::size_type size_type;
	typedef row_major::sparse_element<value_type> Element;
	std::vector<Element> elements;

	size_type size2 = m().size2();
	for(size_type j = 0; j != size2; ++j){
		auto pos_e = e().major_begin(j);
		auto end_e = e().major_end(j);
		for(; pos_e != end_e; ++pos_e){
			Element element;
			element.i = pos_e.index();
			element.j = j;
			element.value = *pos_e;
			elements.push_back(element);
		}
	}
	std::sort(elements.begin(),elements.end());
	//fill m with the contents
	m().clear();
	size_type num_elems = size_type(elements.size());
	for(size_type current = 0; current != num_elems;){
		//count elements in row and reserve enough space for it
		size_type row = elements[current].i;
		size_type row_end = current;
		while(row_end != num_elems && elements[row_end].i == row)
			++ row_end;
		m().major_reserve(row,row_end - current);

		//copy elements
		auto row_pos = m().major_begin(row);
		for(; current != row_end; ++current){
			row_pos = m().set_element(row_pos,elements[current].j,elements[current].value);
		}
	}
}

//triangular row_major,row_major
template<class M, class E, bool Upper, bool Unit>
void matrix_assign(
	matrix_expression<M, cpu_tag>& m,
	matrix_expression<E, cpu_tag> const& e,
	triangular<row_major,triangular_tag<Upper, false> >, triangular<row_major,triangular_tag<Upper, Unit> >,
	packed_tag, packed_tag
) {
	for(std::size_t i = 0; i != m().size1(); ++i){
		auto mpos = m().major_begin(i);
		auto epos = e().major_begin(i);
		auto eend = e().major_end(i);
		if(Unit && Upper){
			*mpos = 1;
			++mpos;
		}
		REMORA_SIZE_CHECK(mpos.index() == epos.index());
		for(; epos != eend; ++epos,++mpos){
			*mpos = *epos;
		}
		if(Unit && Upper){
			*mpos = 1;
		}
	}
}

////triangular row_major,column_major
//todo: this is suboptimal as we do strided access!!!!
template<class M, class E,class Triangular>
void matrix_assign(
	matrix_expression<M, cpu_tag>& m,
	matrix_expression<E, cpu_tag> const& e,
	triangular<row_major,Triangular>, triangular<column_major,Triangular>,
	packed_tag, packed_tag
) {
	auto e_elem = e().elements();
	for(std::size_t i = 0; i != m().size1(); ++i){
		auto mpos = m().major_begin(i);
		auto mend = m().major_end(i);
		for(; mpos!=mend; ++mpos){
			*mpos = e_elem(i,mpos.index());
		}
	}
}
*/
///////////////////////////////////////////////////////////////////////////////////////////
//////Matrix Assignment With Functor implementing +=,-=...
///////////////////////////////////////////////////////////////////////////////////////////

/*//when both are row-major and target is dense we can map to vector case
template<class F, class M, class E, class TagE>
void matrix_assign_functor(
	matrix_expression<M, cpu_tag>& m,
	matrix_expression<E, cpu_tag> const& e,
	F f,
	row_major, row_major, dense_tag, TagE
) {
	for(std::size_t i = 0; i != m().size1(); ++i){
		auto rowM = row(m,i);
		kernels::assign(rowM,row(e,i),f);
	}
}
template<class F, class M, class E>
void matrix_assign_functor(
	matrix_expression<M, cpu_tag>& m,
	matrix_expression<E, cpu_tag> const& e,
	F f,
	row_major, row_major, sparse_tag, sparse_tag
) {
	typedef typename M::value_type value_type;
	value_type zero = value_type();
	typedef row_major::sparse_element<value_type> Element;
	std::vector<Element> elements;
	
	for(std::size_t i = 0; i != major_size(m); ++i){
		//first merge the two rows in elements using the functor
		
		elements.clear();
		auto m_pos = m().major_begin(i);
		auto m_end = m().major_end(i);
		auto e_pos = e().major_begin(i);
		auto e_end = e().major_end(i);
		
		while(m_pos != m_end && e_pos != e_end){
			if(m_pos.index() < e_pos.index()){
				elements.push_back({i,m_pos.index(), f(*m_pos, zero)});
				++m_pos;
			}else if( m_pos.index() == e_pos.index()){
				elements.push_back({i,m_pos.index(), f(*m_pos ,*e_pos)});
				++m_pos;
				++e_pos;
			}
			else{ //m_pos.index() > e_pos.index()
				elements.push_back({i,e_pos.index(), f(zero,*e_pos)});
				++e_pos;
			}
		}
		for(;m_pos != m_end;++m_pos){
			elements.push_back({i,m_pos.index(), f(*m_pos, zero)});
		}
		for(;e_pos != e_end; ++e_pos){
			elements.push_back({i,e_pos.index(), f(zero, *e_pos)});
		}
		
		//clear contents of m and fill with elements
		m().clear_range(m().major_begin(i),m().major_end(i));
		m().major_reserve(i,elements.size());
		m_pos = m().major_begin(i);
		for(auto elem: elements){
			m_pos = m().set_element(m_pos, elem.j, elem.value);
		}
	}
}
	

//we only need to implement the remaining versions for column major second argument

//dense-dense case
template<class F,class M, class E>
void matrix_assign_functor(
	matrix_expression<M, cpu_tag>& m,
	matrix_expression<E, cpu_tag> const& e,
	F f,
	row_major, column_major,dense_tag, dense_tag
) {
	//compute blockwise and wrelem the transposed block.
	std::size_t const blockSize = 16;
	typename M::value_type blockStorage[blockSize][blockSize];

	typedef typename M::size_type size_type;
	size_type size1 = m().size1();
	size_type size2 = m().size2();
	auto e_elem = e().elements();
	for (size_type iblock = 0; iblock < size1; iblock += blockSize){
		for (size_type jblock = 0; jblock < size2; jblock += blockSize){
			std::size_t blockSizei = std::min(blockSize,size1-iblock);
			std::size_t blockSizej = std::min(blockSize,size2-jblock);

			//fill the block with the values of e
			for (size_type j = 0; j < blockSizej; ++j){
				for (size_type i = 0; i < blockSizei; ++i){
					blockStorage[i][j] = e_elem(iblock+i,jblock+j);
				}
			}

			//compute block values and store in m
			for (size_type i = 0; i < blockSizei; ++i){
				for (size_type j = 0; j < blockSizej; ++j){
					m()(iblock+i,jblock+j) = f(m()(iblock+i,jblock+j), blockStorage[i][j]);
				}
			}
		}
	}
}

//dense-sparse case
template<class F,class M, class E>
void matrix_assign_functor(
	matrix_expression<M, cpu_tag>& m,
	matrix_expression<E, cpu_tag> const& e,
	F f,
	row_major, column_major,dense_tag, sparse_tag
) {
	for(std::size_t j = 0; j != m().size2(); ++j){
		auto columnM = column(m,j);
		kernels::assign(columnM,column(e,j),f);
	}
}

//sparse-sparse
template<class F,class M, class E>
void matrix_assign_functor(
	matrix_expression<M, cpu_tag>& m,
	matrix_expression<E, cpu_tag> const& e,
	F f,
	row_major, column_major,sparse_tag t,sparse_tag
) {
	typename matrix_temporary<M>::type eTrans = e;//explicit calculation of the transpose for now
	matrix_assign_functor(m,eTrans,f,row_major(),row_major(),t,t);
}


//kernels for triangular
template<class F, class M, class E, class Triangular, class Tag1, class Tag2>
void matrix_assign_functor(
	matrix_expression<M, cpu_tag>& m,
	matrix_expression<E, cpu_tag> const& e,
	F f,
	triangular<row_major,Triangular>, triangular<row_major,Triangular>,
	Tag1, Tag2
) {
	//there is nothing we can do if F does not leave the non-stored elements 0
	//this is the case for all current assignment functors, but you never know :)

	for(std::size_t i = 0; i != m().size1(); ++i){
		auto mpos = m().major_begin(i);
		auto epos = e().major_begin(i);
		auto mend = m().major_end(i);
		REMORA_SIZE_CHECK(mpos.index() == epos.index());
		for(; mpos!=mend; ++mpos,++epos){
			*mpos = f(*mpos,*epos);
		}
	}
}

//todo: this is suboptimal as we do strided access!!!!
template<class F, class M, class E, class Triangular, class Tag1, class Tag2>
void matrix_assign_functor(
	matrix_expression<M, cpu_tag>& m,
	matrix_expression<E, cpu_tag> const& e,
	F f,
	triangular<row_major,Triangular>, triangular<column_major,Triangular>,
	Tag1, Tag2
) {
	//there is nothing we can do, if F does not leave the non-stored elements 0
	auto e_elem = e().elements();
	for(std::size_t i = 0; i != m().size1(); ++i){
		auto mpos = m().major_begin(i);
		auto mend = m().major_end(i);
		for(; mpos!=mend; ++mpos){
			*mpos = f(*mpos,e_elem(i,mpos.index()));
		}
	}
}


*/


}}

#endif
