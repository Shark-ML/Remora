/*!
 * \brief       Classes used for vector proxies
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
 #ifndef REMORA_VECTOR_PROXY_CLASSES_HPP
#define REMORA_VECTOR_PROXY_CLASSES_HPP

#include "../cpu/iterator.hpp"
#include "traits.hpp"

#include <type_traits>
namespace remora{

template<class T,class I>
class sparse_vector_adaptor: public vector_expression<sparse_vector_adaptor<T,I>, cpu_tag > {
	typedef sparse_vector_adaptor<T,I> self_type;
public:

	//std::container types
	typedef typename std::remove_const<I>::type size_type;
	typedef typename std::remove_const<T>::type value_type;
	typedef value_type result_type;
	typedef value_type const& const_reference;
	typedef const_reference reference;
	
	typedef sparse_vector_adaptor<value_type const,size_type const> const_closure_type;
	typedef sparse_vector_adaptor closure_type;
	typedef sparse_vector_storage<T const,I const> storage_type;
	typedef sparse_vector_storage<value_type const,size_type const> const_storage_type;
	typedef elementwise<sparse_tag> evaluation_category;

	// Construction and destruction

	/// \brief Constructor of a self_type proxy from a Dense VectorExpression
	///
	/// Be aware that the expression must live longer than the proxy!
	/// \param expression Expression from which to construct the Proxy
 	template<class E>
	sparse_vector_adaptor(vector_expression<E, cpu_tag> const& expression)
	: m_nonZeros(expression().raw_storage().nnz)
	, m_indices(expression().raw_storage().indices)
	, m_values(expression().raw_storage().values)
	, m_size(expression().size()){}
	
	
	sparse_vector_adaptor():m_size(0){}
		
	/// \brief Constructor of a vector proxy from a block of memory
	/// \param size the size of the vector represented by the memory
	/// \param values the block of memory used to store the values
	/// \param indices the block of memory used to store the indices
	/// \param memoryLength length of the strip of memory
	sparse_vector_adaptor(
		size_type size, 
		value_type const* values,
		size_type const* indices, 
		size_type memoryLength
	): m_nonZeros(memoryLength)
	, m_indices(indices)
	, m_values(values)
	, m_size(size){}
	
	/// \brief Return the size of the vector
	size_type size() const {
		return m_size;
	}
	
	///\brief Returns the underlying storage_type structure for low level access
	storage_type raw_storage() const{
		return {m_values,m_indices, m_nonZeros};
	}
	
	typename device_traits<cpu_tag>::queue_type& queue(){
		return device_traits<cpu_tag>::default_queue();
	}

	/// \brief Return a const reference to the element \f$i\f$
	/// \param i index of the element
	value_type operator()(size_type i) const {
		REMORA_SIZE_CHECK(i < m_size);
		size_type const* pos = std::lower_bound(m_indices,m_indices+m_nonZeros, i);
		std::ptrdiff_t diff = pos-m_indices;
		if(diff == (std::ptrdiff_t) m_nonZeros || *pos != i)
			return value_type();
		return m_values[diff];
	}

	/// \brief Return a const reference to the element \f$i\f$
	/// \param i index of the element
	value_type operator[](size_type i) const {
		return (*this)(i);
	}

	// --------------
	// ITERATORS
	// --------------
	
	typedef iterators::compressed_storage_iterator<value_type const, size_type const> const_iterator;
	typedef const_iterator iterator;

	/// \brief return an iterator behind the last non-zero element of the vector
	const_iterator begin() const {
		return const_iterator(m_values,m_indices,0);
	}

	/// \brief return an iterator behind the last non-zero element of the vector
	const_iterator end() const {
		return const_iterator(m_values,m_indices,m_nonZeros);
	}
private:
	std::size_t m_nonZeros;
	size_type const* m_indices;
	value_type const* m_values;

	std::size_t m_size;
};

}
#endif
