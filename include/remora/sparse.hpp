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
 #ifndef REMORA_CPU_SPARSE_HPP
#define REMORA_CPU_SPARSE_HPP

#include "cpu/iterator.hpp"
#include "detail/traits.hpp"
#include "assignment.hpp"

#include <vector>

#include <boost/serialization/collection_size_type.hpp>
#include <boost/serialization/nvp.hpp>
#include <boost/serialization/vector.hpp>

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


/** \brief Compressed array based sparse vector
 *
 * a sparse vector of values of type T of variable size. The non zero values are stored as
 * two seperate arrays: an index array and a value array. The index array is always sorted
 * and there is at most one entry for each index. Inserting an element can be time consuming.
 * If the vector contains a few zero entries, then it is better to have a normal vector.
 * If the vector has a very high dimension with a few non-zero values, then this vector is
 * very memory efficient (at the cost of a few more computations).
 *
 * For a \f$n\f$-dimensional compressed vector and \f$0 \leq i < n\f$ the non-zero elements
 * \f$v_i\f$ are mapped to consecutive elements of the index and value container, i.e. for
 * elements \f$k = v_{i_1}\f$ and \f$k + 1 = v_{i_2}\f$ of these containers holds \f$i_1 < i_2\f$.
 *
 * Supported parameters for the adapted array (indices and values) are \c unbounded_array<> ,
 * \c bounded_array<> and \c std::vector<>.
 *
 * \tparam T the type of object stored in the vector (like double, float, complex, etc...)
 * \tparam I the indices stored in the vector
 */
template<class T, class I = std::size_t>
class compressed_vector:public vector_container<compressed_vector<T, I>, cpu_tag > {

	typedef T& true_reference;
	typedef compressed_vector<T, I> self_type;
public:
	typedef T value_type;
	typedef const T& const_reference;

	typedef I size_type;
	
	class reference {
	private:

		const_reference value()const {
			return const_cast<self_type const&>(m_vector)(m_i);
		}
		value_type& ref() const {
			//find position of the index in the array
			size_type const* start = m_vector.m_indices.data();
			size_type const* end = start + m_vector.nnz();
			size_type const *pos = std::lower_bound(start,end,m_i);

			if (pos != end&& *pos == m_i)
				return m_vector.m_values[pos-start];
			else {
				//create iterator to the insertion position and insert new element
				iterator posIter(m_vector.m_values.data(),m_vector.m_indices.data(),pos-start);
				return *m_vector.set_element(posIter, m_i, m_vector.m_zero);
			}
		}

	public:
		// Construction and destruction
		reference(self_type& m, size_type i):
			m_vector(m), m_i(i) {}

		// Assignment
		value_type& operator = (value_type d)const {
			return ref()=d;
		}
		
		value_type& operator=(reference const& v ){
			return ref() = v.value();
		}
		
		value_type& operator += (value_type d)const {
			return ref()+=d;
		}
		value_type& operator -= (value_type d)const {
			return ref()-=d;
		}
		value_type& operator *= (value_type d)const {
			return ref()*=d;
		}
		value_type& operator /= (value_type d)const {
			return ref()/=d;
		}

		// Comparison
		bool operator == (value_type d) const {
			return value() == d;
		}
		bool operator != (value_type d) const {
			return value() != d;
		}
		
		operator const_reference() const{
			return value();
		}
	private:
		self_type& m_vector;
		size_type m_i;
	};

	typedef sparse_vector_adaptor<T const,I const> const_closure_type;
	typedef sparse_vector_adaptor<T, I> closure_type;
	typedef sparse_vector_storage<value_type const,size_type const> const_storage_type;
	typedef elementwise<sparse_tag> evaluation_category;

	// Construction and destruction
	compressed_vector():m_size(0), m_nnz(0),m_indices(1,0),m_zero(0){}
	explicit compressed_vector(size_type size, value_type value = value_type(), size_type non_zeros = 0)
	:m_size(size), m_nnz(0), m_indices(non_zeros,0), m_values(non_zeros),m_zero(0){}
	template<class AE>
	compressed_vector(vector_expression<AE, cpu_tag> const& ae, size_type non_zeros = 0)
	:m_size(ae().size()), m_nnz(0), m_indices(non_zeros,0), m_values(non_zeros),m_zero(0){
		assign(*this, ae);
	}

	// Accessors
	size_type size() const {
		return m_size;
	}
	size_type nnz_capacity() const {
		return m_indices.size();
	}
	size_type nnz() const {
		return m_nnz;
	}

	void set_filled(size_type filled) {
		REMORA_SIZE_CHECK(filled <= nnz_capacity());
		m_nnz = filled;
	}
	
	///\brief Returns the underlying storage structure for low level access
	storage_type raw_storage(){
		return {m_values.data(), m_indices.data(), m_nnz};
	}
	
	///\brief Returns the underlying storage structure for low level access
	const_storage_type raw_storage() const{
		return {m_values.data(), m_indices.data(), m_nnz};
	}
	
	typename device_traits<cpu_tag>::queue_type& queue(){
		return device_traits<cpu_tag>::default_queue();
	}

	void resize(size_type size) {
		m_size = size;
		m_nnz = 0;
	}
	void reserve(size_type non_zeros) {
		if(non_zeros <= nnz_capacity()) return;
		non_zeros = std::min(size(),non_zeros);
		m_indices.resize(non_zeros);
		m_values.resize(non_zeros);
	}

	// Element access
	const_reference operator()(size_type i) const {
		REMORA_SIZE_CHECK(i < m_size);
		std::size_t pos = lower_bound(i);
		if (pos == nnz() || m_indices[pos] != i)
			return m_zero;
		return m_values [pos];
	}
	reference operator()(size_type i) {
		return reference(*this,i);
	}


	const_reference operator [](size_type i) const {
		return (*this)(i);
	}
	reference operator [](size_type i) {
		return (*this)(i);
	}

	// Zeroing
	void clear() {
		m_nnz = 0;
	}

	// Assignment
	compressed_vector& operator = (compressed_vector const& v) {
		m_size = v.m_size;
		m_nnz = v.m_nnz;
		m_indices = v.m_indices;
		m_values = v.m_values;
		return *this;
	}
	template<class C>          // Container assignment without temporary
	compressed_vector& operator = (vector_container<C, cpu_tag> const& v) {
		resize(v().size(), false);
		assign(*this, v);
		return *this;
	}
	template<class AE>
	compressed_vector& operator = (vector_expression<AE, cpu_tag> const& ae) {
		self_type temporary(ae, nnz_capacity());
		swap(temporary);
		return *this;
	}

	// Swapping
	void swap(compressed_vector& v) {
		std::swap(m_size, v.m_size);
		std::swap(m_nnz, v.m_nnz);
		m_indices.swap(v.m_indices);
		m_values.swap(v.m_values);
	}

	friend void swap(compressed_vector& v1, compressed_vector& v2){
		v1.swap(v2);
	}

	// Iterator types
	typedef iterators::compressed_storage_iterator<value_type const, size_type const> const_iterator;
	typedef iterators::compressed_storage_iterator<value_type, size_type const> iterator;

	const_iterator begin() const {
		return const_iterator(m_values.data(),m_indices.data(),0);
	}

	const_iterator end() const {
		return const_iterator(m_values.data(),m_indices.data(),nnz());
	}

	iterator begin() {
		return iterator(m_values.data(),m_indices.data(),0);
	}

	iterator end() {
		return iterator(m_values.data(),m_indices.data(),nnz());
	}
	
	// Element assignment
	iterator set_element(iterator pos, size_type index, value_type value) {
		REMORA_RANGE_CHECK(size_type(pos - begin()) <=m_size);
		
		if(pos != end() && pos.index() == index){
			*pos = value;
			return pos;
		}
		//get position of the new element in the array.
		std::ptrdiff_t arrayPos = pos - begin();
		if (m_nnz <= nnz_capacity())//reserve more space if needed, this invalidates pos.
			reserve(std::max<std::size_t>(2 * nnz_capacity(),1));
		
		//copy the remaining elements to make space for the new ones
		std::copy_backward(
			m_values.begin()+arrayPos,m_values.begin() + m_nnz , m_values.begin() + m_nnz +1
		);
		std::copy_backward(
			m_indices.begin()+arrayPos,m_indices.begin() + m_nnz , m_indices.begin() + m_nnz +1
		);
		//insert new element
		m_values[arrayPos] = value;
		m_indices[arrayPos] = index;
		++m_nnz;
		
		
		//return new iterator to the inserted element.
		return iterator(m_values.data(),m_indices.data(),arrayPos);
	}
	
	iterator clear_range(iterator start, iterator end) {
		//get position of the elements in the array.
		std::ptrdiff_t startPos = start - begin();
		std::ptrdiff_t endPos = end - begin();
		
		//remove the elements in the range
		std::copy(
			m_values.begin()+endPos,m_values.begin() + m_nnz, m_values.begin() + startPos
		);
		std::copy(
			m_indices.begin()+endPos,m_indices.begin() + m_nnz , m_indices.begin() + startPos
		);
		m_nnz -= endPos - startPos;
		//return new iterator to the next element
		return iterator(m_values.data(),m_indices.data(), startPos);
	}

	iterator clear_element(iterator pos){
		//get position of the element in the array.
		std::ptrdiff_t arrayPos = pos - begin();
		if(arrayPos == m_nnz-1){//last element
			--m_nnz;
			return end();
		}
		
		std::copy(
			m_values.begin()+arrayPos+1,m_values.begin() + m_nnz , m_values.begin() + arrayPos
		);
		std::copy(
			m_indices.begin()+arrayPos+1,m_indices.begin() + m_nnz , m_indices.begin() + arrayPos
		);
		//return new iterator to the next element
		return iterator(m_values.data(),m_indices.data(),arrayPos);
	}

	// Serialization
	template<class Archive>
	void serialize(Archive& ar, const unsigned int /* file_version */) {
		boost::serialization::collection_size_type s(m_size);
		ar & boost::serialization::make_nvp("size",s);
		if (Archive::is_loading::value) {
			m_size = s;
		}
		// ISSUE: m_indices and m_values are undefined between m_nnz and capacity (trouble with 'nan'-values)
		ar & boost::serialization::make_nvp("nnz", m_nnz);
		ar & boost::serialization::make_nvp("indices", m_indices);
		ar & boost::serialization::make_nvp("values", m_values);
	}

private:
	std::size_t lower_bound( size_type t)const{
		size_type const* begin = m_indices.data();
		size_type const* end = m_indices.data()+nnz();
		return std::lower_bound(begin, end, t)-begin;
	}

	size_type m_size;
	size_type m_nnz;
	std::vector<size_type> m_indices;
	std::vector<value_type> m_values;
	value_type m_zero;
};

template<class T, class I=std::size_t>
class compressed_matrix:public matrix_container<compressed_matrix<T, I>, cpu_tag > {
	typedef compressed_matrix<T, I> self_type;
public:
	typedef I size_type;
	typedef T value_type;
	

	typedef T const& const_reference;
	class reference {
	private:
		const_reference value()const {
			return const_cast<self_type const&>(m_matrix)(m_i,m_j);
		}
		value_type& ref() const {
			//get array bounds
			size_type const *start = m_matrix.m_indices.data() + m_matrix.m_rowStart[m_i];
			size_type const *end = m_matrix.m_indices.data() + m_matrix.m_rowEnd[m_i];
			//find position of the index in the array
			size_type const *pos = std::lower_bound(start,end,m_j);

			if (pos != end && *pos == m_j)
				return m_matrix.m_values[(pos-start)+m_matrix.m_rowStart[m_i]];
			else {
				//create iterator to the insertion position and insert new element
				row_iterator posIter(
				    m_matrix.m_values.data(),
				    m_matrix.m_indices.data(),
				    pos-start + m_matrix.m_rowStart[m_i]
				    ,m_i
				);
				return *m_matrix.set_element(posIter, m_j, m_matrix.m_zero);
			}
		}

	public:
		// Construction and destruction
		reference(compressed_matrix &m, size_type i, size_type j):
			m_matrix(m), m_i(i), m_j(j) {}

		// Assignment
		value_type& operator = (value_type d)const {
			return ref() = d;
		}
		value_type& operator=(reference const & other){
			return ref() = other.value();
		}
		value_type& operator += (value_type d)const {
			return ref()+=d;
		}
		value_type& operator -= (value_type d)const {
			return ref()-=d;
		}
		value_type& operator *= (value_type d)const {
			return ref()*=d;
		}
		value_type& operator /= (value_type d)const {
			return ref()/=d;
		}
		
		operator const_reference() const {
			return value();
		}
	private:
		compressed_matrix& m_matrix;
		size_type m_i;
		size_type m_j;
	};

	typedef matrix_reference<self_type const> const_closure_type;
	typedef matrix_reference<self_type> closure_type;
	typedef sparse_matrix_storage<T,I> storage_type;
	typedef sparse_matrix_storage<value_type const,size_type const> const_storage_type;
	typedef elementwise<sparse_tag> evaluation_category;
	typedef row_major orientation;

	// Construction and destruction
	compressed_matrix()
		: m_size1(0), m_size2(0), m_nnz(0)
		, m_rowStart(1,0), m_indices(0), m_values(0), m_zero(0) {}

	compressed_matrix(size_type size1, size_type size2, size_type non_zeros = 0)
		: m_size1(size1), m_size2(size2), m_nnz(0)
		, m_rowStart(size1 + 1,0)
		, m_rowEnd(size1,0)
		, m_indices(non_zeros), m_values(non_zeros), m_zero(0) {}

	template<class E>
	compressed_matrix(matrix_expression<E, cpu_tag> const& e, size_type non_zeros = 0)
		: m_size1(e().size1()), m_size2(e().size2()), m_nnz(0)
		, m_rowStart(e().size1() + 1, 0)
		, m_rowEnd(e().size1(), 0)
		, m_indices(non_zeros), m_values(non_zeros), m_zero(0) {
		assign(*this, e);
	}

	// Accessors
	size_type size1() const {
		return m_size1;
	}
	size_type size2() const {
		return m_size2;
	}

	std::size_t nnz_capacity() const {
		return m_values.size();
	}
	std::size_t row_capacity(size_type row)const {
		REMORA_RANGE_CHECK(row < size1());
		return m_rowStart[row+1] - m_rowStart[row];
	}
	std::size_t nnz() const {
		return m_nnz;
	}
	std::size_t inner_nnz(size_type row) const {
		return m_rowEnd[row] - m_rowStart[row];
	}
	
	///\brief Returns the underlying storage structure for low level access
	storage_type raw_storage(){
		return {m_values.data(),m_indices.data(), m_rowStart.data(), m_rowEnd.data()};
	}
	
	///\brief Returns the underlying storage structure for low level access
	const_storage_type raw_storage() const{
		return {m_values.data(),m_indices.data(), m_rowStart.data(), m_rowEnd.data()};
	}
	
	typename device_traits<cpu_tag>::queue_type& queue(){
		return device_traits<cpu_tag>::default_queue();
	}

	void set_filled(std::size_t non_zeros) {
		m_nnz = non_zeros;
	}
	
	void set_row_filled(size_type i,std::size_t non_zeros) {
		REMORA_SIZE_CHECK(i < size1());
		REMORA_SIZE_CHECK(non_zeros <=row_capacity(i));
		
		m_rowEnd[i] = m_rowStart[i]+non_zeros;
		//correct end pointers
		if(i == size1()-1)
			m_rowStart[size1()] = m_rowEnd[i];
	}

	void resize(size_type size1, size_type size2) {
		m_size1 = size1;
		m_size2 = size2;
		m_nnz = 0;
		//clear row data
		m_rowStart.resize(m_size1 + 1);
		m_rowEnd.resize(m_size1);
		std::fill(m_rowStart.begin(),m_rowStart.end(),0);
		std::fill(m_rowEnd.begin(),m_rowEnd.end(),0);
	}
	void reserve(std::size_t non_zeros) {
		if (non_zeros < nnz_capacity()) return;
		//non_zeros = std::min(m_size2*m_size1,non_zeros);//this can lead to totally strange errors.
		m_indices.resize(non_zeros);
		m_values.resize(non_zeros);
	}

	void reserve_row(size_type row, std::size_t non_zeros) {
		REMORA_RANGE_CHECK(row < size1());
		non_zeros = std::min(m_size2,non_zeros);
		if (non_zeros <= row_capacity(row)) return;
		std::size_t spaceDifference = non_zeros - row_capacity(row);

		//check if there is place in the end of the container to store the elements
		if (spaceDifference > nnz_capacity()-m_rowStart.back()) {
			reserve(nnz_capacity()+std::max<std::size_t>(nnz_capacity(),2*spaceDifference));
		}
		//move the elements of the next rows to make room for the reserved space
		for (size_type i = size1()-1; i != row; --i) {
			value_type* values = m_values.data() + m_rowStart[i];
			value_type* valueRowEnd = m_values.data() + m_rowEnd[i];
			size_type* indices = m_indices.data() + m_rowStart[i];
			size_type* indicesEnd = m_indices.data() + m_rowEnd[i];
			std::copy_backward(values,valueRowEnd, valueRowEnd+spaceDifference);
			std::copy_backward(indices,indicesEnd, indicesEnd+spaceDifference);
			m_rowStart[i]+=spaceDifference;
			m_rowEnd[i]+=spaceDifference;
		}
		m_rowStart.back() +=spaceDifference;
		REMORA_SIZE_CHECK(row_capacity(row) == non_zeros);
	}

	void clear() {
		m_nnz = 0;
		m_rowStart [0] = 0;
	}

	// Element access
	const_reference operator()(size_type i, size_type j) const {
		REMORA_SIZE_CHECK(i < size1());
		REMORA_SIZE_CHECK(j < size2());
		//get array bounds
		size_type const *start = m_indices.data() + m_rowStart[i];
		size_type const *end = m_indices.data() + m_rowEnd[i];
		//find position of the index in the array
		size_type const *pos = std::lower_bound(start,end,j);

		if (pos != end && *pos == j)
			return m_values[(pos-start)+m_rowStart[i]];
		else
			return m_zero;
	}

	reference operator()(size_type i, size_type j) {
		REMORA_SIZE_CHECK(i < size1());
		REMORA_SIZE_CHECK(j < size2());
		return reference(*this,i,j);
	}

	// Assignment
	template<class C>          // Container assignment without temporary
	compressed_matrix &operator = (matrix_container<C, cpu_tag> const& m) {
		resize(m().size1(), m().size2());
		assign(*this, m);
		return *this;
	}
	template<class E>
	compressed_matrix &operator = (matrix_expression<E, cpu_tag> const& e) {
		self_type temporary(e, nnz_capacity());
		swap(temporary);
		return *this;
	}

	// Swapping
	void swap(compressed_matrix &m) {
		std::swap(m_size1, m.m_size1);
		std::swap(m_size2, m.m_size2);
		std::swap(m_nnz, m.m_nnz);
		m_rowStart.swap(m.m_rowStart);
		m_rowEnd.swap(m.m_rowEnd);
		m_indices.swap(m.m_indices);
		m_values.swap(m.m_values);
	}

	friend void swap(compressed_matrix &m1, compressed_matrix &m2) {
		m1.swap(m2);
	}

	friend void swap_rows(compressed_matrix& a, size_type i, compressed_matrix& b, size_type j) {
		REMORA_SIZE_CHECK(i < a.size1());
		REMORA_SIZE_CHECK(j < b.size1());
		REMORA_SIZE_CHECK(a.size2() == b.size2());
		
		//rearrange (i,j) such that i has equal or more elements than j
		if(a.inner_nnz(i) < b.inner_nnz(j)){
			swap_rows(b,j,a,i);
			return;
		}
		
		std::size_t nnzi = a.inner_nnz(i);
		std::size_t nnzj = b.inner_nnz(j);
		
		//reserve enough space for swapping
		b.reserve_row(j,nnzi);
		REMORA_SIZE_CHECK(b.row_capacity(j) >= nnzi);
		REMORA_SIZE_CHECK(a.row_capacity(i) >= nnzj);
		
		size_type* indicesi = a.m_indices.data() + a.m_rowStart[i];
		size_type* indicesj = b.m_indices.data() + b.m_rowStart[j];
		value_type* valuesi = a.m_values.data() + a.m_rowStart[i];
		value_type* valuesj = b.m_values.data() + b.m_rowStart[j];
		
		//swap all elements of j with the elements in i, don't care about unitialized elements in j
		std::swap_ranges(indicesi,indicesi+nnzi,indicesj);
		std::swap_ranges(valuesi, valuesi+nnzi,valuesj);
		
		//if both rows had the same number of elements, we are done.
		if(nnzi == nnzj)
			return;
		
		//otherwise correct end pointers
		a.set_row_filled(i,nnzj);
		b.set_row_filled(j,nnzi);
	}
	
	friend void swap_rows(compressed_matrix& a, size_type i, size_type j) {
		if(i == j) return;
		swap_rows(a,i,a,j);
	}
	
	typedef iterators::compressed_storage_iterator<value_type const, size_type const> const_row_iterator;
	typedef iterators::compressed_storage_iterator<value_type, size_type const> row_iterator;

	const_row_iterator row_begin(size_type i) const {
		REMORA_SIZE_CHECK(i < size1());
		REMORA_RANGE_CHECK(m_rowStart[i] <= m_rowEnd[i]);//internal check
		return const_row_iterator(m_values.data(), m_indices.data(), m_rowStart[i],i);
	}

	const_row_iterator row_end(size_type i) const {
		REMORA_SIZE_CHECK(i < size1());
		REMORA_RANGE_CHECK(m_rowStart[i] <= m_rowEnd[i]);//internal check
		return const_row_iterator(m_values.data(), m_indices.data(), m_rowEnd[i],i);
	}

	row_iterator row_begin(size_type i) {
		REMORA_SIZE_CHECK(i < size1());
		REMORA_RANGE_CHECK(m_rowStart[i] <= m_rowEnd[i]);//internal check
		return row_iterator(m_values.data(), m_indices.data(), m_rowStart[i],i);
	}

	row_iterator row_end(size_type i) {
		REMORA_SIZE_CHECK(i < size1());
		REMORA_RANGE_CHECK(m_rowStart[i] <= m_rowEnd[i]);//internal check
		return row_iterator(m_values.data(), m_indices.data(), m_rowEnd[i],i);
	}
	
	typedef iterators::compressed_storage_iterator<value_type const, size_type const> const_column_iterator;
	typedef iterators::compressed_storage_iterator<value_type, size_type const> column_iterator;
	
	row_iterator set_element(row_iterator pos, size_type index, value_type value) {
		std::size_t row = pos.row();
		REMORA_RANGE_CHECK(row < size1());
		REMORA_RANGE_CHECK(size_type(row_end(row) - pos) <= row_capacity(row));
		//todo: check in debug, that iterator position is valid

		//shortcut: element already exists.
		if (pos != row_end(row) && pos.index() == index) {
			*pos = value;
			return pos;
		}

		//get position of the element in the array.
		std::ptrdiff_t arrayPos = (pos - row_begin(row)) + m_rowStart[row];

		//check that there is enough space in the row. this invalidates pos.
		if (row_capacity(row) ==  inner_nnz(row))
			reserve_row(row,std::max<std::size_t>(2*row_capacity(row),1));

		//copy the remaining elements further to make room for the new element
		std::copy_backward(
		    m_values.begin() + arrayPos, m_values.begin() + m_rowEnd[row],
		    m_values.begin() + m_rowEnd[row] + 1
		);
		std::copy_backward(
		    m_indices.begin()+arrayPos, m_indices.begin() + m_rowEnd[row],
		    m_indices.begin() + m_rowEnd[row] + 1
		);
		//insert new element
		m_values[arrayPos] = value;
		m_indices[arrayPos] = index;
		++m_rowEnd[row];
		++m_nnz;

		//return new iterator to the inserted element.
		return row_iterator(m_values.data(), m_indices.data(), arrayPos,row);

	}

	row_iterator clear_range(row_iterator start, row_iterator end) {
		std::size_t row = start.row();
		REMORA_RANGE_CHECK(row == end.row());
		//get position of the elements in the array.
		size_type rowEndPos = m_rowEnd[row];
		size_type rowStartPos = m_rowStart[row];
		size_type rangeStartPos = start - row_begin(row)+rowStartPos;
		size_type rangeEndPos = end - row_begin(row)+rowStartPos;
		std::ptrdiff_t rangeSize = end - start;

		//remove the elements in the range
		std::copy(
		    m_values.begin()+rangeEndPos,m_values.begin() + rowEndPos, m_values.begin() + rangeStartPos
		);
		std::copy(
		    m_indices.begin()+rangeEndPos,m_indices.begin() + rowEndPos , m_indices.begin() + rangeStartPos
		);
		m_rowEnd[row] -= rangeSize;
		m_nnz -= rangeSize;
		//return new iterator to the next element
		return row_iterator(m_values.data(), m_indices.data(), rangeStartPos,row);
	}

	row_iterator clear_element(row_iterator elem) {
		REMORA_RANGE_CHECK(elem != row_end());
		row_iterator next = elem;
		++next;
		clear_range(elem,next);
	}

	// Serialization
	template<class Archive>
	void serialize(Archive &ar, const unsigned int /* file_version */) {
		ar &boost::serialization::make_nvp("outer_indices", m_rowStart);
		ar &boost::serialization::make_nvp("outer_indices_end", m_rowEnd);
		ar &boost::serialization::make_nvp("inner_indices", m_indices);
		ar &boost::serialization::make_nvp("values", m_values);
	}

private:
	size_type m_size1;
	size_type m_size2;
	size_type m_nnz;
	std::vector<size_type> m_rowStart;
	std::vector<size_type> m_rowEnd;
	std::vector<size_type> m_indices;
	std::vector<value_type> m_values;
	value_type m_zero;
};

template<class T, class O>
struct matrix_temporary_type<T,O,sparse_tag, cpu_tag> {
	typedef compressed_matrix<T> type;
};

template<class T>
struct vector_temporary_type<T,sparse_tag, cpu_tag>{
	typedef compressed_vector<T> type;
};

}
#endif
