/*!
 * \brief       Implementation of the sparse matrix class
 * 
 * \author      O. Krause
 * \date        2017
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
#ifndef REMORA_CPU_SPARSE_MATRIX_HPP
#define REMORA_CPU_SPARSE_MATRIX_HPP

namespace remora{namespace detail{

template<class T, class I>
struct MatrixStorage{
	typedef sparse_matrix_storage<T,I> storage_type;
	typedef I size_type;
	typedef T value_type;
	
	MatrixStorage(size_type major_size, size_type minor_size, size_type non_zeros = 0)
	: m_major_indices_begin(major_size + 1,0)
	, m_major_indices_end(major_size,0)
	, m_minor_size(minor_size)
	{
		reserve(non_zeros);
	}
	
	storage_type reserve(size_type non_zeros){
		if(non_zeros > m_indices.size()){
			m_indices.resize(non_zeros);
			m_values.resize(non_zeros);
		}
		return {m_values.data(), m_indices.data(), m_major_indices_begin.data(), m_major_indices_end.data(), m_indices.size()};
	}
	
	std::size_t major_size()const{
		return m_major_indices_end.size();
	}
	
	std::size_t minor_size()const{
		return m_minor_size;
	}
	
	template<class Archive>
	void serialize(Archive& ar, const unsigned int /* file_version */) {
		ar & boost::serialization::make_nvp("indices", m_indices);
		ar & boost::serialization::make_nvp("values", m_values);
		ar & boost::serialization::make_nvp("major_indices_begin", m_major_indices_begin);
		ar & boost::serialization::make_nvp("major_indices_end", m_major_indices_end);
		ar & boost::serialization::make_nvp("minor_size", m_minor_size);
	}
private:
	std::vector<I> m_major_indices_begin;
	std::vector<I> m_major_indices_end;
	std::vector<T> m_values;
	std::vector<I> m_indices;
	std::size_t m_minor_size;
};
template<class StorageManager>
class compressed_matrix_impl{
public:
	typedef typename StorageManager::storage_type storage_type;
	typedef typename StorageManager::size_type size_type;
	typedef typename StorageManager::value_type value_type;

	compressed_matrix_impl(StorageManager const& manager, std::size_t nnz = 0)
	: m_manager(manager)
	, m_storage(m_manager.reserve(nnz)){};
	
	compressed_matrix_impl(compressed_matrix_impl const& impl)
	: m_manager(impl.m_manager)
	, m_storage(m_manager.reserve(impl.nnz_reserved())){};

	// Accessors
	size_type major_size() const {
		return m_manager.major_size();
	}
	size_type minor_size() const {
		return m_manager.minor_size();
	}
	
	storage_type const& raw_storage()const{
		return m_storage;
	}
	
	/// \brief Maximum number of non-zeros that the matrix can store or reserve before memory needs to be reallocated
	std::size_t nnz_capacity() const{
		return m_storage.capacity;
	}
	/// \brief Size of reserved storage in the matrix (> number of nonzeros stored)
	std::size_t nnz_reserved() const {
		return m_storage.major_indices_begin[major_size()];
	}
	/// \brief Number of nonzeros the major index (a major or column depending on orientation) can maximally store before a resize
	std::size_t major_capacity(size_type i)const{
		REMORA_RANGE_CHECK(i < major_size());
		return m_storage.major_indices_begin[i+1] - m_storage.major_indices_begin[i];
	}
	/// \brief Number of nonzeros the major index (a major or column depending on orientation) currently stores
	std::size_t major_nnz(size_type i) const {
		return m_storage.major_indices_end[i] - m_storage.major_indices_begin[i];
	}

	/// \brief Set the number of nonzeros stored in the major index (a major or column depending on orientation)
	void set_major_nnz(size_type i,std::size_t non_zeros) {
		REMORA_SIZE_CHECK(i < major_size());
		REMORA_SIZE_CHECK(non_zeros <= major_capacity(i));
		m_storage.major_indices_end[i] = m_storage.major_indices_begin[i]+non_zeros;
	}
	
	void reserve(std::size_t non_zeros) {
		if (non_zeros < nnz_capacity()) return;
		m_storage = m_manager.reserve(non_zeros);
	}

	void major_reserve(size_type i, std::size_t non_zeros, bool exact_size = false) {
		REMORA_RANGE_CHECK(i < major_size());
		non_zeros = std::min(minor_size(),non_zeros);
		std::size_t current_capacity = major_capacity(i);
		if (non_zeros <= current_capacity) return;
		std::size_t space_difference = non_zeros - current_capacity;

		//check if there is place in the end of the container to store the elements
		if (space_difference > nnz_capacity() - nnz_reserved()){
			std::size_t exact = nnz_capacity() + space_difference;
			std::size_t spaceous = std::max(2*nnz_capacity(),nnz_capacity() + 2*space_difference);
			reserve(exact_size? exact:spaceous);
		}
		//move the elements of the next majors to make room for the reserved space
		for (size_type k = major_size()-1; k != i; --k) {
			value_type* values = m_storage.values + m_storage.major_indices_begin[k];
			value_type* values_end =m_storage.values + m_storage.major_indices_end[k];
			size_type* indices = m_storage.indices + m_storage.major_indices_begin[k];
			size_type* indices_end = m_storage.indices + m_storage.major_indices_end[k];
			std::copy_backward(values, values_end, values_end + space_difference);
			std::copy_backward(indices, indices_end, indices_end + space_difference);
			m_storage.major_indices_begin[k] += space_difference;
			m_storage.major_indices_end[k] += space_difference;
		}
		m_storage.major_indices_begin[major_size()] += space_difference;
	}

	void resize(size_type major, size_type minor){
		m_storage = m_manager.resize(major);
		m_minor_size = minor;
	}
	
	typedef iterators::compressed_storage_iterator<value_type const, size_type const> const_major_iterator;
	typedef iterators::compressed_storage_iterator<value_type, size_type> major_iterator;

	const_major_iterator cmajor_begin(size_type i) const {
		REMORA_SIZE_CHECK(i < major_size());
		REMORA_RANGE_CHECK(m_storage.major_indices_begin[i] <= m_storage.major_indices_end[i]);//internal check
		return const_major_iterator(m_storage.values, m_storage.indices, m_storage.major_indices_begin[i],i);
	}

	const_major_iterator cmajor_end(size_type i) const {
		REMORA_SIZE_CHECK(i < major_size());
		REMORA_RANGE_CHECK(m_storage.major_indices_begin[i] <= m_storage.major_indices_end[i]);//internal check
		return const_major_iterator(m_storage.values, m_storage.indices, m_storage.major_indices_end[i],i);
	}

	major_iterator major_begin(size_type i) {
		REMORA_SIZE_CHECK(i < major_size());
		REMORA_RANGE_CHECK(m_storage.major_indices_begin[i] <= m_storage.major_indices_end[i]);//internal check
		return major_iterator(m_storage.values, m_storage.indices, m_storage.major_indices_begin[i],i);
	}

	major_iterator major_end(size_type i) {
		REMORA_SIZE_CHECK(i < major_size());
		REMORA_RANGE_CHECK(m_storage.major_indices_begin[i] <= m_storage.major_indices_end[i]);//internal check
		return major_iterator(m_storage.values, m_storage.indices, m_storage.major_indices_end[i],i);
	}
	
	major_iterator set_element(major_iterator pos, size_type index, value_type value) {
		std::size_t major_index = pos.major_index();
		std::size_t line_pos = pos - major_begin(major_index);
		REMORA_RANGE_CHECK(major_index < major_size());
		REMORA_RANGE_CHECK(size_type(pos - major_begin(major_index)) <= major_nnz(major_index));

		//shortcut: element already exists.
		if (pos != major_end(major_index) && pos.index() == index) {
			*pos = value;
			return pos + 1;
		}
		
		//get position of the element in the array.
		std::ptrdiff_t arrayPos = line_pos + m_storage.major_indices_begin[major_index];

		//check that there is enough space in the major. this invalidates pos.
		if (major_capacity(major_index) ==  major_nnz(major_index))
			major_reserve(major_index,std::max<std::size_t>(2*major_capacity(major_index),5));

		//copy the remaining elements further to make room for the new element
		std::copy_backward(
			m_storage.values + arrayPos, m_storage.values + m_storage.major_indices_end[major_index],
			m_storage.values + m_storage.major_indices_end[major_index] + 1
		);
		std::copy_backward(
			m_storage.indices + arrayPos, m_storage.indices + m_storage.major_indices_end[major_index],
			m_storage.indices + m_storage.major_indices_end[major_index] + 1
		);
		//insert new element
		m_storage.values[arrayPos] = value;
		m_storage.indices[arrayPos] = index;
		++m_storage.major_indices_end[major_index];

		//return new iterator to the inserted element.
		return major_begin(major_index) + (line_pos + 1);

	}

	major_iterator clear_range(major_iterator start, major_iterator end) {
		REMORA_RANGE_CHECK(start.major_index() == end.major_index());
		std::size_t major_index = start.major_index();
		std::size_t range_size = end - start;
		std::size_t range_start = start - major_begin(major_index);
		std::size_t range_end = range_start + range_size;
		
		//get start of the storage of the row/column we are going to change
		auto values = m_storage.values + m_storage.major_indices_begin[major_index];
		auto indices = m_storage.indices + m_storage.major_indices_begin[major_index];

		//remove the elements in the range by copying the elements after it to the start of the range
		std::copy(values + range_end, values + major_nnz(major_index), values + range_start);
		std::copy(indices + range_end, indices + major_nnz(major_index), indices + range_start);
		//subtract number of removed elements
		m_storage.major_indices_end[major_index] -= range_size;
		//return new iterator to the first element after the end of the range
		return major_begin(major_index) + range_start;
	}

	major_iterator clear_element(major_iterator elem) {
		REMORA_RANGE_CHECK(elem != major_end());
		return clear_range(elem,elem + 1);
	}

private:
	StorageManager m_manager;
	storage_type m_storage;
	size_type m_minor_size;
};

///\brief proxy handling: closure, transpose and rows of a given matrix
template<class M, class Orientation>
class compressed_matrix_proxy: public matrix_expression<compressed_matrix_proxy<M, Orientation>, cpu_tag>{
private:
	template<class,class> friend class compressed_matrix_proxy;
public:
	typedef typename M::size_type size_type;
	typedef typename M::value_type value_type;
	typedef typename M::const_reference const_reference;
	typedef typename remora::reference<M>::type reference;
	
	typedef compressed_matrix_proxy<M const, Orientation> const_closure_type;
	typedef compressed_matrix_proxy<M, Orientation> closure_type;
	typedef typename remora::storage<M>::type storage_type;
	typedef typename M::const_storage_type const_storage_type;
	typedef elementwise<sparse_tag> evaluation_category;
	typedef Orientation orientation;

	
	//conversion matrix->proxy
	compressed_matrix_proxy(M& m): m_matrix(&m), m_major_start(0){
		m_major_end = M::orientation::index_M(m.size1(),m.size2());
	}
	
	//rows/columns proxy
	compressed_matrix_proxy(M& m, size_type start, size_type end): m_matrix(&m), m_major_start(start), m_major_end(end){}
	
	
	//copy-ctor/const-conversion
	compressed_matrix_proxy( compressed_matrix_proxy<typename std::remove_const<M>::type, Orientation> const& proxy)
	:m_matrix(proxy.m_matrix), m_major_start(proxy.m_major_start), m_major_end(proxy.m_major_end){}
	
	M& matrix() const{
		return *m_matrix;
	}
	
	size_type start() const{
		return m_major_start;
	}
	size_type end() const{
		return m_major_end;
	}
	
	//assignment from different expression
	template<class E>
	compressed_matrix_proxy& operator=(matrix_expression<E, cpu_tag> const& e){
		return assign(*this, typename matrix_temporary<E>::type(e));
	}
	compressed_matrix_proxy& operator=(compressed_matrix_proxy const& e){
		return assign(*this, typename matrix_temporary<M>::type(e));
	}
	
	///\brief Number of rows of the matrix
	size_type size1() const {
		return orientation::index_M( m_major_end - m_major_start, minor_size(*m_matrix));
	}
	
	///\brief Number of columns of the matrix
	size_type size2() const {
		return orientation::index_m(m_major_end - m_major_start, minor_size(*m_matrix));
	}

	/// \brief Number of nonzeros the major index (a row or column depending on orientation) can maximally store before a resize
	std::size_t major_capacity(size_type i)const{
		return m_matrix->major_capacity(m_major_start + i);
	}
	/// \brief Number of nonzeros the major index (a row or column depending on orientation) currently stores
	std::size_t major_nnz(size_type i) const {
		return m_matrix->major_nnz(m_major_start + i);
	}
	
	const_storage_type raw_storage()const{
		return m_matrix->raw_storage();
	}
	
	typename device_traits<cpu_tag>::queue_type& queue() const{
		return m_matrix->queue();
	}

	void major_reserve(size_type i, std::size_t non_zeros, bool exact_size = false) {
		m_matrix->major_reserve(m_major_start + i, non_zeros, exact_size);
	}
	
	typedef typename major_iterator<M>::type major_iterator;
	typedef major_iterator const_major_iterator;

	major_iterator major_begin(size_type i) const {
		return m_matrix->major_begin(m_major_start + i);
	}

	major_iterator major_end(size_type i) const{
		return m_matrix->major_end(m_major_start + i);
	}
	
	major_iterator set_element(major_iterator pos, size_type index, value_type value){
		return m_matrix->set_element(pos, index, value);
	}

	major_iterator clear_range(major_iterator start, major_iterator end) {
		return m_matrix->clear_range(start,end);
	}
	
	void clear() {
		if(m_major_start == 0 && m_major_end == major_size(*m_matrix))
			m_matrix->clear();
		else for(std::size_t i = m_major_start; i != m_major_end; ++i)
			m_matrix->clear_range(m_matrix -> major_begin(i), m_matrix -> major_end(i));
	}
private:
	M* m_matrix;
	size_type m_major_start;
	size_type m_major_end;
};



template<class M>
class compressed_matrix_row: public vector_expression<compressed_matrix_row<M>, cpu_tag>{
private:
	template<class> friend class compressed_matrix_row;
public:
	typedef typename closure<M>::type matrix_type;
	typedef typename M::size_type size_type;
	typedef typename M::value_type value_type;
	typedef typename M::const_reference const_reference;
	typedef typename remora::reference<M>::type reference;
	
	typedef compressed_matrix_row<M const> const_closure_type;
	typedef compressed_matrix_row closure_type;
	typedef typename remora::storage<M>::type::template row_storage<row_major>::type storage_type;
	typedef typename M::const_storage_type::template row_storage<row_major>::type const_storage_type;
	typedef elementwise<sparse_tag> evaluation_category;

	// Construction
	compressed_matrix_row(matrix_type const& m, size_type i):m_matrix(m), m_row(i){}
	//copy or conversion ctor non-const ->const proxy
	compressed_matrix_row(compressed_matrix_row<typename std::remove_const<M>::type > const& ref):m_matrix(ref.m_matrix){}
	
	//assignment from different expression
	template<class E>
	compressed_matrix_row& operator=(vector_expression<E, cpu_tag> const& e){
		return assign(*this, typename vector_temporary<E>::type(e));
	}
	
	compressed_matrix_row& operator=(compressed_matrix_row const& e){
		return assign(*this, typename vector_temporary<M>::type(e));
	}
	
	///\brief Returns the underlying storage structure for low level access
	storage_type raw_storage(){
		return m_matrix.raw_storage().row(m_row, row_major());
	}
	
	///\brief Returns the underlying storage structure for low level access
	const_storage_type raw_storage() const{
		return m_matrix.raw_storage().row(m_row, row_major());
	}
	
	/// \brief Return the size of the vector
	size_type size() const {
		return m_matrix.size2();
	}
	
	size_type nnz() const {
		return m_matrix.major_nnz(m_row);
	}
	
	size_type nnz_capacity(){
		return m_matrix.major_capacity(m_row);
	}
	
	void reserve(size_type non_zeros) {
		m_matrix.major_reserve(m_row, non_zeros);
	}
	
	typename device_traits<cpu_tag>::queue_type& queue(){
		return device_traits<cpu_tag>::default_queue();
	}
	
	// --------------
	// ITERATORS
	// --------------

	typedef typename major_iterator<matrix_type>::type iterator;
	typedef iterator const_iterator;

	/// \brief return an iterator tp the first non-zero element of the vector
	iterator begin() const {
		return m_matrix.major_begin(m_row);
	}

	/// \brief return an iterator behind the last non-zero element of the vector
	iterator end() const {
		return m_matrix.major_end(m_row);
	}
	
	iterator set_element(iterator pos, size_type index, value_type value) {
		return m_matrix.set_element(pos,index,value);
	}
	
	iterator clear_range(iterator start, iterator end) {
		m_matrix.clear_range(start,end);
	}

	iterator clear_element(iterator pos){
		m_matrix.clear_element(pos);
	}
	
	void clear(){
		m_matrix.clear_range(begin(),end());
	}
private:
	matrix_type m_matrix;
	size_type m_row;
};

}}
#endif