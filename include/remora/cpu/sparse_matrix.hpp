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

#include <iostream>
namespace remora{namespace detail{
template<class Mat>
struct MatrixReference{
	typedef typename Mat::storage_type storage_type;
	typedef typename Mat::size_type size_type;
	typedef typename Mat::value_type value_type;
	
	void set_nnz(size_type non_zeros){
		m_matrix->set_nnz(non_zeros);
	}
	storage_type reserve(size_type non_zeros){
		m_matrix->reserve(non_zeros);
		return m_matrix->raw_storage();
	}
	
	MatrixReference(Mat* matrix): m_matrix(matrix){}
private:
	Mat* m_matrix;
};

template<class T, class I>
struct ConstantMatrixStorage{
	typedef sparse_matrix_storage<T,I> storage_type;
	typedef I size_type;
	typedef typename std::remove_const<T>::type value_type;
	
	void set_nnz(size_type non_zeros){
		assert(non_zeros <= m_storage.capacity);
		m_storage.nnz = non_zeros;
	}
	storage_type reserve(size_type non_zeros){
		assert(non_zeros <= m_storage.capacity);
		return m_storage;
	}
	
	ConstantMatrixStorage(storage_type storage): m_storage(storage){}
private:
	storage_type m_storage;
};

template<class T, class I>
struct MatrixStorage{
	typedef sparse_matrix_storage<T,I> storage_type;
	typedef I size_type;
	typedef T value_type;
	
	MatrixStorage(size_type major_size, size_type minor_size)
	: m_major_indices_begin(major_size + 1,0)
	, m_major_indices_end(major_size,0)
	, m_nnz(0)
	, m_minor_size(minor_size){}
	
	void set_nnz(size_type non_zeros){
		m_nnz = non_zeros;
	}
	
	storage_type reserve(size_type non_zeros){
		if(non_zeros > m_indices.size()){
			m_indices.resize(non_zeros);
			m_values.resize(non_zeros);
		}
		return {m_values.data(), m_indices.data(), m_major_indices_begin.data(), m_major_indices_end.data(), m_nnz, m_indices.size()};
	}
	
	std::size_t major_size()const{
		return m_major_indices_end.size();
	}
	
	std::size_t minor_size()const{
		return m_minor_size;
	}
	
	MatrixStorage(): m_nnz(0){}
	
	template<class Archive>
	void serialize(Archive& ar, const unsigned int /* file_version */) {
		ar & boost::serialization::make_nvp("nnz", m_nnz);
		ar & boost::serialization::make_nvp("indices", m_indices);
		ar & boost::serialization::make_nvp("values", m_values);
		ar & boost::serialization::make_nvp("major_indices_begin", m_major_indices_begin);
		ar & boost::serialization::make_nvp("major_indices_end", m_major_indices_end);
	}
private:
	std::vector<I> m_major_indices_begin;
	std::vector<I> m_major_indices_end;
	std::vector<T> m_values;
	std::vector<I> m_indices;
	std::size_t m_nnz;
	std::size_t m_minor_size;
};
template<class StorageManager>
class compressed_matrix_impl{
public:
	typedef typename StorageManager::storage_type storage_type;
	typedef typename StorageManager::size_type size_type;
	typedef typename StorageManager::value_type value_type;

	compressed_matrix_impl(StorageManager const& manager, std::size_t nnz)
	: m_manager(manager)
	, m_storage(m_manager.reserve(nnz)){};

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
	
	/// \brief Number of nonzeros this matrix can maximally store before requiring new memory
	std::size_t nnz_capacity() const{
		return m_storage.capacity;
	}
	/// \brief Total Number of nonzeros this matrix stores
	std::size_t nnz() const {
		return m_storage.nnz;
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

	/// \brief Set the total number of nonzeros stored by the matrix
	void set_nnz(std::size_t non_zeros) {
		m_manager.set_nnz(non_zeros);
		m_storage.nnz = non_zeros;
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
		if (space_difference > nnz_capacity() - m_storage.major_indices_begin[major_size()]){
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
		set_nnz(nnz()+1);

		//return new iterator to the inserted element.
		return major_begin(major_index) + (line_pos + 1);

	}

	major_iterator clear_range(major_iterator start, major_iterator end) {
		REMORA_RANGE_CHECK(start.index() == end.index());
		std::size_t major_index = start.index();
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
		set_nnz(nnz() - range_size);
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

}}
#endif