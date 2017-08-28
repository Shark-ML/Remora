/*!
 * \brief       Dense Matrix and Vector classes
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
#ifndef REMORA_CPU_DENSE_HPP
#define REMORA_CPU_DENSE_HPP

#include "iterator.hpp"
#include "../detail/proxy_optimizers_fwd.hpp"

#include <initializer_list>
#include <boost/serialization/collection_size_type.hpp>
#include <boost/serialization/array.hpp>
#include <boost/serialization/nvp.hpp>
#include <boost/serialization/vector.hpp>

namespace remora{
	
/// \brief Represents a given chunk of memory as a dense vector of elements of type T.
///
/// This adaptor is read/write if T is non-const and read-only if T is const.
template<class T, class Tag>
class dense_vector_adaptor<T, Tag, cpu_tag>: public vector_expression<dense_vector_adaptor<T, Tag, cpu_tag>, cpu_tag > {
	typedef dense_vector_adaptor<T, Tag> self_type;
public:

	typedef std::size_t size_type;
	typedef typename std::remove_const<T>::type value_type;
	typedef value_type const& const_reference;
	typedef T&  reference;

	typedef dense_vector_adaptor<T const, Tag, cpu_tag> const_closure_type;
	typedef dense_vector_adaptor closure_type;
	typedef dense_vector_storage<T, Tag> storage_type;
	typedef dense_vector_storage<value_type const, Tag> const_storage_type;
	typedef elementwise<dense_tag> evaluation_category;

	// Construction and destruction

	/// \brief Constructor of a self_type proxy from a Dense VectorExpression
	///
	/// Be aware that the expression must live longer than the proxy!
	/// \param expression The Expression from which to construct the Proxy
	template<class E>
	dense_vector_adaptor(vector_expression<E, cpu_tag> const& expression)
	: m_values(expression().raw_storage().values)
	, m_size(expression().size())
	, m_stride(expression().raw_storage().stride){
		static_assert(!std::is_convertible<typename E::storage_type::storage_tag,Tag>::value, "Can not convert storage type of argunent to the given Tag");
	}
	
	/// \brief Constructor of a self_type proxy from a Dense VectorExpression
	///
	/// Be aware that the expression must live longer than the proxy!
	/// \param expression The Expression from which to construct the Proxy
	template<class E>
	dense_vector_adaptor(vector_expression<E,cpu_tag>& expression)
	: m_values(expression().raw_storage().values)
	, m_size(expression().size())
	, m_stride(expression().raw_storage().stride){
		static_assert(!std::is_convertible<typename E::storage_type::storage_tag,Tag>::value, "Can not convert storage type of argunent to the given Tag");
	}
		
	/// \brief Constructor of a self_type proxy from a block of memory
	/// \param values the block of memory used
	/// \param size size of the self_type
	/// \param stride distance between elements of the self_type in memory
	dense_vector_adaptor(T* values, size_type size, size_type stride = 1 ):
		m_values(values),m_size(size),m_stride(stride){}

	
	dense_vector_adaptor(storage_type const& storage, no_queue, size_type size):
		m_values(storage.values),m_size(size),m_stride(storage.stride){}	

		
	/// \brief Copy-constructor of a self_type
	/// \param v is the proxy to be copied
	template<class U, class Tag2>
	dense_vector_adaptor(dense_vector_adaptor<U, Tag2> const& v)
	: m_values(v.raw_storage().values)
	, m_size(v.size())
	, m_stride(v.raw_storage().stride){
		static_assert(!std::is_convertible<Tag2,Tag>::value, "Can not convert storage type of argunent to the given Tag");
	}
	
	/// \brief Return the size of the vector.
	size_type size() const {
		return m_size;
	}
	
	///\brief Returns the underlying storage_type structure for low level access
	storage_type raw_storage() const{
		return {m_values,m_stride};
	}
	
	typename device_traits<cpu_tag>::queue_type& queue(){
		return device_traits<cpu_tag>::default_queue();
	}
	// --------------
	// Element access
	// --------------

	/// \brief Return a const reference to the element \f$i\f$
	/// \param i index of the element
	const_reference operator()(size_type i) const {
		return m_values[i*m_stride];
	}
	
	/// \brief Return a reference to the element \f$i\f$
	/// \param i index of the element
	reference operator()(size_type i) {
		return m_values[i*m_stride];
	}	

	/// \brief Return a const reference to the element \f$i\f$
	/// \param i index of the element
	const_reference operator[](size_type i) const {
		return m_values[i*m_stride];
	}
	
	/// \brief Return a reference to the element \f$i\f$
	/// \param i index of the element
	reference operator[](size_type i) {
		return m_values[i*m_stride];
	}

	// ------------------
	// Element assignment
	// ------------------
	
	/// \brief Set element \f$i\f$ to the value \c t
	/// \param i index of the element
	/// \param t reference to the value to be set
	reference insert_element(size_type i, const_reference t) {
		return(*this)[i] = t;
	}

	/// \brief Set element \f$i\f$ to the \e zero value
	/// \param i index of the element
	void erase_element(size_type i) {
		(*this)[i] = value_type/*zero*/();
	}
		

	dense_vector_adaptor& operator = (dense_vector_adaptor const& e) {
		return assign(typename vector_temporary<self_type>::type(e));
	}
	template<class E>
	dense_vector_adaptor& operator = (vector_expression<E, cpu_tag> const& e) {
		return assign(typename vector_temporary<E>::type(e));
	}
	
	// --------------
	// ITERATORS
	// --------------
	

	typedef iterators::dense_storage_iterator<T> iterator;
	typedef iterators::dense_storage_iterator<value_type const> const_iterator;

	/// \brief return an iterator on the first element of the vector
	const_iterator begin() const {
		return const_iterator(m_values,0);
	}

	/// \brief return an iterator after the last element of the vector
	const_iterator end() const {
		return const_iterator(m_values+size()*m_stride,size());
	}

	/// \brief Return an iterator on the first element of the vector
	iterator begin(){
		return iterator(m_values,0);
	}

	/// \brief Return an iterator at the end of the vector
	iterator end(){
		return iterator(m_values+size()*m_stride,size());
	}
	
	//insertion and erasing of elements
	iterator set_element(iterator pos, size_type index, value_type value) {
		REMORA_SIZE_CHECK(pos.index() == index);
		(*this)(index) = value;
		return pos;
	}

	iterator clear_element(iterator pos) {
		REMORA_SIZE_CHECK(pos != end());
		v(pos.index()) = value_type();
		
		//return new iterator to the next element
		return pos+1;
	}
	
	iterator clear_range(iterator start, iterator end) {
		REMORA_RANGE_CHECK(start < end);
		for(; start != end; ++start){
			*start = value_type/*zero*/();
		}
		return end;
	}
private:
	T* m_values;
	size_type m_size;
	size_type m_stride;
};
	

template<class T,class Orientation, class Tag>
class dense_matrix_adaptor<T,Orientation,Tag, cpu_tag>: public matrix_expression<dense_matrix_adaptor<T,Orientation, Tag, cpu_tag>, cpu_tag > {
	typedef dense_matrix_adaptor<T,Orientation, cpu_tag> self_type;
public:
	typedef std::size_t size_type;
	typedef typename std::remove_const<T>::type value_type;
	typedef value_type result_type;
	typedef value_type const& const_reference;
	typedef T& reference;

	typedef dense_matrix_adaptor<T,Orientation, Tag, cpu_tag> closure_type;
	typedef dense_matrix_adaptor<value_type const,Orientation, Tag, cpu_tag> const_closure_type;
	typedef dense_matrix_storage<T,dense_tag> storage_type;
	typedef dense_matrix_storage<value_type const,Tag> const_storage_type;
	typedef Orientation orientation;
	typedef elementwise<dense_tag> evaluation_category;

	template<class,class,class> friend class dense_matrix_adaptor;

	// Construction and destruction
	template<class U, class TagU>
	dense_matrix_adaptor(dense_matrix_adaptor<U, Orientation, TagU, cpu_tag> const& expression)
	: m_values(expression.m_values)
	, m_size1(expression.size1())
	, m_size2(expression.size2())
	, m_stride1(expression.m_stride1)
	, m_stride2(expression.m_stride2)
	{static_assert(!std::is_convertible<TagU,Tag>::value, "Can not convert storage type of argunent to the given Tag");}
	
	template<class E>
	dense_matrix_adaptor(storage_type const& storage, no_queue, std::size_t size1, std::size_t size2)
	: m_size1(size1)
	, m_size2(size2)
	{
		m_values = storage.values;
		m_stride1 = Orientation::index_M(storage.leading_dimension,1);
		m_stride2 = Orientation::index_m(storage.leading_dimension,1);
	}

	/// \brief Constructor of a vector proxy from a Dense MatrixExpression
	///
	/// Be aware that the expression must live longer than the proxy!
	/// \param expression Expression from which to construct the Proxy
	template<class E>
	dense_matrix_adaptor(matrix_expression<E, cpu_tag> const& expression)
	: m_size1(expression().size1())
	, m_size2(expression().size2())
	{
		auto storage_type = expression().raw_storage();
		m_values = storage_type.values;
		m_stride1 = Orientation::index_M(storage_type.leading_dimension,1);
		m_stride2 = Orientation::index_m(storage_type.leading_dimension,1);
		static_assert(std::is_same<typename E::orientation,orientation>::value, "matrix orientation mismatch");
		static_assert(!std::is_convertible<typename E::storage_type::storage_tag,Tag>::value, "Can not convert storage type of argunent to the given Tag");
	}

	/// \brief Constructor of a vector proxy from a Dense MatrixExpression
	///
	/// Be aware that the expression must live longer than the proxy!
	/// \param expression Expression from which to construct the Proxy
	template<class E>
	dense_matrix_adaptor(matrix_expression<E, cpu_tag>& expression)
	: m_size1(expression().size1())
	, m_size2(expression().size2())
	{
		auto storage_type = expression().raw_storage();
		m_values = storage_type.values;
		m_stride1 = Orientation::index_M(storage_type.leading_dimension,1);
		m_stride2 = Orientation::index_m(storage_type.leading_dimension,1);
		static_assert(std::is_same<typename E::orientation,orientation>::value, "matrix orientation mismatch");
		static_assert(!std::is_convertible<typename E::storage_type::storage_tag,Tag>::value, "Can not convert storage type of argunent to the given Tag");
	}
		
	/// \brief Constructor of a vector proxy from a block of memory
	/// \param values the block of memory used
	/// \param size1 size in 1st direction
	/// \param size2 size in 2nd direction
	/// \param stride1 distance in 1st direction between elements of the self_type in memory
	/// \param stride2 distance in 2nd direction between elements of the self_type in memory
	dense_matrix_adaptor(
		T* values, 
		size_type size1, size_type size2,
		size_type stride1 = 0, size_type stride2 = 0 
	)
	: m_values(values)
	, m_size1(size1)
	, m_size2(size2)
	, m_stride1(stride1)
	, m_stride2(stride2)
	{
		if(!m_stride1)
			m_stride1= Orientation::stride1(m_size1,m_size2);
		if(!m_stride2)
			m_stride2= Orientation::stride2(m_size1,m_size2);
	}
	
	
	template<class E>
	dense_matrix_adaptor(vector_expression<E, cpu_tag> const& expression, std::size_t size1, std::size_t size2)
	: m_values(expression().raw_storage().values)
	, m_size1(size1)
	, m_size2(size2)
	{
		m_stride1= Orientation::stride1(m_size1,m_size2);
		m_stride2= Orientation::stride2(m_size1,m_size2);
	}
	
	template<class E>
	dense_matrix_adaptor(vector_expression<E, cpu_tag>& expression, std::size_t size1, std::size_t size2)
	: m_values(expression().raw_storage().values)
	, m_size1(size1)
	, m_size2(size2)
	{
		m_stride1= Orientation::stride1(m_size1,m_size2);
		m_stride2= Orientation::stride2(m_size1,m_size2);
	}
	
	
	// ---------
	// Dense low level interface
	// ---------
		
	/// \brief Return the number of rows of the matrix
	size_type size1() const {
		return m_size1;
	}
	/// \brief Return the number of columns of the matrix
	size_type size2() const {
		return m_size2;
	}
	
	///\brief Returns the underlying storage_type structure for low level access
	storage_type raw_storage()const{
		return {m_values, orientation::index_M(m_stride1,m_stride2)};
	}
	
	typename device_traits<cpu_tag>::queue_type& queue()const{
		return device_traits<cpu_tag>::default_queue();
	}
	
	// ---------
	// High level interface
	// ---------
	
	// -------
	// ASSIGNING
	// -------
	
	self_type& operator = (self_type const& e) {
		REMORA_SIZE_CHECK(size1() == e().size1());
		REMORA_SIZE_CHECK(size2() == e().size2());
		return assign(*this, typename matrix_temporary<self_type>::type(e));
	}
	template<class E>
	self_type& operator = (matrix_expression<E, cpu_tag> const& e) {
		REMORA_SIZE_CHECK(size1() == e().size1());
		REMORA_SIZE_CHECK(size2() == e().size2());
		return assign(*this, typename matrix_temporary<self_type>::type(e));
	}
	
	// --------------
	// Element access
	// --------------
	
	reference operator() (size_type i, size_type j) const {
		REMORA_SIZE_CHECK( i < m_size1);
		REMORA_SIZE_CHECK( j < m_size2);
		return m_values[i*m_stride1+j*m_stride2];
		}
	void set_element(size_type i, size_type j,value_type t){
		REMORA_SIZE_CHECK( i < m_size1);
		REMORA_SIZE_CHECK( j < m_size2);
		m_values[i*m_stride1+j*m_stride2]  = t;
	}

	// --------------
	// ITERATORS
	// --------------

	typedef iterators::dense_storage_iterator<T> row_iterator;
	typedef iterators::dense_storage_iterator<T> column_iterator;
	typedef iterators::dense_storage_iterator<value_type const> const_row_iterator;
	typedef iterators::dense_storage_iterator<value_type const> const_column_iterator;

	const_row_iterator row_begin(size_type i) const {
		return const_row_iterator(m_values+i*m_stride1,0,m_stride2);
	}
	const_row_iterator row_end(size_type i) const {
		return const_row_iterator(m_values+i*m_stride1+size2()*m_stride2,size2(),m_stride2);
	}
	row_iterator row_begin(size_type i){
		return row_iterator(m_values+i*m_stride1,0,m_stride2);
	}
	row_iterator row_end(size_type i){
		return row_iterator(m_values+i*m_stride1+size2()*m_stride2,size2(),m_stride2);
	}
	
	const_column_iterator column_begin(size_type j) const {
		return const_column_iterator(m_values+j*m_stride2,0,m_stride1);
	}
	const_column_iterator column_end(size_type j) const {
		return const_column_iterator(m_values+j*m_stride2+size1()*m_stride1,size1(),m_stride1);
	}
	column_iterator column_begin(size_type j){
		return column_iterator(m_values+j*m_stride2,0,m_stride1);
	}
	column_iterator column_end(size_type j){
		return column_iterator(m_values+j*m_stride2+size1()*m_stride1,size1(),m_stride1);
	}
	
	typedef typename major_iterator<self_type>::type major_iterator;
	
	major_iterator set_element(major_iterator pos, size_type index, value_type value) {
		REMORA_RANGE_CHECK(pos.index() == index);
		*pos=value;
		return pos;
	}
	
	major_iterator clear_element(major_iterator elem) {
		*elem = value_type();
		return elem+1;
	}
	
	major_iterator clear_range(major_iterator start, major_iterator end) {
		std::fill(start,end,value_type());
		return end;
	}
	
	void swap_rows(size_type i, size_type j){
		for(std::size_t k = 0; k != size2(); ++k){
			std::swap((*this)(i,k),(*this)(j,k));
		}
	}
	
	void swap_columns(size_type i, size_type j){
		for(std::size_t k = 0; k != size1(); ++k){
			std::swap((*this)(k,i),(*this)(k,j));
		}
	}
	
		
	void clear(){
		for(size_type i = 0; i != size1(); ++i){
			for(size_type j = 0; j != size2(); ++j){
				(*this)(i,j) = value_type();
			}
		}
	}
private:
	T* m_values;
	size_type m_size1;
	size_type m_size2;
	size_type m_stride1;
	size_type m_stride2;
};


/** \brief A dense matrix of values of type \c T.
 *
 * For a \f$(m \times n)\f$-dimensional matrix and \f$ 0 \leq i < m, 0 \leq j < n\f$, every element \f$ m_{i,j} \f$ is mapped to
 * the \f$(i.n + j)\f$-th element of the container for row major orientation or the \f$ (i + j.m) \f$-th element of
 * the container for column major orientation. In a dense matrix all elements are represented in memory in a
 * contiguous chunk of memory by definition.
 *
 * Orientation can also be specified, otherwise a \c row_major is used.
 *
 * \tparam T the type of object stored in the matrix (like double, float, complex, etc...)
 * \tparam L the storage organization. It can be either \c row_major or \c column_major. Default is \c row_major
 */
template<class T, class L>
class matrix<T,L,cpu_tag>:public matrix_container<matrix<T, L, cpu_tag>, cpu_tag > {
	typedef matrix<T, L> self_type;
	typedef std::vector<T> array_type;
public:
	typedef typename array_type::value_type value_type;
	typedef typename array_type::const_reference const_reference;
	typedef typename array_type::reference reference;
	typedef typename array_type::size_type size_type;

	typedef dense_matrix_adaptor<T const,L, continuous_dense_tag, cpu_tag> const_closure_type;
	typedef dense_matrix_adaptor<T,L, continuous_dense_tag, cpu_tag> closure_type;
	typedef dense_matrix_storage<T, continuous_dense_tag> storage_type;
	typedef dense_matrix_storage<T const, continuous_dense_tag> const_storage_type;
	typedef elementwise<dense_tag> evaluation_category;
	typedef L orientation;

	// Construction

	/// \brief Default dense matrix constructor. Make a dense matrix of size (0,0)
	matrix():m_size1(0), m_size2(0){}
	
	/// \brief Constructor from a nested initializer list.
	///
	/// Constructs a matrix like this: m = {{1,2},{3,4}}.
	/// \param list The nested initializer list storing the values of the matrix.
	matrix(std::initializer_list<std::initializer_list<T> > list)
	: m_size1(list.size())
	, m_size2(list.begin()->size())
	, m_data(m_size1*m_size2){
		auto pos = list.begin();
		for(std::size_t i = 0; i != list.size(); ++i,++pos){
			REMORA_SIZE_CHECK(pos->size() == m_size2);
			std::copy(pos->begin(),pos->end(),row_begin(i));
		}
	}

	/// \brief Dense matrix constructor with defined size
	/// \param size1 number of rows
	/// \param size2 number of columns
	matrix(size_type size1, size_type size2)
	:m_size1(size1)
	, m_size2(size2)
	, m_data(size1 * size2) {}

	/// \brief  Dense matrix constructor with defined size a initial value for all the matrix elements
	/// \param size1 number of rows
	/// \param size2 number of columns
	/// \param init initial value assigned to all elements
	matrix(size_type size1, size_type size2, value_type const& init)
	: m_size1(size1)
	, m_size2(size2)
	, m_data(size1 * size2, init) {}

	/// \brief Copy-constructor of a dense matrix
	///\param m is a dense matrix
	matrix(matrix const& m) = default;
			
	/// \brief Move-constructor of a dense matrix
	///\param m is a dense matrix
	//~ matrix(matrix&& m) = default; //vc++ can not default this
	matrix(matrix&& m):m_size1(m.m_size1), m_size2(m.m_size2), m_data(std::move(m.m_data)){}

	/// \brief Constructor of a dense matrix from a matrix expression.
	/// 
	/// Constructs the matrix by evaluating the expression and assigning the
	/// results to the newly constructed matrix using a call to assign.
	///
	/// \param e is a matrix expression
	template<class E>
	matrix(matrix_expression<E, cpu_tag> const& e)
	: m_size1(e().size1())
	, m_size2(e().size2())
	, m_data(m_size1 * m_size2) {
		assign(*this,e);
	}
	
	// Assignment
	
	/// \brief Assigns m to this
	matrix& operator = (matrix const& m) = default;
	
	/// \brief Move-Assigns m to this
	//~ matrix& operator = (matrix&& m) = default;//vc++ can not default this
	matrix& operator = (matrix&& m) {
		m_size1 = m.m_size1;
		m_size2 = m.m_size2;
		m_data = std::move(m.m_data);
		return *this;
	}

	
	/// \brief Assigns m to this
	/// 
	/// evaluates the expression and assign the
	/// results to this using a call to assign.
	/// As containers are assumed to not overlap, no temporary is created
	///
	/// \param m is a matrix expression
	template<class C>
	matrix& operator = (matrix_container<C, cpu_tag> const& m) {
		resize(m().size1(), m().size2());
		assign(*this, m);
		return *this;
	}
	/// \brief Assigns e to this
	/// 
	/// evaluates the expression and assign the
	/// results to this using a call to assign.
	/// A temporary is created to prevent aliasing.
	///
	/// \param e is a matrix expression
	template<class E>
	matrix& operator = (matrix_expression<E, cpu_tag> const& e) {
		self_type temporary(e);
		swap(temporary);
		return *this;
	}
	
	///\brief Returns the number of rows of the matrix.
	size_type size1() const {
		return m_size1;
	}
	///\brief Returns the number of columns of the matrix.
	size_type size2() const {
		return m_size2;
	}
	
	///\brief Returns the underlying storage structure for low level access
	storage_type raw_storage(){
		return {m_data.data(), orientation::index_m(m_size1,m_size2)};
	}
	
	///\brief Returns the underlying storage structure for low level access
	const_storage_type raw_storage()const{
		return {m_data.data(), orientation::index_m(m_size1,m_size2)};
	}
	typename device_traits<cpu_tag>::queue_type& queue(){
		return device_traits<cpu_tag>::default_queue();
	}
	
	// ---------
	// High level interface
	// ---------

	// Resizing
	/// \brief Resize a matrix to new dimensions. If resizing is performed, the data is not preserved.
	/// \param size1 the new number of rows
	/// \param size2 the new number of colums
	void resize(size_type size1, size_type size2) {
		m_data.resize(size1* size2);
		m_size1 = size1;
		m_size2 = size2;
	}
	
	void clear(){
		std::fill(m_data.begin(), m_data.end(), value_type/*zero*/());
	}

	// Element access
	const_reference operator()(size_type i, size_type j) const {
		REMORA_SIZE_CHECK(i < size1());
		REMORA_SIZE_CHECK(j < size2());
		return m_data[orientation::element(i, m_size1, j, m_size2)];
	}
	reference operator()(size_type i, size_type j) {
		REMORA_SIZE_CHECK(i < size1());
		REMORA_SIZE_CHECK(j < size2());
		return m_data[orientation::element(i, m_size1, j, m_size2)];
	}
	
	void set_element(size_type i, size_type j,value_type t){
		REMORA_SIZE_CHECK(i < size1());
		REMORA_SIZE_CHECK(j < size2());
		m_data[orientation::element(i, m_size1, j, m_size2)]  = t;
	}

	// Swapping
	void swap(matrix& m) {
		std::swap(m_size1, m.m_size1);
		std::swap(m_size2, m.m_size2);
		m_data.swap(m.m_data);
	}
	friend void swap(matrix& m1, matrix& m2) {
		m1.swap(m2);
	}
	
	friend void swap_rows(matrix& a, size_type i, matrix& b, size_type j){
		REMORA_SIZE_CHECK(i < a.size1());
		REMORA_SIZE_CHECK(j < b.size1());
		REMORA_SIZE_CHECK(a.size2() == b.size2());
		for(std::size_t k = 0; k != a.size2(); ++k){
			std::swap(a(i,k),b(j,k));
		}
	}
	
	void swap_rows(size_type i, size_type j) {
		if(i == j) return;
		for(std::size_t k = 0; k != size2(); ++k){
			std::swap((*this)(i,k),(*this)(j,k));
		}
	}
	
	
	friend void swap_columns(matrix& a, size_type i, matrix& b, size_type j){
		REMORA_SIZE_CHECK(i < a.size2());
		REMORA_SIZE_CHECK(j < b.size2());
		REMORA_SIZE_CHECK(a.size1() == b.size1());
		for(std::size_t k = 0; k != a.size1(); ++k){
			std::swap(a(k,i),b(k,j));
		}
	}
	
	void swap_columns(size_type i, size_type j) {
		if(i == j) return;
		for(std::size_t k = 0; k != size1(); ++k){
			std::swap((*this)(k,i),(*this)(k,j));
		}
	}

	//Iterators
	typedef iterators::dense_storage_iterator<value_type> row_iterator;
	typedef iterators::dense_storage_iterator<value_type> column_iterator;
	typedef iterators::dense_storage_iterator<value_type const> const_row_iterator;
	typedef iterators::dense_storage_iterator<value_type const> const_column_iterator;

	const_row_iterator row_begin(size_type i) const {
		return const_row_iterator(m_data.data() + i*stride1(),0,stride2());
	}
	const_row_iterator row_end(size_type i) const {
		return const_row_iterator(m_data.data() + i*stride1()+stride2()*size2(),size2(),stride2());
	}
	row_iterator row_begin(size_type i){
		return row_iterator(m_data.data() + i*stride1(),0,stride2());
	}
	row_iterator row_end(size_type i){
		return row_iterator(m_data.data() + i*stride1()+stride2()*size2(),size2(),stride2());
	}
	
	const_row_iterator column_begin(std::size_t j) const {
		return const_column_iterator(m_data.data() + j*stride2(),0,stride1());
	}
	const_column_iterator column_end(std::size_t j) const {
		return const_column_iterator(m_data.data() + j*stride2()+ stride1()*size1(),size1(),stride1());
	}
	column_iterator column_begin(std::size_t j){
		return column_iterator(m_data.data() + j*stride2(),0,stride1());
	}
	column_iterator column_end(std::size_t j){
		return column_iterator(m_data.data() + j * stride2()+ stride1() * size1(), size1(), stride1());
	}
	
	typedef typename major_iterator<self_type>::type major_iterator;
	
	//sparse interface
	major_iterator set_element(major_iterator pos, size_type index, value_type value) {
		REMORA_RANGE_CHECK(pos.index() == index);
		*pos=value;
		return pos;
	}
	
	major_iterator clear_element(major_iterator elem) {
		*elem = value_type();
		return elem+1;
	}
	
	major_iterator clear_range(major_iterator start, major_iterator end) {
		std::fill(start,end,value_type());
		return end;
	}
	
	void reserve(size_type non_zeros) {}
	void reserve_row(std::size_t, std::size_t){}
	void reserve_column(std::size_t, std::size_t){}

	// Serialization
	template<class Archive>
	void serialize(Archive& ar, const unsigned int /* file_version */) {

		// we need to copy to a collection_size_type to get a portable
		// and efficient boost::serialization
		boost::serialization::collection_size_type s1(m_size1);
		boost::serialization::collection_size_type s2(m_size2);

		// serialize the sizes
		ar& boost::serialization::make_nvp("size1",s1)
		& boost::serialization::make_nvp("size2",s2);

		// copy the values back if loading
		if (Archive::is_loading::value) {
			m_size1 = s1;
			m_size2 = s2;
		}
		ar& boost::serialization::make_nvp("data",m_data);
	}

private:
	size_type stride1() const {
		return orientation::stride1(m_size1, m_size2);
	}
	size_type stride2() const {
		return orientation::stride2(m_size1, m_size2);
	}

	size_type m_size1;
	size_type m_size2;
	array_type m_data;
};

namespace remora{
template<class T>
class vector<T,cpu_tag>: public vector_container<vector<T, cpu_tag>, cpu_tag > {

	typedef vector<T> self_type;
	typedef std::vector<typename std::conditional<std::is_same<T,bool>::value,char,T>::type > array_type;
public:
	typedef typename array_type::value_type value_type;
	typedef typename array_type::const_reference const_reference;
	typedef typename array_type::reference reference;
	typedef typename array_type::size_type size_type;

	typedef dense_vector_adaptor<T const, continuous_dense_tag, cpu_tag> const_closure_type;
	typedef dense_vector_adaptor<T,continuous_dense_tag, cpu_tag> closure_type;
	typedef dense_vector_storage<value_type, continuous_dense_tag> storage_type;
	typedef dense_vector_storage<value_type const, continuous_dense_tag> const_storage_type;
	typedef elementwise<dense_tag> evaluation_category;

	// Construction and destruction

	/// \brief Constructor of a vector
	/// By default it is empty, i.e. \c size()==0.
	vector() = default;

	/// \brief Constructor of a vector with a predefined size
	/// By default, its elements are initialized to 0.
	/// \param size initial size of the vector
	explicit vector(size_type size):m_storage(size) {}

	/// \brief Constructor of a vector with a predefined size and a unique initial value
	/// \param size of the vector
	/// \param init value to assign to each element of the vector
	vector(size_type size, const value_type& init):m_storage(size, init) {}

	/// \brief Copy-constructor of a vector
	/// \param v is the vector to be duplicated
	vector(vector const& v) = default;
		
	/// \brief Move-constructor of a vector
	/// \param v is the vector to be moved
	//~ vector(vector && v) = default; //vc++ can not default this. true story
	vector(vector && v): m_storage(std::move(v.m_storage)){}
		
	vector(std::initializer_list<T>  list) : m_storage(list.begin(),list.end()){}
		
	/// \brief Constructs the vector from a predefined range
	template<class Iter>
	vector(Iter begin, Iter end):m_storage(begin,end){}

	/// \brief Copy-constructor of a vector from a vector_expression
	/// \param e the vector_expression whose values will be duplicated into the vector
	template<class E>
	vector(vector_expression<E, cpu_tag> const& e):m_storage(e().size()) {
		assign(*this, e);
	}
	
	// -------------------
	// Assignment operators
	// -------------------
	
	/// \brief Assign a full vector (\e RHS-vector) to the current vector (\e LHS-vector)
	/// Assign a full vector (\e RHS-vector) to the current vector (\e LHS-vector). This method does not create any temporary.
	/// \param v is the source vector container
	/// \return a reference to a vector (i.e. the destination vector)
	vector& operator = (vector const& v) = default;
	
	/// \brief Move-Assign a full vector (\e RHS-vector) to the current vector (\e LHS-vector)
	/// \param v is the source vector container
	/// \return a reference to a vector (i.e. the destination vector)
	//~ vector& operator = (vector && v) = default; //vc++ can not default this. true story
	vector& operator = (vector && v){
		m_storage = std::move(v.m_storage);
		return *this;
	}
	
	/// \brief Assign a full vector (\e RHS-vector) to the current vector (\e LHS-vector)
	/// Assign a full vector (\e RHS-vector) to the current vector (\e LHS-vector). This method does not create any temporary.
	/// \param v is the source vector container
	/// \return a reference to a vector (i.e. the destination vector)
	template<class C>          // Container assignment without temporary
	vector& operator = (vector_container<C, cpu_tag> const& v) {
		resize(v().size());
		return assign(*this, v);
	}

	/// \brief Assign the result of a vector_expression to the vector
	/// Assign the result of a vector_expression to the vector.
	/// \param e is a const reference to the vector_expression
	/// \return a reference to the resulting vector
	template<class E>
	vector& operator = (vector_expression<E, cpu_tag> const& e) {
		self_type temporary(e);
		swap(*this,temporary);
		return *this;
	}

	// ---------
	// Storage interface
	// ---------
	
	/// \brief Return the size of the vector.
	size_type size() const {
		return m_storage.size();
	}
	
	///\brief Returns the underlying storage structure for low level access
	storage_type raw_storage(){
		return {m_storage.data(),1};
	}
	
	///\brief Returns the underlying storage structure for low level access
	const_storage_type raw_storage() const{
		return {m_storage.data(),1};
	}
	typename device_traits<cpu_tag>::queue_type& queue(){
		return device_traits<cpu_tag>::default_queue();
	}
	
	// ---------
	// High level interface
	// ---------

	/// \brief Return the maximum size of the data container.
	/// Return the upper bound (maximum size) on the data container. Depending on the container, it can be bigger than the current size of the vector.
	size_type max_size() const {
		return m_storage.max_size();
	}

	/// \brief Return true if the vector is empty (\c size==0)
	/// \return \c true if empty, \c false otherwise
	bool empty() const {
		return m_storage.empty();
	}

	/// \brief Resize the vector
	/// \param size new size of the vector
	void resize(size_type size) {
		m_storage.resize(size);
	}

	// --------------
	// Element access
	// --------------

	/// \brief Return a const reference to the element \f$i\f$
	/// Return a const reference to the element \f$i\f$. With some compilers, this notation will be faster than \c operator[]
	/// \param i index of the element
	const_reference operator()(size_type i) const {
		REMORA_RANGE_CHECK(i < size());
		return m_storage[i];
	}

	/// \brief Return a reference to the element \f$i\f$
	/// Return a reference to the element \f$i\f$. With some compilers, this notation will be faster than \c operator[]
	/// \param i index of the element
	reference operator()(size_type i) {
		REMORA_RANGE_CHECK(i < size());
		return m_storage[i];
	}

	/// \brief Return a const reference to the element \f$i\f$
	/// \param i index of the element
	const_reference operator [](size_type i) const {
		REMORA_RANGE_CHECK(i < size());
		return m_storage[i];
	}

	/// \brief Return a reference to the element \f$i\f$
	/// \param i index of the element
	reference operator [](size_type i) {
		REMORA_RANGE_CHECK(i < size());
		return m_storage[i];
	}
	
	///\brief Returns the first element of the vector
	reference front(){
		return m_storage[0];
	}
	///\brief Returns the first element of the vector
	const_reference front()const{
		return m_storage[0];
	}
	///\brief Returns the last element of the vector
	reference back(){
		return m_storage[size()-1];
	}
	///\brief Returns the last element of the vector
	const_reference back()const{
		return m_storage[size()-1];
	}
	
	///\brief resizes the vector by appending a new element to the end. this invalidates storage 
	void push_back(value_type const& element){
		m_storage.push_back(element);
	}

	/// \brief Clear the vector, i.e. set all values to the \c zero value.
	void clear() {
		std::fill(m_storage.begin(), m_storage.end(), value_type/*zero*/());
	}
	
	// Iterator types
	typedef iterators::dense_storage_iterator<value_type> iterator;
	typedef iterators::dense_storage_iterator<value_type const> const_iterator;
	
	/// \brief return an iterator on the first element of the vector
	const_iterator cbegin() const {
		return const_iterator(m_storage.data(),0);
	}

	/// \brief return an iterator after the last element of the vector
	const_iterator cend() const {
		return const_iterator(m_storage.data()+size(),size());
	}

	/// \brief return an iterator on the first element of the vector
	const_iterator begin() const {
		return cbegin();
	}

	/// \brief return an iterator after the last element of the vector
	const_iterator end() const {
		return cend();
	}

	/// \brief Return an iterator on the first element of the vector
	iterator begin(){
		return iterator(m_storage.data(),0);
	}

	/// \brief Return an iterator at the end of the vector
	iterator end(){
		return iterator(m_storage.data()+size(),size());
	}
	
	/////////////////sparse interface///////////////////////////////
	iterator set_element(iterator pos, size_type index, value_type value) {
		REMORA_SIZE_CHECK(pos.index() == index);
		(*this)(index) = value;
		
		return pos;
	}

	iterator clear_element(iterator pos) {
		REMORA_SIZE_CHECK(pos != end());
		v(pos.index()) = value_type();
		
		//return new iterator to the next element
		return pos+1;
	}
	
	iterator clear_range(iterator start, iterator end) {
		REMORA_RANGE_CHECK(start <= end);
		std::fill(start,end,value_type());
		return end;
	}
	
	void reserve(size_type) {}
	
	/// \brief Swap the content of two vectors
	/// \param v1 is the first vector. It takes values from v2
	/// \param v2 is the second vector It takes values from v1
	friend void swap(vector& v1, vector& v2) {
		v1.m_storage.swap(v2.m_storage);
	}
	// -------------
	// Serialization
	// -------------

	/// Serialize a vector into and archive as defined in Boost
	/// \param ar Archive object. Can be a flat file, an XML file or any other stream
	/// \param file_version Optional file version (not yet used)
	template<class Archive>
	void serialize(Archive &ar, const unsigned int file_version) {
		boost::serialization::collection_size_type count(size());
		ar & count;
		if(!Archive::is_saving::value){
			resize(count);
		}
		if (!empty())
			ar & boost::serialization::make_array(m_storage.data(),size());
		(void) file_version;//prevent warning
	}

private:
	array_type m_storage;
};


}

#endif
