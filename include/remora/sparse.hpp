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


#include "detail/traits.hpp"
#include "cpu/sparse.hpp"
#include "expression_types.hpp"
#include "assignment.hpp"


namespace remora{

template<class T, class I> class compressed_vector_reference;
template<class T, class I = std::size_t> class compressed_vector;
template<class T, class I = std::size_t, class Orientation = row_major> class compressed_matrix;
template<class T, class I, class Orientation> class compressed_matrix_reference;

template<class T,class I>
class compressed_vector_reference
: public vector_expression<compressed_vector_reference<T,I>, cpu_tag >
, public detail::BaseSparseVector<detail::VectorReference<compressed_vector<T,I> >, true >{
public:
	typedef I size_type;
	typedef T value_type;
	typedef value_type const& const_reference;
	typedef value_type& reference;
	
	typedef compressed_vector_reference<value_type const,size_type const> const_closure_type;
	typedef compressed_vector_reference closure_type;
	typedef sparse_vector_storage<T,I> storage_type;
	typedef sparse_vector_storage<value_type const,size_type const> const_storage_type;
	typedef elementwise<sparse_tag> evaluation_category;

	// Construction and destruction
	/// \brief Constructor of a vector proxy from a compressed vector
	compressed_vector_reference(compressed_vector<value_type,size_type>& v)
	: detail::BaseSparseVector<detail::VectorReference<compressed_vector<T,I> >, true >(
		detail::VectorReference<compressed_vector<T,I> >(&v),
		v.size(), 
		v.nnz()
	){}
	
	// Assignment
	compressed_vector_reference& operator = (compressed_vector_reference const& v){
		return assign(*this, typename vector_temporary<compressed_vector_reference>::type(v));
	}
	template<class E>
	compressed_vector_reference& operator = (vector_expression<E, cpu_tag> const& e) {
		return assign(*this, typename vector_temporary<E>::type(e));
	}
		
	///\brief Returns the underlying storage_type structure for low level access
	storage_type raw_storage() const{
		return this->m_storage;
	}

private:
	compressed_vector_reference(compressed_vector<value_type,size_type>&&);
};


template<class T,class I>
class compressed_vector_reference<T const, I const>
: public vector_expression<compressed_vector_reference<T const,I const>, cpu_tag >
, public detail::BaseSparseVector<detail::ConstantStorage<T const, I const>, false >{
public:
	//std::container types
	typedef I size_type;
	typedef T value_type;
	typedef value_type const& const_reference;
	typedef const_reference reference;
	
	typedef compressed_vector_reference const_closure_type;
	typedef const_closure_type closure_type;
	typedef sparse_vector_storage<value_type const,size_type const> const_storage_type;
	typedef const_storage_type storage_type;
	typedef elementwise<sparse_tag> evaluation_category;

	// Construction and destruction
	/// \brief Constructor of a vector proxy from a compressed vector
	compressed_vector_reference(compressed_vector<value_type,size_type> const& v)
	: detail::BaseSparseVector<detail::ConstantStorage<T const,I const>, false >(
		detail::ConstantStorage<T const, I const>(v.raw_storage()),
		v.size(), 
		v.nnz()
	){}
	
	//conversion non-const -> const
	compressed_vector_reference(compressed_vector_reference<T, I> const& v)
	: detail::BaseSparseVector<detail::ConstantStorage<T const,I const>, false >(
		detail::ConstantStorage<T const, I const>(v.raw_storage()),
		v.size(), 
		v.nnz()
	){}
		
	///\brief Returns the underlying storage_type structure for low level access
	const_storage_type raw_storage() const{
		return this->m_storage;
	}

private:
	compressed_vector_reference(compressed_vector<value_type,size_type>&&);
	compressed_vector_reference& operator = (compressed_vector_reference const&);
	compressed_vector_reference& operator = (compressed_vector_reference &&);
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
template<class T, class I>
class compressed_vector
:public vector_container<compressed_vector<T, I>, cpu_tag >
,public detail::BaseSparseVector<detail::VectorStorage<T,I >, true >{
public:
	typedef T value_type;
	typedef I size_type;
	typedef T const& const_reference;
	typedef T& reference;
	
	typedef compressed_vector_reference<T const,I const> const_closure_type;
	typedef compressed_vector_reference<T, I> closure_type;
	typedef sparse_vector_storage<T,I> storage_type;
	typedef sparse_vector_storage<T const,I const> const_storage_type;
	typedef elementwise<sparse_tag> evaluation_category;

	// Construction and destruction
	compressed_vector()
	: detail::BaseSparseVector<detail::VectorStorage<T,I>, true >(
		detail::VectorStorage<T, I>(),0,0
	){}
	explicit compressed_vector(size_type size, size_type non_zeros = 0)
	: detail::BaseSparseVector<detail::VectorStorage<T,I>, true >(
		detail::VectorStorage<T, I>(),size,non_zeros
	){}
	template<class E>
	compressed_vector(vector_expression<E, cpu_tag> const& e, size_type non_zeros = 0)
	: detail::BaseSparseVector<detail::VectorStorage<T,I>, true >(
		detail::VectorStorage<T, I>(),e.size(),non_zeros
	){
		assign(*this,e);
	}
	
	void resize(size_type size, bool keep){
		this->do_resize(size, keep);
	}
	
	///\brief Returns the underlying storage structure for low level access
	storage_type raw_storage(){
		return this->m_storage;
	}
	
	///\brief Returns the underlying storage structure for low level access
	const_storage_type raw_storage() const{
		return this->m_storage;
	}

	friend void swap(compressed_vector& v1, compressed_vector& v2){
		std::swap(v1.m_size, v2.m_size);
		v1.m_indices.swap(v2.m_indices);
		v1.m_values.swap(v2.m_values);
		v1.m_storage.swap(v2.m_storage);
	}

	// Assignment
	compressed_vector& operator = (compressed_vector const& v) = default;
	template<class C>          // Container assignment without temporary
	compressed_vector& operator = (vector_container<C, cpu_tag> const& v) {
		this->resize(v().size(), false);
		assign(*this, v);
		return *this;
	}
	template<class AE>
	compressed_vector& operator = (vector_expression<AE, cpu_tag> const& ae) {
		compressed_vector temporary(ae, this->nnz_capacity());
		swap(temporary);
		return *this;
	}

	// Serialization
	template<class Archive>
	void serialize(Archive& ar, const unsigned int /* file_version */) {
		ar & this->m_manager;
		ar & boost::serialization::make_nvp("size", this->m_size());
		if (Archive::is_loading::value) {
			this->m_storage = this->m_manager.reserve(this->nnz());
		}
	}
};

//~ template<class T, class I, class Orientation> 
//~ class compressed_matrix_reference: public matrix_expression<compressed_matrix_reference<T, I, Orientation> >{
//~ public:
	//~ typedef I size_type;
	//~ typedef T value_type;
	//~ typedef T const& const_reference;
	//~ typedef T& reference;
	//~ typedef compressed_matrix_reference<T const,I const> const_closure_type;
	//~ typedef compressed_matrix_reference<T, I> closure_type;
	//~ typedef sparse_matrix_storage<T,I> storage_type;
	//~ typedef sparse_matrix_storage<value_type const,size_type const> const_storage_type;
	//~ typedef elementwise<sparse_tag> evaluation_category;
	//~ typedef row_major orientation;

	//~ compressed_matrix_reference(compressed_matrix<T,I,Orientation>& matrix)
	//~ :m_impl(detail::MatrixReference<compressed_matrix<T,I> >(matrix));
	
	//~ size_type size1() const {
		//~ return orientation::index_M(m_impl.major_size(),m_impl.minor_size());
	//~ }
	//~ size_type size2() const {
		//~ return orientation::index_m(m_impl.major_size(),m_impl.minor_size());
	//~ }
	
	//~ /// \brief Number of nonzeros this matrix can maximally store before requiring new memory
	//~ std::size_t nnz_capacity() const{
		//~ return return m_impl.capacity();
	//~ }
	//~ /// \brief Total Number of nonzeros this matrix stores
	//~ std::size_t nnz() const {
		//~ return m_impl.nnz(i);
	//~ }
	//~ /// \brief Number of nonzeros the major index (a major or column depending on orientation) can maximally store before a resize
	//~ std::size_t major_capacity(size_type i)const{
		//~ return m_impl.major_capacity(i);
	//~ }
	//~ /// \brief Number of nonzeros the major index (a major or column depending on orientation) currently stores
	//~ std::size_t major_nnz(size_type i) const {
		//~ return m_impl.major_nnz(i);
	//~ }

	//~ /// \brief Set the total number of nonzeros stored by the matrix
	//~ void set_nnz(std::size_t non_zeros) {
		//~ m_impl.set_nnz(non_zeros);
	//~ }
	//~ /// \brief Set the number of nonzeros stored in the major index (a major or column depending on orientation)
	//~ void set_major_nnz(size_type i,std::size_t non_zeros) {
		//~ m_impl.set_major_nnz(i,non_zeros);
	//~ }
	
	//~ storage_type raw_storage()const{
		//~ return m_impl.raw_storage();
	//~ }
	
	//~ void reserve(std::size_t non_zeros) {
		//~ m_impl.reserve(non_zeros);
	//~ }

	//~ void major_reserve(size_type i, std::size_t non_zeros, bool exact_size = false) {
		//~ m_impl.major_reserve(i, non_zeros, exact_size);
	//~ }
	
	//~ typedef typename detail::sparse_matrix_impl<detail::MatrixReference<compressed_matrix<T,I> > >::const_major_iterator const_major_iterator;
	//~ typedef typename detail::sparse_matrix_impl<detail::MatrixReference<compressed_matrix<T,I> > >::major_iterator major_iterator;

	//~ const_major_iterator major_begin(size_type i) const {
		//~ return m_impl.cmajor_begin(i);
	//~ }

	//~ const_major_iterator major_end(size_type i) const{
		//~ return m_impl.cmajor_end(i);
	//~ }

	//~ major_iterator major_begin(size_type i) {
		//~ return m_impl.major_begin(i);
	//~ }

	//~ major_iterator major_end(size_type i) {
		//~ return m_impl.major_end(i);
	//~ }
	
	//~ major_iterator set_element(major_iterator pos, size_type index, value_type value){
		//~ m_impl.set_element(pos, index, value);
	//~ }

	//~ major_iterator clear_range(major_iterator start, major_iterator end) {
		//~ m_impl.clear_range(start,end);
	//~ }

//~ private:
	//~ compressed_matrix_reference(compressed_matrix<value_type,size_type>&&);
	//~ detail::sparse_matrix_impl<detail::MatrixReference<compressed_matrix<T,I> > > m_impl;
//~ };

//~ template<class T, class I, class Orientation> 
//~ class compressed_matrix_reference<T const, I const, Orientation> : public matrix_expression<compressed_matrix_reference<T, I, Orientation> >{
//~ public:
	//~ typedef I size_type;
	//~ typedef T value_type;
	//~ typedef T const& const_reference;
	//~ typedef T const& reference;
	
	//~ typedef compressed_matrix_reference<T const,I const> const_closure_type;
	//~ typedef const_closure_type closure_type;
	//~ typedef sparse_matrix_storage<T const,I const> const_storage_type;
	//~ typedef const_storage_type storage_type;
	//~ typedef elementwise<sparse_tag> evaluation_category;
	//~ typedef row_major orientation;

	//~ compressed_matrix_reference(compressed_matrix<T,I,Orientation> const& matrix)
	//~ :m_impl(detail::ConstantMatrixStorage<T const, I const>(matrix.raw_storage())){}
	
	//~ compressed_matrix_reference(compressed_matrix_reference<T,I,Orientation> const& matrix)
	//~ :m_impl(detail::ConstantMatrixStorage<T const, I const>(matrix.raw_storage())){}
	
	//~ size_type size1() const {
		//~ return orientation::index_M(m_impl.major_size(),m_impl.minor_size());
	//~ }
	//~ size_type size2() const {
		//~ return orientation::index_m(m_impl.major_size(),m_impl.minor_size());
	//~ }
	
	//~ /// \brief Number of nonzeros this matrix can maximally store before requiring new memory
	//~ std::size_t nnz_capacity() const{
		//~ return return m_impl.nnz();
	//~ }
	//~ /// \brief Total Number of nonzeros this matrix stores
	//~ std::size_t nnz() const {
		//~ return m_impl.nnz(i);
	//~ }
	//~ /// \brief Number of nonzeros the major index (a major or column depending on orientation) can maximally store before a resize
	//~ std::size_t major_capacity(size_type i)const{
		//~ return m_impl.major_nnz(i);
	//~ }
	//~ /// \brief Number of nonzeros the major index (a major or column depending on orientation) currently stores
	//~ std::size_t major_nnz(size_type i) const {
		//~ return m_impl.major_nnz(i);
	//~ }
	
	//~ const_storage_type raw_storage()const{
		//~ return m_impl.raw_storage();
	//~ }
	
	//~ typedef typename detail::sparse_matrix_impl<detail::ConstantMatrixStorage<T const, I const> >::const_major_iterator const_major_iterator;
	//~ typedef typename detail::sparse_matrix_impl<detail::ConstantMatrixStorage<T const, I const> >::major_iterator major_iterator;

	//~ const_major_iterator major_begin(size_type i) const {
		//~ return m_impl.cmajor_begin(i);
	//~ }

	//~ const_major_iterator major_end(size_type i) const{
		//~ return m_impl.cmajor_end(i);
	//~ }

//~ private:
	//~ detail::sparse_matrix_impl<detail::ConstantMatrixStorage<T const, I const> > m_impl;
	//~ compressed_matrix_reference(compressed_matrix<value_type,size_type>&&);
//~ };



//~ template<class T, class I=std::size_t>
//~ class compressed_matrix:public matrix_container<compressed_matrix<T, I>, cpu_tag >{
	//~ typedef compressed_matrix<T, I> self_type;
//~ public:
	//~ typedef I size_type;
	//~ typedef T value_type;
	//~ typedef T const& const_reference;
	//~ typedef T& reference;
	
	//~ typedef matrix_reference<self_type const> const_closure_type;
	//~ typedef matrix_reference<self_type> closure_type;
	//~ typedef sparse_matrix_storage<T,I> storage_type;
	//~ typedef sparse_matrix_storage<value_type const,size_type const> const_storage_type;
	//~ typedef elementwise<sparse_tag> evaluation_category;
	//~ typedef row_major orientation;

	//~ compressed_matrix():m_impl(detail::MatrixStorage<T,I>(0,0,0)){}
	
	//~ compressed_matrix(size_type rows, size_type cols, size_type non_zeros = 0)
	//~ :m_impl(detail::MatrixStorage<T,I>(orientation::index_M(rows,cols),orientation::index_m(rows,cols),non_zeros)){}
	
	//~ compressed_matrix(matrix_expression<E, cpu_tag> const& m, size_type non_zeros = 0)
	//~ :m_impl(detail::MatrixStorage<T,I>(orientation::index_M(m.size1(), m.size2()),orientation::index_m(m.size1(), m.size2()),v)){
		//~ assign(*this,m);
	//~ }
	
	//~ compressed_matrix operator=(matrix_container<E, cpu_tag> const& m){
		//~ resize(m.size1(),m.size2());
		//~ assign(*this,m);
		//~ return *this;
	//~ }
	
	//~ compressed_matrix& operator=(matrix_expression<E, cpu_tag> const& m){
		//~ vector temporary(e);
		//~ swap(*this,temporary);
		//~ return *this;
	//~ }

	//~ /// \brief Number of nonzeros this matrix can maximally store before requiring new memory
	//~ std::size_t nnz_capacity() const{
		//~ return return m_impl.capacity();
	//~ }
	//~ /// \brief Total Number of nonzeros this matrix stores
	//~ std::size_t nnz() const {
		//~ return m_impl.nnz(i);
	//~ }
	//~ /// \brief Number of nonzeros the major index (a major or column depending on orientation) can maximally store before a resize
	//~ std::size_t major_capacity(size_type i)const{
		//~ return m_impl.major_capacity(i);
	//~ }
	//~ /// \brief Number of nonzeros the major index (a major or column depending on orientation) currently stores
	//~ std::size_t major_nnz(size_type i) const {
		//~ return m_impl.major_nnz(i);
	//~ }

	//~ /// \brief Set the total number of nonzeros stored by the matrix
	//~ void set_nnz(std::size_t non_zeros) {
		//~ m_impl.set_nnz(non_zeros);
	//~ }
	//~ /// \brief Set the number of nonzeros stored in the major index (a major or column depending on orientation)
	//~ void set_major_nnz(size_type i,std::size_t non_zeros) {
		//~ m_impl.set_major_nnz(i,non_zeros);
	//~ }
	
	//~ const_storage_type raw_storage()const{
		//~ return m_impl.raw_storage();
	//~ }
	//~ storage_type raw_storage(){
		//~ return m_impl.raw_storage();
	//~ }
	
	//~ void reserve(std::size_t non_zeros) {
		//~ m_impl.reserve(non_zeros);
	//~ }

	//~ void major_reserve(size_type i, std::size_t non_zeros, bool exact_size = false) {
		//~ m_impl.major_reserve(i, non_zeros, exact_size);
	//~ }

	//~ void resize(size_type rows, size_type columns){
		//~ m_impl.resize(orientation::index_M(rows,columns),orientation::index_m(rows,columns));
	//~ }
	
	//~ typedef typename detail::sparse_matrix_impl<MatrixStorage<T,I> >::const_major_iterator const_major_iterator;
	//~ typedef typename detail::sparse_matrix_impl<MatrixStorage<T,I> >::major_iterator major_iterator;

	//~ const_major_iterator major_begin(size_type i) const {
		//~ return m_impl.cmajor_begin(i);
	//~ }

	//~ const_major_iterator major_end(size_type i) const{
		//~ return m_impl.cmajor_end(i);
	//~ }

	//~ major_iterator major_begin(size_type i) {
		//~ return m_impl.major_begin(i);
	//~ }

	//~ major_iterator major_end(size_type i) {
		//~ return m_impl.major_end(i);
	//~ }
	
	//~ major_iterator set_element(major_iterator pos, size_type index, value_type value){
		//~ m_impl.set_element(pos, index, value);
	//~ }

	//~ major_iterator clear_range(major_iterator start, major_iterator end) {
		//~ m_impl.clear_range(start,end);
	//~ }
	//~ // Serialization
	//~ template<class Archive>
	//~ void serialize(Archive &ar, const unsigned int /* file_version */) {
		//~ ar & m_impl;
	//~ }

//~ private:
	//~ detail::sparse_matrix_impl<detail::MatrixStorage<T,I> > m_impl;
//~ };




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
