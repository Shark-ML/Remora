/*!
 * \brief       Classes used for matrix proxies
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
 #ifndef REMORA_MATRIX_PROXY_CLASSES_HPP
#define REMORA_MATRIX_PROXY_CLASSES_HPP

#include "traits.hpp"
#include "../expression_types.hpp"
#include "../assignment.hpp"

#include <type_traits>
namespace remora{

// Matrix based vector range class representing (off-)diagonals of a matrix.
template<class M>
class matrix_vector_range: public vector_expression<matrix_vector_range<M>, typename M::device_type > {
private:
	typedef typename closure<M>::type matrix_closure_type;
public:
	typedef typename M::value_type value_type;
	typedef typename M::const_reference const_reference;
	typedef typename reference<M>::type reference;
	typedef typename M::size_type size_type;

	typedef matrix_vector_range<M> closure_type;
	typedef matrix_vector_range<typename const_expression<M>::type> const_closure_type;
	typedef typename storage<M>::type::diag_storage storage_type;
	typedef typename M::const_storage_type const_storage_type;
	typedef typename M::evaluation_category evaluation_category;
	typedef typename M::device_type device_type;

	// Construction and destruction
	matrix_vector_range(matrix_closure_type expression, size_type start1, size_type end1, size_type start2, size_type end2)
	:m_expression(expression), m_start1(start1), m_start2(start2), m_size(end1-start1){
		REMORA_SIZE_CHECK(start1 <= expression.size1());
		REMORA_SIZE_CHECK(end1 <= expression.size1());
		REMORA_SIZE_CHECK(start2 <= expression.size2());
		REMORA_SIZE_CHECK(end2 <= expression.size2());
		REMORA_SIZE_CHECK(m_size == end2-start2);
	}
	
	template<class E>
	matrix_vector_range(matrix_vector_range<E> const& other)
	: m_expression(other.expression())
	, m_start1(other.start1())
	, m_start2(other.start2()), m_size(other.size()){}
	
	// Accessors
	size_type start1() const {
		return m_start1;
	}
	size_type start2() const {
		return m_start2;
	}
	
	matrix_closure_type const& expression() const {
		return m_expression;
	}
	matrix_closure_type& expression() {
		return m_expression;
	}
	
	///\brief Returns the size of the vector
	size_type size() const {
		return m_size;
	}
	
	/// \brief Returns the underlying storage_type structure for low level access
	storage_type raw_storage()const{
		return m_expression.raw_storage().diag();
	}

	typename device_traits<typename M::device_type>::queue_type& queue()const{
		return m_expression.queue();
	}

	// ---------
	// High level interface
	// ---------

	// Element access
	reference operator()(std::size_t i) const{
		return m_expression(start1()+i,start2()+i);
	}
	reference operator [](size_type i) const {
		return (*this)(i);
	}
	
	void set_element(size_type i,value_type t){
		expression().set_element(start1()+i,start2()+i,t);
	}
	// Assignment
	
	template<class E>
	matrix_vector_range& operator = (vector_expression<E, typename M::device_type> const& e) {
		return assign(*this, typename vector_temporary<M>::type(e));
	}

	typedef typename device_traits<typename M::device_type>:: template indexed_iterator<closure_type>::type iterator;
	typedef typename device_traits<typename M::device_type>:: template indexed_iterator<const_closure_type>::type const_iterator;

	// Element lookup
	const_iterator begin()const{
		return const_iterator(*this, 0);
	}
	const_iterator end()const{
		return const_iterator(*this, size());
	}

	iterator begin() {
		return iterator(*this, 0);
	}
	iterator end() {
		return iterator(*this, size());
	}
	
	void reserve(){}
	void reserve_row(size_type, size_type) {}
	void reserve_column(size_type, size_type ){}

private:
	matrix_closure_type m_expression;
	size_type m_start1;
	size_type m_start2;
	size_type m_size;
};

template<class M>
class linearized_matrix: public vector_expression<linearized_matrix<M>, typename M::device_type > {
private:
	typedef typename closure<M>::type matrix_closure_type;
	typedef typename std::conditional<
		std::is_same<typename M::orientation::orientation, unknown_orientation>::value,
		row_major,
		typename M::orientation::orientation
	>::type orientation;

	template<class IndexExpr>
	struct OperatorReturn{//workaround for gcc 4.6 which would not like the type below inside a function signature.
		typedef decltype(device_traits<typename matrix_closure_type::device_type>::linearized_matrix_element(
			std::declval<matrix_closure_type const&>(),std::declval<IndexExpr const&>()
		)) type;
	};
public:
	typedef typename M::value_type value_type;
	typedef typename M::const_reference const_reference;
	typedef typename reference<M>::type reference;
	typedef typename M::size_type size_type;

	typedef linearized_matrix<M> closure_type;
	typedef linearized_matrix<typename const_expression<M>::type> const_closure_type;
	typedef unknown_storage storage_type;
	typedef storage_type const_storage_type;
	typedef typename M::evaluation_category evaluation_category;
	typedef typename M::device_type device_type;

	// Construction and destruction
	linearized_matrix(matrix_closure_type expression)
	:m_expression(expression){}
	
	template<class E>
	linearized_matrix(linearized_matrix<E> const& other)
	: m_expression(other.expression()){}
	
	matrix_closure_type const& expression() const {
		return m_expression;
	}
	matrix_closure_type& expression() {
		return m_expression;
	}
	
	///\brief Returns the size of the vector
	size_type size() const {
		return m_expression.size1() * m_expression.size2();
	}
	
	typename device_traits<typename M::device_type>::queue_type& queue()const{
		return m_expression.queue();
	}

	// ---------
	// High level interface
	// ---------

	// Element access
	template <class IndexExpr>
	typename OperatorReturn<IndexExpr>::type operator()(IndexExpr const& i) const{
		return device_traits<typename M::device_type>::linearized_matrix_element(m_expression,i);
	}
	reference operator [](size_type i) const{
		return (*this)(i);
	}
	
	void set_element(size_type i,value_type t){
		(*this)(i) = t;
	}

	// Assignment
	
	template<class E>
	linearized_matrix& operator = (vector_expression<E, typename M::device_type> const& e) {
		return assign(*this, typename vector_temporary<M>::type(e));
	}

	typedef typename device_traits<typename M::device_type>:: template indexed_iterator<closure_type>::type iterator;
	typedef typename device_traits<typename M::device_type>:: template indexed_iterator<const_closure_type>::type const_iterator;

	// Element lookup
	const_iterator begin()const{
		return const_iterator(*this, 0);
	}
	const_iterator end()const{
		return const_iterator(*this, size());
	}

	iterator begin() {
		return iterator(*this, 0);
	}
	iterator end() {
		return iterator(*this, size());
	}
	
	void reserve(){}
	void reserve_row(size_type, size_type) {}
	void reserve_column(size_type, size_type ){}

private:
	matrix_closure_type m_expression;
};


//~ template<class M, class Device>
//~ typename std::enable_if<
	//~ std::is_same<typename M::evaluation_category::tag,dense_tag>::value,
	//~ linearized_matrix<typename const_expression<M>::type>
//~ >::type 


template<class E>
struct ExpressionToFunctor<matrix_vector_range<E> >{
	typedef typename E::device_type device_type;
	static auto transform(matrix_vector_range<E> const& e) -> decltype(device_traits<device_type>::make_compose_binary(
		typename device_traits<device_type>::template add_scalar<std::size_t>(0),
		typename device_traits<device_type>::template add_scalar<std::size_t>(0),
		to_functor(e.expression())
	)){
		return device_traits<device_type>::make_compose_binary(
			typename device_traits<device_type>::template add_scalar<std::size_t>(e.start1()),
			typename device_traits<device_type>::template add_scalar<std::size_t>(e.start2()),
			to_functor(e.expression())
		);
	}
};


template<class E>
struct ExpressionToFunctor<linearized_matrix<E> >{
private:
	typedef device_traits<typename E::device_type> traits;
	template<class C>
	static auto functor(C const& e, row_major) -> decltype(traits::make_compose_binary(
		typename traits::template divide_scalar<std::size_t>(e.size2()),
		typename traits::template modulo_scalar<std::size_t>(e.size2()),
		to_functor(std::declval<C const&>())
	)){
		return traits::make_compose_binary(
			typename traits::template divide_scalar<std::size_t>(e.size2()),
			typename traits::template modulo_scalar<std::size_t>(e.size2()),
			to_functor(e)
		);
	}
	template<class C>
	static auto functor(C const& e, column_major) -> decltype(traits::make_compose_binary(
		typename traits::template modulo_scalar<std::size_t>(e.size1()),
		typename traits::template divide_scalar<std::size_t>(e.size1()),
		to_functor(std::declval<C const&>())
	)){
		return traits::make_compose_binary(
			typename traits::template modulo_scalar<std::size_t>(e.size1()),
			typename traits::template divide_scalar<std::size_t>(e.size1()),
			to_functor(e)
		);
	}
public:
	static auto transform(linearized_matrix<E> const& e) -> decltype(functor(std::declval<E const&>(), typename E::orientation())){
		return functor(e.expression(), typename E::orientation());
	}
};


}


#ifdef REMORA_USE_GPU
#include "../gpu/matrix_proxy_classes.hpp"
#endif

#endif
