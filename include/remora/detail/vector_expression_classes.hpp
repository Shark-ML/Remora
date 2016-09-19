/*!
 * \brief       Classes used for vector expressions
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
 #ifndef SHARK_LINALG_BLAS_VECTOR_EXPRESSION_CLASSES_HPP
#define SHARK_LINALG_BLAS_VECTOR_EXPRESSION_CLASSES_HPP

#include "../assignment.hpp"
#include "iterator.hpp"
#include "traits.hpp"

namespace shark {
namespace blas {

///\brief Implements multiplications of a vector by a scalar
template<class E>
class vector_scalar_multiply:public vector_expression<vector_scalar_multiply <E>, typename E::device_type > {
public:
	typedef typename E::const_closure_type expression_closure_type;
	typedef typename E::index_type index_type;
	typedef typename E::value_type value_type;
	typedef typename E::scalar_type scalar_type;
	typedef value_type const_reference;
	typedef value_type reference;

	typedef vector_scalar_multiply const_closure_type;
	typedef vector_scalar_multiply closure_type;
	typedef unknown_storage storage_type;
	typedef unknown_storage const_storage_type;
	typedef typename E::evaluation_category evaluation_category;
	typedef typename E::device_type device_type;

	// Construction and destruction
	// May be used as mutable expression.
	vector_scalar_multiply(expression_closure_type const& e, scalar_type scalar):
		m_expression(e), m_scalar(scalar) {}

	// Accessors
	index_type size() const {
		return m_expression.size();
	}

	// Expression accessors
	expression_closure_type const &expression() const {
		return m_expression;
	}
	
	scalar_type scalar()const{
		return m_scalar;
	}

public:
	// Element access
	const_reference operator()(index_type i) const {
		return m_scalar * m_expression(i);
	}

	const_reference operator[](index_type i) const {
		return m_scalar * m_expression(i);
	}
	
	
	//computation kernels
	template<class VecX>
	void assign_to(vector_expression<VecX, device_type>& x, scalar_type alpha = scalar_type(1) )const{
		m_expression.assign_to(x,alpha*m_scalar);
	}
	template<class VecX>
	void plus_assign_to(vector_expression<VecX, device_type>& x, scalar_type alpha = scalar_type(1) )const{
		m_expression.plus_assign_to(x,alpha*m_scalar);
	}
	
	template<class VecX>
	void minus_assign_to(vector_expression<VecX, device_type>& x, scalar_type alpha = scalar_type(1) )const{
		m_expression.minus_assign_to(x,alpha*m_scalar);
	}

	
	//iterators
	typedef transform_iterator<typename E::const_iterator,scalar_multiply1<value_type, scalar_type> > const_iterator;
	typedef const_iterator iterator;
	
	const_iterator begin() const {
		return const_iterator(m_expression.begin(),scalar_multiply1<value_type, scalar_type>(m_scalar));
	}
	const_iterator end() const {
		return const_iterator(m_expression.end(),scalar_multiply1<value_type, scalar_type>(m_scalar));
	}
private:
	expression_closure_type m_expression;
	scalar_type m_scalar;
};

/// \brief Vector expression representing a constant valued vector.
template<class T>
class scalar_vector:public vector_expression<scalar_vector<T>, cpu_tag > {
public:
	typedef std::size_t index_type;
	typedef T value_type;
	typedef T scalar_type;
	typedef const T& const_reference;
	typedef const_reference reference;

	typedef scalar_vector const_closure_type;
	typedef scalar_vector closure_type;
	typedef unknown_storage storage_type;
	typedef unknown_storage const_storage_type;
	typedef elementwise<dense_tag> evaluation_category;
	
	// Construction and destruction
	scalar_vector()
	:m_size(0), m_value() {}
	explicit scalar_vector(index_type size, value_type value)
	:m_size(size), m_value(value) {}
	scalar_vector(const scalar_vector& v)
	:m_size(v.m_size), m_value(v.m_value) {}

	// Accessors
	index_type size() const {
		return m_size;
	}

	// Element access
	const_reference operator()(index_type /*i*/) const {
		return m_value;
	}

	const_reference operator [](index_type /*i*/) const {
		return m_value;
	}

public:
	typedef constant_iterator<T> iterator;
	typedef constant_iterator<T> const_iterator;

	const_iterator begin() const {
		return const_iterator(0,m_value);
	}
	const_iterator end() const {
		return const_iterator(m_size,m_value);
	}

private:
	index_type m_size;
	value_type m_value;
};

///\brief Class implementing vector transformation expressions.
///
///transforms a vector Expression e of type E using a Function f of type F as an elementwise transformation f(e(i))
///This transformation needs f to be constant, meaning that applying f(x), f(y), f(z) yields the same results independent of the
///order of application. Also F must provide a type F::result_type indicating the result type of the functor.
template<class E, class F>
class vector_unary: public vector_expression<vector_unary<E, F>, typename E::device_type > {
public:
	typedef F functor_type;
	typedef typename E::const_closure_type expression_closure_type;
	typedef typename E::index_type index_type;
	typedef typename F::result_type value_type;
	typedef value_type scalar_type;
	typedef value_type const_reference;
	typedef value_type reference;
	typedef vector_unary const_closure_type;
	typedef vector_unary closure_type;
	typedef unknown_storage storage_type;
	typedef unknown_storage const_storage_type;
	typedef typename E::evaluation_category evaluation_category;
	typedef typename E::device_type device_type;
	// Construction and destruction
	// May be used as mutable expression.
	vector_unary(expression_closure_type const& e, F const &functor):
		m_expression(e), m_functor(functor) {}

	// Accessors
	index_type size() const {
		return m_expression.size();
	}

	// Expression accessors
	expression_closure_type const &expression() const {
		return m_expression;
	}
	
	functor_type const& functor()const{
		return m_functor;
	}
	
	//computation kernels
	template<class VecX>
	void assign_to(vector_expression<VecX, device_type>& x, scalar_type alpha = scalar_type(1) )const{
		x().clear();
		plus_assign_to(x,eval_block(m_expression), alpha);
	}
	template<class VecX>
	void plus_assign_to(vector_expression<VecX, device_type>& x, scalar_type alpha = scalar_type(1) )const{
		plus_assign_to(x,eval_block(m_expression), alpha);
	}
	
	template<class VecX>
	void minus_assign_to(vector_expression<VecX, device_type>& x, scalar_type alpha = scalar_type(1) )const{
		plus_assign_to(x,eval_block(m_expression), -alpha);
	}

	// Element access
	const_reference operator()(index_type i) const {
		return m_functor(m_expression(i));
	}

	const_reference operator[](index_type i) const {
		return m_functor(m_expression[i]);
	}

	typedef transform_iterator<typename E::const_iterator,functor_type> const_iterator;
	typedef const_iterator iterator;

	// Element lookup
	const_iterator begin() const {
		return const_iterator(m_expression.begin(),m_functor);
	}
	const_iterator end() const {
		return const_iterator(m_expression.end(),m_functor);
	}
private:
	template<class VecX, class VecV>
	void plus_assign_to(
		vector_expression<VecX, device_type>& x,
		vector_expression<VecV, device_type> const& v,
		scalar_type alpha
	)const{
		vector_unary<VecV, F> e(v(), m_functor);
		vector_scalar_multiply<vector_unary<VecV,F> > e1(e,alpha);
		plus_assign(x,e1);
	}

	expression_closure_type m_expression;
	F m_functor;
};

template<class E1, class E2>
class vector_addition: public vector_expression<vector_addition<E1,E2>, typename E1::device_type > {
private:
	typedef scalar_binary_plus<
		typename E1::value_type,
		typename E2::value_type
	> functor_type;
public:
	typedef typename functor_type::result_type value_type;
	typedef value_type scalar_type;
	typedef value_type const_reference;
	typedef value_type reference;
	typedef typename E1::index_type index_type;

	typedef typename E1::const_closure_type lhs_closure_type;
	typedef typename E2::const_closure_type rhs_closure_type;
	
	typedef vector_addition const_closure_type;
	typedef vector_addition closure_type;
	typedef unknown_storage storage_type;
	typedef unknown_storage const_storage_type;
	typedef typename evaluation_restrict_traits<E1,E2>::type evaluation_category;
	typedef typename E1::device_type device_type;

	// Construction and destruction
	explicit vector_addition (
		lhs_closure_type e1, 
		rhs_closure_type e2
	):m_lhs(e1),m_rhs(e2){
		SIZE_CHECK(e1.size() == e2.size());
	}

	// Accessors
	index_type size() const {
		return m_lhs.size();
	}

	// Expression accessors
	lhs_closure_type const& lhs() const {
		return m_lhs;
	}
	rhs_closure_type const& rhs() const {
		return m_rhs;
	}

	// Element access
	const_reference operator() (index_type i) const {
		SIZE_CHECK(i < size());
		return m_lhs(i) + m_rhs(i);
	}

	const_reference operator[] (index_type i) const {
		SIZE_CHECK(i < size());
		return m_lhs(i) + m_rhs(i);
	}
	
	//computation kernels
	template<class VecX>
	void assign_to(vector_expression<VecX, device_type>& x, scalar_type alpha = scalar_type(1) )const{
		assign(x,alpha*m_lhs);
		plus_assign(x,alpha*m_rhs);
	}
	template<class VecX>
	void plus_assign_to(vector_expression<VecX, device_type>& x, scalar_type alpha = scalar_type(1) )const{
		plus_assign(x,alpha*m_lhs);
		plus_assign(x,alpha*m_rhs);
	}
	
	template<class VecX>
	void minus_assign_to(vector_expression<VecX, device_type>& x, scalar_type alpha = scalar_type(1) )const{
		minus_assign(x,alpha*m_lhs);
		minus_assign(x,alpha*m_rhs);
	}

	// Iterator types
	typedef binary_transform_iterator<
		typename E1::const_iterator,
		typename E2::const_iterator,
		functor_type
	> const_iterator;
	typedef const_iterator iterator;

	const_iterator begin () const {
		return const_iterator(functor_type(),
			m_lhs.begin(),m_lhs.end(),
			m_rhs.begin(),m_rhs.end()
		);
	}
	const_iterator end() const {
		return const_iterator(functor_type(),
			m_lhs.end(),m_lhs.end(),
			m_rhs.end(),m_rhs.end()
		);
	}

private:
	lhs_closure_type m_lhs;
	rhs_closure_type m_rhs;
};

template<class E1, class E2, class F>
class vector_binary:public vector_expression<vector_binary<E1,E2, F>,typename E1::device_type > {
	typedef typename E1::const_closure_type lhs_closure_type;
	typedef typename E2::const_closure_type rhs_closure_type;
public:
	typedef F functor_type;
	typedef typename functor_type::result_type value_type;
	typedef typename E1::index_type index_type;
	typedef value_type scalar_type;
	typedef value_type const_reference;
	typedef value_type reference;
	
	typedef vector_binary const_closure_type;
	typedef vector_binary closure_type;
	typedef unknown_storage storage_type;
	typedef unknown_storage const_storage_type;
	typedef typename evaluation_restrict_traits<E1,E2>::type evaluation_category;
	typedef typename E1::device_type device_type;
	// Construction and destruction
	explicit vector_binary (
		lhs_closure_type e1, 
		rhs_closure_type e2,
		F functor
	):m_lhs(e1),m_rhs(e2), m_functor(functor) {
		SIZE_CHECK(e1.size() == e2.size());
	}

	// Accessors
	index_type size() const {
		return m_lhs.size ();
	}

	// Expression accessors
	lhs_closure_type const& lhs() const {
		return m_lhs;
	}
	rhs_closure_type const& rhs() const {
		return m_rhs;
	}
	
	functor_type const& functor()const{
		return m_functor;
	}

	// Element access
	const_reference operator() (index_type i) const {
		SIZE_CHECK(i < size());
		return m_functor(m_lhs(i),m_rhs(i));
	}

	const_reference operator[] (index_type i) const {
		SIZE_CHECK(i < size());
		return m_functor(m_lhs(i),m_rhs(i));
	}
	
	template<class VecX>
	void assign_to(vector_expression<VecX, device_type>& x, scalar_type alpha = scalar_type(1) )const{
		x().clear();
		plus_assign_to(x,eval_block(m_lhs), eval_block(m_rhs), alpha);
	}
	template<class VecX>
	void plus_assign_to(vector_expression<VecX, device_type>& x, scalar_type alpha = scalar_type(1) )const{
		plus_assign_to(x,eval_block(m_lhs), eval_block(m_rhs), alpha);
	}
	
	template<class VecX>
	void minus_assign_to(vector_expression<VecX, device_type>& x, scalar_type alpha = scalar_type(1) )const{
		plus_assign_to(x,eval_block(m_lhs), eval_block(m_rhs), -alpha);
	}

	// Iterator types
	
	// Iterator enhances the iterator of the referenced expressions
	// with the unary functor.
	typedef binary_transform_iterator<
		typename E1::const_iterator,
		typename E2::const_iterator,
		functor_type
	> const_iterator;
	typedef const_iterator iterator;

	const_iterator begin () const {
		return const_iterator (m_functor,
			m_lhs.begin(),m_lhs.end(),
			m_rhs.begin(),m_rhs.end()
		);
	}
	const_iterator end() const {
		return const_iterator (m_functor,
			m_lhs.end(),m_lhs.end(),
			m_rhs.end(),m_rhs.end()
		);
	}

private:
	lhs_closure_type m_lhs;
	rhs_closure_type m_rhs;
	functor_type m_functor;

	template<class VecX, class LHS, class RHS>
	void plus_assign_to(
		vector_expression<VecX, device_type>& x,
		vector_expression<LHS, device_type> const& lhs, vector_expression<RHS, device_type> const& rhs,
		scalar_type alpha
	)const{
		//we know that lhs and rhs are elementwise expressions so we can now create the elementwise expression and assign it.
		vector_binary<LHS,RHS, F> e(lhs(),rhs(), m_functor);
		vector_scalar_multiply<vector_binary<LHS,RHS,F> > e1(e,alpha);
		plus_assign(x,e);
	}
};

}}
#endif