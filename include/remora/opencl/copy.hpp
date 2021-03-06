/*!
 * \brief       Implements operations to copy data from cpu to opencl and back
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
#ifndef REMORA_OPENCL_COPY_HPP
#define REMORA_OPENCL_COPY_HPP

#include "../detail/traits.hpp"
#include "../dense.hpp" //required for vector proxy on cpu
#include "../assignment.hpp"

namespace remora{namespace opencl{

///////////////////////////////////////
//////// Vector Transport
///////////////////////////////////////	

template<class E>
class vector_transport_to_host: public vector_expression<vector_transport_to_host<E>, cpu_tag>{
public:
	typedef typename E::const_closure_type expression_closure_type;

	typedef typename E::value_type value_type;
	typedef typename E::size_type size_type;
	typedef value_type const& const_reference;
	typedef const_reference reference;

	typedef vector_transport_to_host const_closure_type;
	typedef vector_transport_to_host closure_type;
	typedef unknown_storage storage_type;
	typedef unknown_storage const_storage_type;
	typedef blockwise<typename E::evaluation_category::tag> evaluation_category;


	//FIXME: This is required even though iterators for block expressions are meaningless
	typedef typename E::const_iterator const_iterator;
	typedef const_iterator iterator;

	// Construction and destruction
	explicit vector_transport_to_host(
		expression_closure_type const& expression
	):m_expression(expression){}

	size_type size() const {
		return m_expression.size();
	}
	
	expression_closure_type const& expression() const {
		return m_expression;
	}
	boost::compute::command_queue& queue() const{
		return m_expression.queue();
	}
	
	no_functor elements() const{return no_functor();}
	
	//dispatcher to computation kernels
	template<class VecX>
	void assign_to(vector_expression<VecX, cpu_tag>& x)const{
		//in case the expression can not be mapped to memory, evaluate it
		//this does nothing for proxies
		auto e_eval = eval_expression(m_expression);
		
		auto storageE = e_eval.raw_storage();
		auto& buffer = storageE.buffer;
		
		//map buffer to host memory
		auto p = (typename E::value_type*) m_expression.queue().enqueue_map_buffer(
			buffer, CL_MAP_READ, 0, buffer.size()
		);
		//adapt host memory buffer to vector and assign
		auto adaptE = adapt_vector(size(), p + storageE.offset, storageE.stride);
		assign(x, adaptE);
		
		//unmap memory
		m_expression.queue().enqueue_unmap_buffer(buffer,p);
	}
	template<class VecX>
	void plus_assign_to(vector_expression<VecX, cpu_tag>& x)const{
		//in case the expression can not be mapped to memory, evaluate it
		//this does nothing for proxies
		auto e_eval = eval_expression(m_expression);
		
		auto storageE = e_eval.raw_storage();
		auto& buffer = storageE.buffer;
		//map buffer to host memory
		auto p = (value_type*) m_expression.queue().enqueue_map_buffer(
			buffer, CL_MAP_READ, 0, buffer.size()
		);
		//adapt host memory buffer to vector and assign
		auto adaptE = adapt_vector(size(), p + storageE.offset, storageE.stride);
		plus_assign(x,adaptE);
		
		//unmap memory
		m_expression.queue().enqueue_unmap_buffer(buffer,p);
	}
	
private:
	expression_closure_type m_expression;
};

template<class E>
class vector_transport_to_device: public vector_expression<vector_transport_to_device<E>, opencl_tag>{
public:
	typedef typename E::const_closure_type expression_closure_type;

	typedef typename E::value_type value_type;
	typedef typename E::size_type size_type;
	typedef value_type const& const_reference;
	typedef const_reference reference;

	typedef vector_transport_to_device const_closure_type;
	typedef vector_transport_to_device closure_type;
	typedef unknown_storage storage_type;
	typedef unknown_storage const_storage_type;
	typedef blockwise<typename E::evaluation_category::tag> evaluation_category;


	//FIXME: This is required even though iterators for block expressions are meaningless
	typedef typename E::const_iterator const_iterator;
	typedef const_iterator iterator;

	// Construction and destruction
	explicit vector_transport_to_device(
		expression_closure_type const& expression,
		boost::compute::command_queue& queue
	):m_expression(expression), m_queue(&queue){}

	size_type size() const {
		return m_expression.size();
	}
	expression_closure_type const& expression() const {
		return m_expression;
	}
	boost::compute::command_queue& queue() const{
		return *m_queue;
	}
	
	no_functor elements() const{return no_functor();}
	
	//dispatcher to computation kernels
	template<class VecX>
	void assign_to(vector_expression<VecX, opencl_tag>& x)const{
		auto storagex = x().raw_storage();
		auto& buffer = storagex.buffer;
		//map buffer to host memory
		auto p = (typename VecX::value_type*) x().queue().enqueue_map_buffer(
			buffer, CL_MAP_WRITE, 0, buffer.size()
		);
		//adapt host memory buffer to vector and assign
		auto adaptX = adapt_vector(size(), p + storagex.offset, storagex.stride);
		assign(adaptX,m_expression);
		
		//unmap memory
		x().queue().enqueue_unmap_buffer(buffer,p);
	}
	template<class VecX>
	void plus_assign_to(vector_expression<VecX, opencl_tag>& x)const{
		auto storagex = x().raw_storage();
		auto& buffer = storagex.buffer;
		//map buffer to host memory
		auto p = (typename VecX::value_type*) x().queue().enqueue_map_buffer(
			buffer, CL_MAP_WRITE, 0, buffer.size()
		);
		//adapt host memory buffer to vector and assign
		auto adaptX = adapt_vector(size(), p + storagex.offset, storagex.stride);
		plus_assign(adaptX,m_expression);
		
		//unmap memory
		x().queue().enqueue_unmap_buffer(buffer,p); 
	}
	
private:

	expression_closure_type m_expression;
	boost::compute::command_queue* m_queue;
};


///////////////////////////////////////
//////// Matrix Transport
///////////////////////////////////////	

template<class E>
class matrix_transport_to_cpu: public matrix_expression<matrix_transport_to_cpu<E>, cpu_tag>{
public:
	typedef typename E::const_closure_type expression_closure_type;

	typedef typename E::value_type value_type;
	typedef typename E::size_type size_type;
	typedef value_type const& const_reference;
	typedef const_reference reference;

	typedef matrix_transport_to_cpu const_closure_type;
	typedef matrix_transport_to_cpu closure_type;
	typedef unknown_storage storage_type;
	typedef unknown_storage const_storage_type;
	typedef blockwise<typename E::evaluation_category::tag> evaluation_category;
	typedef typename E::orientation orientation;

	typedef no_iterator const_major_iterator;
	typedef no_iterator major_iterator;

	// Construction and destruction
	explicit matrix_transport_to_cpu(
		expression_closure_type const& expression
	):m_expression(expression){}

	size_type size1() const {
		return m_expression.size1();
	}
	size_type size2() const {
		return m_expression.size2();
	}
	expression_closure_type const& expression() const {
		return m_expression;
	}
	boost::compute::command_queue& queue() const{
		return m_expression.queue();
	}
	
	no_functor elements() const{return no_functor();}
	
	//dispatcher to computation kernels
	template<class MatX>
	void assign_to(matrix_expression<MatX, cpu_tag>& X) const{
		//in case the expression can not be mapped to memory, evaluate it
		//this does nothing for proxies
		auto e_eval = eval_expression(m_expression);
		
		auto storageE = e_eval().raw_storage();
		auto& buffer = storageE.buffer;
		//map buffer to host memory
		auto p = (typename E::value_type*) m_expression.queue().enqueue_map_buffer(
			buffer, CL_MAP_READ, 0, buffer.size()
		);
		//adapt host memory buffer to matrix and assign
		typedef typename decltype(e_eval)::orientation EOrientation;
		typedef dense_matrix_adaptor<typename E::value_type, EOrientation> AdaptE;
		AdaptE adaptE(p + storageE.offset,size1(), size2(), storageE.leading_dimension);
		
		assign(X, adaptE);
		
		//unmap memory
		m_expression.queue().enqueue_unmap_buffer(buffer,p);
	}
	template<class MatX>
	void plus_assign_to(matrix_expression<MatX, cpu_tag>& X)const{
		//in case the expression can not be mapped to memory, evaluate it
		//this does nothing for proxies
		auto e_eval = eval_expression(m_expression);
		
		auto storageE = e_eval().raw_storage();
		auto& buffer = storageE.buffer;
		//map buffer to host memory
		auto p = (typename E::value_type*) m_expression.queue().enqueue_map_buffer(
			buffer, CL_MAP_READ, 0, buffer.size()
		);
		//adapt host memory buffer to matrix and assign
		typedef typename decltype(e_eval)::orientation EOrientation;
		typedef dense_matrix_adaptor<typename E::value_type, EOrientation> AdaptE;
		AdaptE adaptE(p + storageE.offset, size1(), size2(), storageE.leading_dimension);
		
		plus_assign(X, adaptE);
		
		//unmap memory
		m_expression.queue().enqueue_unmap_buffer(buffer,p);
	}
private:
	expression_closure_type m_expression;
};

template<class E>
class matrix_transport_to_opencl: public matrix_expression<matrix_transport_to_opencl<E>, opencl_tag>{
public:
	typedef typename E::const_closure_type expression_closure_type;

	typedef typename E::value_type value_type;
	typedef typename E::size_type size_type;
	typedef value_type const& const_reference;
	typedef const_reference reference;

	typedef matrix_transport_to_opencl const_closure_type;
	typedef matrix_transport_to_opencl closure_type;
	typedef unknown_storage storage_type;
	typedef unknown_storage const_storage_type;
	typedef blockwise<typename E::evaluation_category::tag> evaluation_category;
	typedef typename E::orientation orientation;

	typedef no_iterator const_major_iterator;
	typedef no_iterator major_iterator;

	// Construction and destruction
	explicit matrix_transport_to_opencl(
		expression_closure_type const& expression,
		boost::compute::command_queue& queue
	):m_expression(expression), m_queue(&queue){}

	size_type size1() const {
		return m_expression.size1();
	}
	size_type size2() const {
		return m_expression.size2();
	}
	expression_closure_type const& expression() const {
		return m_expression;
	}
	boost::compute::command_queue& queue() const{
		return *m_queue;
	}
	
	no_functor elements() const{return no_functor();}
	
	//dispatcher to computation kernels
	template<class MatX>
	void assign_to(matrix_expression<MatX, opencl_tag>& X)const{
		auto storageX = X().raw_storage();
		auto& buffer = storageX.buffer;
		//map buffer to host memory
		typename MatX::value_type* p = (typename MatX::value_type*) X().queue().enqueue_map_buffer(
			buffer, CL_MAP_WRITE, 0, buffer.size()
		);
		//adapt host memory buffer to vector and assign
		typedef typename MatX::orientation XOrientation;
		dense_matrix_adaptor<typename MatX::value_type, XOrientation> adaptX(p, size1(), size2(), storageX.leading_dimension);
		assign(adaptX, m_expression);
		
		//unmap memory
		X().queue().enqueue_unmap_buffer(buffer,p);
	}
	template<class MatX>
	void plus_assign_to(matrix_expression<MatX, opencl_tag>& X) const{
		auto storageX = X().raw_storage();
		auto& buffer = storageX.buffer;
		//map buffer to host memory
		typename MatX::value_type* p = (typename MatX::value_type*) X().queue().enqueue_map_buffer(
			buffer, CL_MAP_WRITE, 0, buffer.size()
		);
		//adapt host memory buffer to matrix and assign
		typedef typename MatX::orientation XOrientation;
		typedef dense_matrix_adaptor<typename MatX::value_type, XOrientation> AdaptX;
		AdaptX adaptX(p + storageX.offset, size1(), size2(), storageX.leading_dimension);
		
		plus_assign(adaptX, m_expression);
		
		//unmap memory
		X().queue().enqueue_unmap_buffer(buffer,p);
	}

private:
	expression_closure_type m_expression;
	boost::compute::command_queue* m_queue;
};

}

///////////////////////////////////////////////
//////// Expression Optimizers
///////////////////////////////////////////////

namespace detail{
template<class E>
struct matrix_scalar_multiply_optimizer<opencl::vector_transport_to_device<E> >{
	typedef vector_scalar_multiply_optimizer<E> opt;
	typedef opencl::vector_transport_to_device<typename opt::type> type;
	static type create(opencl::vector_transport_to_device<E> const& v, typename type::value_type alpha){
		return type(opt::create(v.expression(), alpha), v.queue());
	}
};
template<class E>
struct matrix_scalar_multiply_optimizer<opencl::vector_transport_to_host<E> >{
	typedef vector_scalar_multiply_optimizer<E> opt;
	typedef opencl::vector_transport_to_host<typename opt::type> type;
	static type create(opencl::vector_transport_to_host<E> const& v, typename type::value_type alpha){
		return type(opt::create(v.expression(), alpha));
	}
};

template<class E>
struct matrix_scalar_multiply_optimizer<opencl::matrix_transport_to_opencl<E> >{
	typedef matrix_scalar_multiply_optimizer<E> opt;
	typedef opencl::matrix_transport_to_opencl<typename opt::type> type;
	static type create(opencl::matrix_transport_to_opencl<E> const& m, typename type::value_type alpha){
		return type(opt::create(m.expression(), alpha), m.queue());
	}
};

template<class E>
struct matrix_scalar_multiply_optimizer<opencl::matrix_transport_to_cpu<E> >{
	typedef matrix_scalar_multiply_optimizer<E> opt;
	typedef opencl::matrix_transport_to_cpu<typename opt::type> type;
	static type create(opencl::matrix_transport_to_cpu<E> const& m, typename type::value_type alpha){
		return type(opt::create(m.expression(), alpha));
	}
};
}

//TODO: proxy(copy_to_opencl) should be possible...

///////////////////////////////////////////////
//////// Expressions
///////////////////////////////////////////////

template<class E>
opencl::vector_transport_to_host<E> copy_to_cpu(vector_expression<E, opencl_tag> const& e){
	return opencl::vector_transport_to_host<E>(e());
}

template<class E>
opencl::matrix_transport_to_cpu<E> copy_to_cpu(matrix_expression<E, opencl_tag> const& e){
	return opencl::matrix_transport_to_cpu<E>(e());
}
template<class E>
opencl::vector_transport_to_device<E> copy_to_opencl(
	vector_expression<E, cpu_tag> const& e,
	boost::compute::command_queue& queue = boost::compute::system::default_queue()
){
	return opencl::vector_transport_to_device<E>(e(), queue);
}

template<class E>
opencl::matrix_transport_to_opencl<E> copy_to_opencl(
	matrix_expression<E, cpu_tag> const& e,
	boost::compute::command_queue& queue = boost::compute::system::default_queue()
){
	return opencl::matrix_transport_to_opencl<E>(e(),queue);
}

//moving opencl->opencl is for free
template<class E>
E const& copy_to_opencl(
	vector_expression<E, opencl_tag> const& e,
	boost::compute::command_queue& queue = boost::compute::system::default_queue()
){
	return e();
}

template<class E>
E const& copy_to_opencl(
	matrix_expression<E, opencl_tag> const& e,
	boost::compute::command_queue& queue = boost::compute::system::default_queue()
){
	return e();
}

template<class E, class Device>
auto copy_to_device(vector_expression<E, Device> const& e, opencl_tag)->decltype(copy_to_opencl(e)){
	return copy_to_opencl(e);
}


template<class E, class Device>
auto copy_to_device(matrix_expression<E, Device> const& e, opencl_tag)->decltype(copy_to_opencl(e)){
	return copy_to_opencl(e);
}
	

}

#endif
