//===========================================================================
/*!
 * 
 *
 * \brief       Traits of gpu expressions
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
//===========================================================================

#ifndef REMORA_GPU_TRAITS_HPP
#define REMORA_GPU_TRAITS_HPP

#include "iterators.hpp"
#include <boost/compute/command_queue.hpp>
#include <boost/compute/core.hpp>
#include <boost/compute/container/vector.hpp>
#include <boost/compute/functional/operator.hpp>
#include <boost/compute/functional.hpp>

namespace remora{namespace gpu{
	
template<class T, class Tag>
struct dense_vector_storage{
	typedef Tag storage_tag;
	
	boost::compute::buffer buffer;
	std::size_t offset;
	std::size_t stride;
	
	dense_vector_storage(){}
	dense_vector_storage(boost::compute::buffer const& buffer, std::size_t offset, std::size_t stride)
	:buffer(buffer), offset(offset), stride(stride){}
	template<class U, class Tag2>
	dense_vector_storage(dense_vector_storage<U, Tag2> const& storage):
	buffer(storage.buffer), offset(storage.offset), stride(storage.stride){
		static_assert(!(std::is_same<Tag,continuous_dense_tag>::value && std::is_same<Tag2,dense_tag>::value), "Trying to assign dense to continuous dense storage");
	}
	
	dense_vector_storage<T,Tag> sub_region(std::size_t offset) const{
		return {buffer, this->offset+offset * stride, stride};
	}
};

template<class T, class Tag>
struct dense_matrix_storage{
	typedef Tag storage_tag;
	template<class O>
	struct row_storage: public std::conditional<
		std::is_same<O,row_major>::value,
		dense_vector_storage<T, Tag>,
		dense_vector_storage<T, dense_tag>
	>{};
	template<class O>
	struct rows_storage: public std::conditional<
		std::is_same<O,row_major>::value,
		dense_matrix_storage<T, Tag>,
		dense_matrix_storage<T, dense_tag>
	>{};
	
	typedef dense_vector_storage<T,Tag> diag_storage;
	typedef dense_matrix_storage<T,dense_tag> sub_region_storage;
	
	boost::compute::buffer buffer;
	std::size_t offset;
	std::size_t leading_dimension;
	
	dense_matrix_storage(){}
	dense_matrix_storage(boost::compute::buffer const& buffer, std::size_t offset, std::size_t leading_dimension)
	:buffer(buffer), offset(offset), leading_dimension(leading_dimension){
	}
	template<class U, class Tag2>
	dense_matrix_storage(dense_matrix_storage<U, Tag2> const& storage):
	buffer(storage.buffer), offset(storage.offset), leading_dimension(storage.leading_dimension){
		static_assert(!(std::is_same<Tag,continuous_dense_tag>::value && std::is_same<Tag2,dense_tag>::value), "Trying to assign dense to continuous dense storage");
	}
	
	template<class Orientation>
	sub_region_storage sub_region(std::size_t offset1, std::size_t offset2, Orientation) const{
		std::size_t offset_major = Orientation::index_M(offset1,offset2);
		std::size_t offset_minor = Orientation::index_m(offset1,offset2);
		return {buffer, offset + offset_major*leading_dimension+offset_minor, leading_dimension};
	}
	
	template<class Orientation>
	typename row_storage<Orientation>::type row(std::size_t i, Orientation) const{
		return {buffer, offset + i * Orientation::index_M(leading_dimension,std::size_t(1)), Orientation::index_m(leading_dimension,std::size_t(1))};
	}
	
	template<class Orientation>
	typename rows_storage<Orientation>::type sub_rows(std::size_t i, Orientation) const{
		std::size_t stride = Orientation::index_M(leading_dimension,(std::size_t)1);
		return {buffer,offset + i * stride, leading_dimension};
	}
	
	diag_storage diag(){
		return {buffer, offset, leading_dimension+1};
	}
	
	dense_vector_storage<T, continuous_dense_tag> linear() const{
		return {buffer, offset, 1};
	}
};


//Expression objects and generated by the functors which are then turned into code by meta_kernel operator<<
//Note that often the type of the stored scalar can be different from its actual type, this is to allow replacing the value
//by a variable representing it (i.e. a kernel argument). This way we prevent hard coding variables in the generated source code
//in case the variable is indeed not constant
namespace detail{
template<class T, class Stored = T>
struct invoked_constant{
	Stored m_value;
};
	
template<class Arg1, class T, char Op, class Stored>
struct invoked_operator_scalar{
	typedef T result_type;
	Arg1 arg1;
	Stored m_scalar;
};

template<class Arg1, class T, class Stored = T>
struct invoked_add_scalar{
	typedef T result_type;
	Arg1 arg1;
	Stored m_scalar;
};

template<class Arg1, class Arg2, class T, class Stored=T>
struct invoked_multiply_and_add{
	typedef T result_type;
	Arg1 arg1;
	Arg2 arg2;
	Stored m_scalar;
};

template<class Arg1, class T>
struct invoked_soft_plus{
	typedef T result_type;
	Arg1 arg1;
};
template<class Arg1, class T>
struct invoked_sigmoid{
	typedef T result_type;
	Arg1 arg1;
};

template<class Arg1, class T>
struct invoked_sqr{
	typedef T result_type;
	Arg1 arg1;
};

template<class Arg1, class T>
struct invoked_inv{
	typedef T result_type;
	Arg1 arg1;
};

template<class Arg1, class Arg2, class T, class S>
struct invoked_safe_div{
	typedef T result_type;
	Arg1 arg1;
	Arg2 arg2;
	S default_value;
};


template<class Arg1, class T, char Op, class S>
boost::compute::detail::meta_kernel& operator<<(boost::compute::detail::meta_kernel& k, invoked_operator_scalar<Arg1,T, Op, S> const& e){
	return k << '('<<e.arg1 << Op << e.m_scalar<<')';
}
template<class Arg1, class Arg2, class T, class S>
boost::compute::detail::meta_kernel& operator<<(boost::compute::detail::meta_kernel& k, invoked_multiply_and_add<Arg1,Arg2,T, S> const& e){
	return k << '('<<e.arg1<<'+'<<e.m_scalar << '*'<< e.arg2<<')';
}
template<class Arg1, class T>
boost::compute::detail::meta_kernel& operator<<(boost::compute::detail::meta_kernel& k, invoked_soft_plus<Arg1,T> const& e){
	return k << "(log(1+exp("<< e.arg1<<")))";
}
template<class Arg1, class T>
boost::compute::detail::meta_kernel& operator<<(boost::compute::detail::meta_kernel& k, invoked_sigmoid<Arg1,T> const& e){
	return k << "(1/(1+exp(-"<< e.arg1<<")))";
}
template<class Arg1, class T>
boost::compute::detail::meta_kernel& operator<<(boost::compute::detail::meta_kernel& k, invoked_sqr<Arg1,T> const& e){
	return k << '('<<e.arg1<<'*'<<e.arg1<<')';
}
template<class Arg1, class T>
boost::compute::detail::meta_kernel& operator<<(boost::compute::detail::meta_kernel& k, invoked_inv<Arg1,T> const& e){
	return k << "1/("<<e.arg1<<')';
}

template<class T, class S>
boost::compute::detail::meta_kernel& operator<<(boost::compute::detail::meta_kernel& k, invoked_constant<T, S> const& e){
	return k << e.m_value;
}


template<class Arg1, class Arg2, class T, class S>
boost::compute::detail::meta_kernel& operator<<(boost::compute::detail::meta_kernel& k, invoked_safe_div<Arg1,Arg2,T, S> const& e){
	return k << "(("<<e.arg2<<"!=0)?"<<e.arg1<<'/'<<e.arg2<<':'<<e.default_value<<')';
}

}//End namespace detail
}//End namespace gpu

template<>
struct device_traits<gpu_tag>{
	typedef boost::compute::command_queue queue_type;
	
	static queue_type& default_queue(){
		return boost::compute::system::default_queue();
	}
	
	// iterators
	
	template <class Iterator, class Functor>
	struct transform_iterator{
		typedef boost::compute::transform_iterator<Iterator, Functor> type;
	};
	
	template <class Iterator1, class Iterator2, class Functor>
	struct binary_transform_iterator{
		typedef gpu::detail::binary_transform_iterator<Iterator1,Iterator2, Functor> type;
	};
	
	template<class T>
	struct constant_iterator{
		typedef boost::compute::constant_iterator<T> type;
	};
	
	template<class T>
	struct one_hot_iterator{
		typedef iterators::one_hot_iterator<T> type;
	};
	
	template<class Closure>
	struct indexed_iterator{
		typedef gpu::detail::indexed_iterator<Closure> type;
	};
	
	//functional
	
	//G(F(args))
	template<class F, class G>
	struct compose{
		typedef typename G::result_type result_type;
		compose(F const& f, G const& g): m_f(f), m_g(g){ }
		
		template<class Arg1>
		auto operator()( Arg1 const& x) const -> decltype(std::declval<G const&>()(std::declval<F const&>()(x))){
			return m_g(m_f(x));
		}
		template<class Arg1, class Arg2>
		auto operator()( Arg1 const& x, Arg2 const& y) const -> decltype(std::declval<G const&>()(std::declval<F const&>()(x,y))){
			return m_g(m_f(x,y));
		}
		
		F m_f;
		G m_g;
	};
	
	//G(F1(args),F2(args))
	template<class F1, class F2, class G>
	struct compose_binary{
		typedef typename G::result_type result_type;
		compose_binary(F1 const& f1, F2 const& f2, G const& g): m_f1(f1), m_f2(f2), m_g(g){ }
		
		template<class Arg1>
		auto operator()( Arg1 const& x) const -> decltype(std::declval<G const&>()(std::declval<F1 const&>()(x),std::declval<F2 const&>()(x))){
			return m_g(m_f1(x), m_f2(x));
		}
		template<class Arg1, class Arg2>
		auto operator()( Arg1 const& x, Arg2 const& y) const -> decltype(std::declval<G const&>()(std::declval<F1 const&>()(x,y),std::declval<F2 const&>()(x,y))){
			return m_g(m_f1(x,y), m_f2(x,y));
		}
		
		F1 m_f1;
		F2 m_f2;
		G m_g;
	};
	
	
	//G(F1(arg1),F2(arg2))
	template<class F1, class F2, class G>
	struct transform_arguments{
		typedef typename G::result_type result_type;
		transform_arguments(F1 const& f1, F2 const& f2, G const& g): m_f1(f1), m_f2(f2), m_g(g){ }
		
		template<class Arg1, class Arg2>
		auto operator()( Arg1 const& x, Arg2 const& y) const -> decltype(std::declval<G const&>()(std::declval<F1 const&>()(x),std::declval<F2 const&>()(y))){
			return m_g(m_f1(x),m_f2(y));
		}
		
		F1 m_f1;
		F2 m_f2;
		G m_g;
	};
	
	template<class F, class Arg2>
	struct bind_second{
		typedef typename F::result_type result_type;
		bind_second(F const& f, Arg2 const& arg2) : m_function(f), m_arg2(arg2){ }
		
		template<class Arg1>
		auto operator()(Arg1 const& arg1) const -> decltype(std::declval<F const&>()(arg1,std::declval<Arg2 const&>()))
		{
			return m_function(arg1, m_arg2);
		}
		
		F m_function;
		Arg2 m_arg2;
	};
	
	
	//helper functions
	template<class F, class G>
	static compose<F,G> make_compose(F const& f, G const&g){
		return compose<F,G>(f,g);
	}
	
	template<class F1, class F2, class G>
	static compose_binary<F1, F2, G> make_compose_binary(F1 const& f1, F2 const& f2, G const&g){
		return compose_binary<F1, F2, G>(f1, f2, g);
	}
	
	template<class F1, class F2, class G>
	static transform_arguments<F1, F2, G> make_transform_arguments(F1 const& f1, F2 const& f2, G const& g){
		return transform_arguments<F1, F2, G>(f1, f2, g);
	}
	
	template<class F, class Arg2>
	static bind_second<F,Arg2> make_bind_second(F const& f, Arg2 const& arg2){
		return bind_second<F,Arg2>(f,arg2);
	}
	
	
	//functors
	
	//basic arithmetic
	template<class T>
	using add = boost::compute::plus<T>;
	template<class T>
	using subtract = boost::compute::minus<T>;
	template<class T>
	using multiply = boost::compute::multiplies<T>;
	template<class T>
	using divide = boost::compute::divides<T>;
	template<class T>
	using modulo = boost::compute::modulus<T>;
	template<class T>
	using pow = boost::compute::pow<T>;
	template<class T, class S=T>
	struct safe_divide{
		typedef T result_type;
		safe_divide(S const& default_value) : default_value(default_value) { }
		
		template<class Arg1, class Arg2>
		gpu::detail::invoked_safe_div<Arg1,Arg2, T,S> operator()(const Arg1 &x, const Arg2& y) const
		{
			return {x,y,default_value};
		}
		S default_value;
	};
	template<class T, class S= T>
	struct multiply_and_add{
		typedef T result_type;
		multiply_and_add(S const& scalar) :m_scalar(scalar) { }
		
		template<class Arg1, class Arg2>
		gpu::detail::invoked_multiply_and_add<Arg1,Arg2,T,S> operator()(const Arg1 &x, const Arg2& y) const
		{
			return {x,y, m_scalar};
		}
		S m_scalar;
	};
	
	
	template<class T, char Op, class S>
	struct operator_scalar{
		typedef T result_type;
		operator_scalar(S const& scalar) : m_scalar(scalar) { }
		
		template<class Arg1>
		gpu::detail::invoked_operator_scalar<Arg1,T, Op, S> operator()(Arg1 const& x) const
		{
			return {x, m_scalar};
		}
		S m_scalar;
	};
	
	template<class T>
	using multiply_scalar = operator_scalar<T, '*', T>;
	template<class T>
	using add_scalar = operator_scalar<T, '+', T>;
	template<class T>
	using divide_scalar = operator_scalar<T, '/', T>;
	template<class T>
	using modulo_scalar = operator_scalar<T, '%', T>;
	
	template<class T, class S=T>
	struct multiply_assign{
		typedef T result_type;
		multiply_assign(S const& scalar): m_scalar(scalar) { }
		
		template<class Arg1, class Arg2>
		gpu::detail::invoked_operator_scalar<Arg2,T,'*',S> operator()(const Arg1&, const Arg2& y) const
		{
			return {y, m_scalar};
		}
		S m_scalar;
	};
	template<class T>
	struct identity{
		typedef T result_type;
		
		template<class Arg>
		Arg const& operator()(Arg const& arg) const{
			return arg;
		}
	};
	
	template<class T>
	struct left_arg{
		typedef T result_type;
		
		template<class Arg1, class Arg2>
		Arg1 const& operator()(Arg1 const& arg1, Arg2 const&) const{
			return arg1;
		}
	};
	template<class T>
	struct right_arg{
		typedef T result_type;
		
		template<class Arg1, class Arg2>
		Arg2 const& operator()(Arg1 const&, Arg2 const& arg2) const{
			return arg2;
		}
	};
	
	template<class T, class S=T>
	struct constant{
		typedef T result_type;
		constant(S const& value): m_value(value){}
		
		template<class Arg>
		gpu::detail::invoked_constant<T,S> operator()(Arg const&) const
		{
			return {m_value};
		}
		template<class Arg1, class Arg2>
		gpu::detail::invoked_constant<T,S> operator()(Arg1 const&, Arg2 const&) const
		{
			return {m_value};
		}
		
		S m_value;
	};
	
	
	//math unary functions
	template<class T>
	using log = boost::compute::log<T>;
	template<class T>
	using exp = boost::compute::exp<T>;
	template<class T>
	using sin = boost::compute::sin<T>;
	template<class T>
	using cos = boost::compute::cos<T>;
	template<class T>
	using tan = boost::compute::tan<T>;
	template<class T>
	using asin = boost::compute::asin<T>;
	template<class T>
	using acos = boost::compute::acos<T>;
	template<class T>
	using atan = boost::compute::atan<T>;
	template<class T>
	using tanh = boost::compute::tanh<T>;
	template<class T>
	using sqrt = boost::compute::sqrt<T>;
	template<class T>
	using cbrt = boost::compute::cbrt<T>;
	template<class T>
	using abs = boost::compute::fabs<T>;
	
	template<class T>
	using erf = boost::compute::erf<T>;
	template<class T>
	using erfc = boost::compute::erfc<T>;
	
	template<class T>
	struct sqr{
		typedef T result_type;
		
		template<class Arg1>
		gpu::detail::invoked_sqr<Arg1,T> operator()(const Arg1 &x) const{
			return {x};
		}
	};
	template<class T>
	struct soft_plus{
		typedef T result_type;
		
		template<class Arg1>
		gpu::detail::invoked_soft_plus<Arg1,T> operator()(const Arg1 &x) const{
			return {x};
		}
	};
	template<class T>
	struct sigmoid{
		typedef T result_type;
		
		template<class Arg1>
		gpu::detail::invoked_sigmoid<Arg1,T> operator()(const Arg1 &x) const{
			return {x};
		}
	};
	template<class T>
	struct inv{
		typedef T result_type;
		
		template<class Arg1>
		gpu::detail::invoked_inv<Arg1,T> operator()(const Arg1 &x) const{
			return {x};
		}
	};
	
	//min/max
	template<class T>
	using min = boost::compute::fmin<T>;
	template<class T>
	using max = boost::compute::fmax<T>;
	
	//comparison
	template<class T>
	using less = boost::compute::less<T>;
	template<class T>
	using less_equal  = boost::compute::less_equal<T>;
	template<class T>
	using greater = boost::compute::greater<T>;
	template<class T>
	using greater_equal  = boost::compute::greater_equal<T>;
	template<class T>
	using equal = boost::compute::equal_to<T>;
	template<class T>
	using not_equal  = boost::compute::not_equal_to<T>;
};

namespace gpu{namespace detail{
	
    
struct meta_kernel;

template<class Entity>
struct register_with_compute_kernel{
	typedef Entity type;
	static type const& reg(meta_kernel&, Entity const& e){
		return e;
	}
};

struct meta_kernel: public boost::compute::detail::meta_kernel{
	meta_kernel(std::string const& name):boost::compute::detail::meta_kernel(name), m_id(0){}
	
    template<class T>
    std::string register_kernel_arg(T const& value){
        ++m_id;
        std::string name = "rem_var"+std::to_string(m_id);
        this->add_set_arg<T>(name,value);
        return name;
    }
    
	template<class Entity>
	typename register_with_compute_kernel<Entity>::type
	register_args(Entity const& e){
		return register_with_compute_kernel<Entity>::reg(*this,e);
	}
private:
	std::size_t m_id;
};

template<class F, class Arg2>
struct register_with_compute_kernel<device_traits<gpu_tag>::template bind_second<F,Arg2> >{
	typedef typename register_with_compute_kernel<F>::type f_type;
	typedef device_traits<gpu_tag>::template bind_second<f_type,std::string> type;
	static type reg(
		meta_kernel& k,
		device_traits<gpu_tag>::template bind_second<F,Arg2> const& f
	){
		std::string arg2_name = k.register_kernel_arg(f.m_arg2);
		return type(register_with_compute_kernel<F>::reg(k,f.m_function),arg2_name);
	}
};

template<class F, class G>
struct register_with_compute_kernel<device_traits<gpu_tag>::template compose<F, G> >{
	typedef typename register_with_compute_kernel<F>::type f_type;
	typedef typename register_with_compute_kernel<G>::type g_type;
	typedef typename device_traits<gpu_tag>::template compose<f_type, g_type> type;
	static type reg(
		meta_kernel& k,
		device_traits<gpu_tag>::compose<F, G> const& composed
	){
		auto f_reg = register_with_compute_kernel<F>::reg(k,composed.m_f);
		auto g_reg = register_with_compute_kernel<G>::reg(k,composed.m_g);
		return type(f_reg, g_reg);
	}
};

template<class F1, class F2, class G>
struct register_with_compute_kernel<device_traits<gpu_tag>::template compose_binary<F1, F2, G> >{
	typedef typename register_with_compute_kernel<F1>::type f1_type;
	typedef typename register_with_compute_kernel<F2>::type f2_type;
	typedef typename register_with_compute_kernel<G>::type g_type;
	typedef typename device_traits<gpu_tag>::template compose_binary<f1_type, f2_type, g_type> type;
	static type reg(
		meta_kernel& k,
		device_traits<gpu_tag>::compose_binary<F1, F2, G> const& composed
	){
		auto f1_reg = register_with_compute_kernel<F1>::reg(k,composed.m_f1);
		auto f2_reg = register_with_compute_kernel<F2>::reg(k,composed.m_f2);
		auto g_reg = register_with_compute_kernel<G>::reg(k,composed.m_g);
		return type(f1_reg, f2_reg, g_reg);
	}
};


template<class F1, class F2, class G>
struct register_with_compute_kernel<device_traits<gpu_tag>::template transform_arguments<F1, F2, G> >{
	typedef typename register_with_compute_kernel<F1>::type f1_type;
	typedef typename register_with_compute_kernel<F2>::type f2_type;
	typedef typename register_with_compute_kernel<G>::type g_type;
	typedef typename device_traits<gpu_tag>::template transform_arguments<f1_type, f2_type, g_type> type;
	static type reg(
		meta_kernel& k,
		device_traits<gpu_tag>::transform_arguments<F1, F2, G> const& composed
	){
		auto f1_reg = register_with_compute_kernel<F1>::reg(k,composed.m_f1);
		auto f2_reg = register_with_compute_kernel<F2>::reg(k,composed.m_f2);
		auto g_reg = register_with_compute_kernel<G>::reg(k,composed.m_g);
		return type(f1_reg, f2_reg, g_reg);
	}
};

template<class T>
struct register_with_compute_kernel<device_traits<gpu_tag>::template constant<T,T> >{
	typedef typename device_traits<gpu_tag>::template constant<T,std::string> type;
	static type reg(
		meta_kernel& k,
		device_traits<gpu_tag>::constant<T,T> const& f
	){
		return type(k.register_kernel_arg(f.m_value));
	}
};

template<class T>
struct register_with_compute_kernel<device_traits<gpu_tag>::template safe_divide<T,T> >{
	typedef typename device_traits<gpu_tag>::template safe_divide<T,std::string> type;
	static type reg(
		meta_kernel& k,
		device_traits<gpu_tag>::safe_divide<T,T> const& f
	){
		return type(k.register_kernel_arg(f.default_value));
	}
};

template<class T>
struct register_with_compute_kernel<device_traits<gpu_tag>::template multiply_and_add<T,T> >{
	typedef typename device_traits<gpu_tag>::template multiply_and_add<T,std::string> type;
	static type reg(
		meta_kernel& k,
		device_traits<gpu_tag>::multiply_and_add<T,T> const& f
	){
		return type(k.register_kernel_arg(f.m_scalar));
	}
};

template<class T, char Op>
struct register_with_compute_kernel<device_traits<gpu_tag>::template operator_scalar<T, Op, T> >{
	typedef typename device_traits<gpu_tag>::template operator_scalar<T, Op, std::string> type;
	static type reg(
		meta_kernel& k,
		device_traits<gpu_tag>::operator_scalar<T, Op, T> const& f
	){
		return type(k.register_kernel_arg(f.m_scalar));
	}
};

template<class T>
struct register_with_compute_kernel<device_traits<gpu_tag>::template multiply_assign<T,T> >{
	typedef typename device_traits<gpu_tag>::template multiply_assign<T,std::string> type;
	static type reg(
		meta_kernel& k,
		device_traits<gpu_tag>::multiply_assign<T,T> const& f
	){
		return type(k.register_kernel_arg(f.m_scalar));
	}
};


//vector element
template<class Arg, class T, class S>
struct invoked_dense_vector_element{
	typedef T result_type;
	Arg arg;
	S stride;
	S offset;
	boost::compute::buffer buffer;
};

template<class Arg,class T, class S>
boost::compute::detail::meta_kernel& operator<< (
	boost::compute::detail::meta_kernel& k, 
	invoked_dense_vector_element<Arg, T, S> const& e
){
	return k<< k.get_buffer_identifier<T>(e.buffer, boost::compute::memory_object::global_memory)
		<<" [ "<<e.offset <<"+("<<e.arg <<") *"<<e.stride<<']';
}

template<class T, class S=std::size_t>
struct dense_vector_element{
	typedef T result_type;
	
	template<class Arg>
	gpu::detail::invoked_dense_vector_element<Arg,T, S> operator()(Arg const& x) const{
		return {x, m_stride, m_offset, m_buffer};
	}
	boost::compute::buffer m_buffer;
	S m_stride;
	S m_offset;
};

template<class T>
struct register_with_compute_kernel<dense_vector_element<T,std::size_t> >{
	typedef dense_vector_element<T,std::string> type;
	static type reg(
		meta_kernel& k,
		dense_vector_element<T,std::size_t> const& e
	){
		return {e.m_buffer, k.register_kernel_arg(e.m_stride),k.register_kernel_arg(e.m_offset)};
	}
};

//matrix element
template<class Arg1, class Arg2,  class T, class S>
struct invoked_matrix_element{
	typedef T result_type;
	Arg1 arg1;
	Arg2 arg2;
	S stride1;
	S stride2;
	S offset;
	boost::compute::buffer buffer;
};


template<class Arg1, class Arg2, class T, class S>
boost::compute::detail::meta_kernel& operator<< (
	boost::compute::detail::meta_kernel& k, 
	invoked_matrix_element<Arg1, Arg2, T, S> const& e
){
	return k << k.get_buffer_identifier<T>(e.buffer, boost::compute::memory_object::global_memory)
				 <<'['<<e.offset<<"+ ("<<e.arg1 <<") * "<<e.stride1<<" + ("<<e.arg2 <<") * "<<e.stride2<<']';
}

template<class T, class S=std::size_t>
struct dense_matrix_element{
	typedef T result_type;

	template<class Arg1, class Arg2>
	gpu::detail::invoked_matrix_element<Arg1, Arg2, T, S> operator()(Arg1 const& x, Arg2 const& y) const{
		return {x, y, m_stride1, m_stride2, m_offset, m_buffer};
	}
	
	boost::compute::buffer m_buffer;
	S m_stride1;
	S m_stride2;
	S m_offset;
};

template<class T>
struct register_with_compute_kernel<dense_matrix_element<T,std::size_t> >{
	typedef dense_matrix_element<T,std::string> type;
	static type reg(
		meta_kernel& k,
		dense_matrix_element<T,std::size_t> const& e
	){
		auto const& stride1 = k.register_kernel_arg(e.m_stride1); 
		auto const& stride2 = k.register_kernel_arg(e.m_stride2); 
		auto const& offset = k.register_kernel_arg(e.m_offset); 
		return {e.m_buffer, stride1, stride2, offset};
	}
};

}}

}

#endif