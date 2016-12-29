/*!
 * \brief       expression templates for vector valued math
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
#ifndef SHARK_LINALG_BLAS_VECTOR_EXPRESSION_HPP
#define SHARK_LINALG_BLAS_VECTOR_EXPRESSION_HPP

#include "detail/expression_optimizers.hpp"
#include "kernels/dot.hpp"
#include <boost/utility/enable_if.hpp>

namespace shark {
namespace blas {
	
///////////////////VECTOR REDUCTION FUNCTORS/////////////////////////

//Functor implementing reduction of the form f(v_n,f(v_{n-1},f(....f(v_0,seed))))
// we assume for sparse vectors that the following holds:
// f(0,0) = 0 and f(v,f(0,w))=f(f(v,w),0)
//second argument to the function is the default value(seed).
template<class F>
struct vector_fold{
	
	vector_fold(F const& f):m_functor(f){}
	vector_fold(){}
	
	template<class E, class T>
	T operator()(
		vector_expression<E, cpu_tag> const& v,
		T seed
	) {
		return apply(v(),seed, typename E::evaluation_category::tag());
	}
private:
	//Dense Case
	template<class E, class T>
	T apply(
		E const& v,
		T seed,
		dense_tag
	) {
		std::size_t size = v.size();
		T result = seed;
		for(std::size_t i = 0; i != size; ++i){
			result = m_functor(result,v(i));
		}
		return result;
	}
	//Sparse Case
	template<class E, class T>
	T apply(
		E const& v,
		T seed,
		sparse_tag
	) {
		typename E::const_iterator iter=v.begin();
		typename E::const_iterator end=v.end();
		
		T result = seed;
		std::size_t nnz = 0;
		for(;iter != end;++iter,++nnz){
			result = m_functor(result,*iter);
		}
		//apply final operator f(0,v)
		if(nnz != v.size())
			result = m_functor(result,*iter);
		return result;
	}
	F m_functor;
};

template<class T, class VecV, class Device>
typename boost::enable_if<
	std::is_convertible<T, typename VecV::value_type >,
	vector_scalar_multiply<VecV>
>::type
operator* (vector_expression<VecV, Device> const& v, T scalar){
	typedef typename VecV::value_type value_type;
	return vector_scalar_multiply<VecV>(v(), value_type(scalar));
}
template<class T, class VecV, class Device>
typename boost::enable_if<
	std::is_convertible<T, typename VecV::value_type >,
        vector_scalar_multiply<VecV>
>::type
operator* (T scalar, vector_expression<VecV, Device> const& v){
	typedef typename VecV::value_type value_type;
	return vector_scalar_multiply<VecV>(v(), value_type(scalar));//explicit cast prevents warning, alternative would be to template functors::scalar_multiply on T as well
}

template<class VecV, class Device>
vector_scalar_multiply<VecV> operator-(vector_expression<VecV, Device> const& v){
	typedef typename VecV::value_type value_type;
	return vector_scalar_multiply<VecV>(v(), value_type(-1));//explicit cast prevents warning, alternative would be to template functors::scalar_multiply on T as well
}

///\brief Creates a vector having a constant value.
///
///@param scalar the value which is repeated
///@param elements the size of the resulting vector
template<class T>
typename boost::enable_if<std::is_arithmetic<T>, scalar_vector<T> >::type
repeat(T scalar, std::size_t elements){
	return scalar_vector<T>(elements,scalar);
}


#define SHARK_UNARY_VECTOR_TRANSFORMATION(name, F)\
template<class VecV, class Device>\
vector_unary<VecV,F<typename VecV::value_type> >\
name(vector_expression<VecV, Device> const& v){\
	return vector_unary<VecV, F<typename VecV::value_type> >(v(), F<typename VecV::value_type>());\
}
SHARK_UNARY_VECTOR_TRANSFORMATION(abs, functors::scalar_abs)
SHARK_UNARY_VECTOR_TRANSFORMATION(log, functors::scalar_log)
SHARK_UNARY_VECTOR_TRANSFORMATION(exp, functors::scalar_exp)
SHARK_UNARY_VECTOR_TRANSFORMATION(tanh,functors::scalar_tanh)
SHARK_UNARY_VECTOR_TRANSFORMATION(sqr, functors::scalar_sqr)
SHARK_UNARY_VECTOR_TRANSFORMATION(sqrt, functors::scalar_sqrt)
SHARK_UNARY_VECTOR_TRANSFORMATION(sigmoid, functors::scalar_sigmoid)
SHARK_UNARY_VECTOR_TRANSFORMATION(softPlus, functors::scalar_soft_plus)
SHARK_UNARY_VECTOR_TRANSFORMATION(elem_inv, functors::scalar_inverse)
#undef SHARK_UNARY_VECTOR_TRANSFORMATION

///\brief Adds two vectors
template<class VecV1, class VecV2, class Device>
vector_addition<VecV1, VecV2 > operator+ (
	vector_expression<VecV1, Device> const& v1,
	vector_expression<VecV2, Device> const& v2
){
	SIZE_CHECK(v1().size() == v2().size());
	return vector_addition<VecV1, VecV2>(v1(),v2());
}
///\brief Subtracts two vectors
template<class VecV1, class VecV2, class Device>
vector_addition<VecV1, vector_scalar_multiply<VecV2> > operator- (
	vector_expression<VecV1, Device> const& v1,
	vector_expression<VecV2, Device> const& v2
){
	SIZE_CHECK(v1().size() == v2().size());
	return vector_addition<VecV1, vector_scalar_multiply<VecV2> >(v1(),-v2());
}

///\brief Adds a vector plus a scalar which is interpreted as a constant vector
template<class VecV, class T, class Device>
typename boost::enable_if<
	std::is_convertible<T, typename VecV::value_type>, 
	vector_addition<VecV, scalar_vector<T> >
>::type operator+ (
	vector_expression<VecV, Device> const& v,
	T t
){
	return v + scalar_vector<T>(v().size(),t);
}

///\brief Adds a vector plus a scalar which is interpreted as a constant vector
template<class T, class VecV, class Device>
typename boost::enable_if<
	std::is_convertible<T, typename VecV::value_type>,
	vector_addition<VecV, scalar_vector<T> >
>::type operator+ (
	T t,
	vector_expression<VecV, Device> const& v
){
	return v + scalar_vector<T>(v().size(),t);
}

///\brief Subtracts a scalar which is interpreted as a constant vector.
template<class VecV, class T, class Device>
typename boost::enable_if<
	std::is_convertible<T, typename VecV::value_type> ,
	vector_addition<VecV, vector_scalar_multiply<scalar_vector<T> > >
>::type operator- (
	vector_expression<VecV, Device> const& v,
	T t
){
	return v - scalar_vector<T>(v().size(),t);
}

///\brief Subtracts a vector from a scalar which is interpreted as a constant vector
template<class VecV, class T, class Device>
typename boost::enable_if<
	std::is_convertible<T, typename VecV::value_type>,
	vector_addition<scalar_vector<T>, vector_scalar_multiply<VecV> >
>::type operator- (
	T t,
	vector_expression<VecV, Device> const& v
){
	return scalar_vector<T>(v().size(),t) - v;
}

#define SHARK_BINARY_VECTOR_EXPRESSION(name, F)\
template<class VecV1, class VecV2, class Device>\
vector_binary<VecV1, VecV2, F<typename common_value_type<VecV1,VecV2>::type> >\
name(vector_expression<VecV1, Device> const& v1, vector_expression<VecV2, Device> const& v2){\
	SIZE_CHECK(v1().size() == v2().size());\
	typedef typename common_value_type<VecV1,VecV2>::type type;\
	return vector_binary<VecV1, VecV2, F<type> >(v1(),v2(), F<type>());\
}
SHARK_BINARY_VECTOR_EXPRESSION(operator*, functors::scalar_binary_multiply)
SHARK_BINARY_VECTOR_EXPRESSION(element_prod, functors::scalar_binary_multiply)
SHARK_BINARY_VECTOR_EXPRESSION(operator/, functors::scalar_binary_divide)
SHARK_BINARY_VECTOR_EXPRESSION(element_div, functors::scalar_binary_divide)
SHARK_BINARY_VECTOR_EXPRESSION(min, functors::scalar_binary_min)
SHARK_BINARY_VECTOR_EXPRESSION(max, functors::scalar_binary_max)
#undef SHARK_BINARY_VECTOR_EXPRESSION


//operations of the form op(v,t)[i] = op(v[i],t)
#define SHARK_VECTOR_SCALAR_TRANSFORMATION(name, F)\
template<class T, class VecV, class Device> \
typename boost::enable_if< \
	std::is_convertible<T, typename VecV::value_type >,\
        vector_binary<VecV, scalar_vector<T>, F<typename std::common_type<typename VecV::value_type,T>::type> > \
>::type \
name (vector_expression<VecV, Device> const& v, T t){ \
	typedef typename std::common_type<typename VecV::value_type,T>::type type;\
	return  vector_binary<VecV, scalar_vector<T>, F<type> >(v(), scalar_vector<T>(v().size(),t) ,F<type>()); \
}
SHARK_VECTOR_SCALAR_TRANSFORMATION(operator/, functors::scalar_binary_divide)
SHARK_VECTOR_SCALAR_TRANSFORMATION(operator<, functors::scalar_less_than)
SHARK_VECTOR_SCALAR_TRANSFORMATION(operator<=, functors::scalar_less_equal_than)
SHARK_VECTOR_SCALAR_TRANSFORMATION(operator>, functors::scalar_bigger_than)
SHARK_VECTOR_SCALAR_TRANSFORMATION(operator>=, functors::scalar_bigger_equal_than)
SHARK_VECTOR_SCALAR_TRANSFORMATION(operator==, functors::scalar_equal)
SHARK_VECTOR_SCALAR_TRANSFORMATION(operator!=, functors::scalar_not_equal)
SHARK_VECTOR_SCALAR_TRANSFORMATION(min, functors::scalar_binary_min)
SHARK_VECTOR_SCALAR_TRANSFORMATION(max, functors::scalar_binary_max)
SHARK_VECTOR_SCALAR_TRANSFORMATION(pow, functors::scalar_binary_pow)
#undef SHARK_VECTOR_SCALAR_TRANSFORMATION

// operations of the form op(t,v)[i] = op(t,v[i])
#define SHARK_VECTOR_SCALAR_TRANSFORMATION_2(name, F)\
template<class T, class VecV, class Device> \
typename boost::enable_if< \
	std::is_convertible<T, typename VecV::value_type >,\
         vector_binary<scalar_vector<T>, VecV, F<typename std::common_type<typename VecV::value_type,T>::type> > \
>::type \
name (T t, vector_expression<VecV, Device> const& v){ \
	typedef typename std::common_type<typename VecV::value_type,T>::type type;\
	return  vector_binary<scalar_vector<T>, VecV, F<type> >(scalar_vector<T>(v().size(),t), v() ,F<type>()); \
}
SHARK_VECTOR_SCALAR_TRANSFORMATION_2(min, functors::scalar_binary_min)
SHARK_VECTOR_SCALAR_TRANSFORMATION_2(max, functors::scalar_binary_max)
#undef SHARK_VECTOR_SCALAR_TRANSFORMATION_2

template<class VecV1, class VecV2, class Device>
vector_binary<VecV1, VecV2, 
	functors::scalar_binary_safe_divide<typename common_value_type<VecV1,VecV2>::type > 
>
safe_div(
	vector_expression<VecV1, Device> const& v1, 
	vector_expression<VecV2, Device> const& v2, 
	typename common_value_type<VecV1,VecV2>::type defaultValue
){
	SIZE_CHECK(v1().size() == v2().size());
	typedef typename common_value_type<VecV1,VecV2>::type result_type;
	
	typedef functors::scalar_binary_safe_divide<result_type> functor_type;
	return vector_binary<VecV1, VecV2, functor_type>(v1(),v2(), functor_type(defaultValue));
}

/////VECTOR REDUCTIONS

/// \brief sum v = sum_i v_i
template<class VecV, class Device>
typename VecV::value_type
sum(vector_expression<VecV, Device> const& v) {
	typedef typename VecV::value_type value_type;
	vector_fold<functors::scalar_binary_plus<typename VecV::value_type> > kernel;
	return kernel(eval_block(v),value_type());
}

/// \brief max v = max_i v_i
template<class VecV, class Device>
typename VecV::value_type
max(vector_expression<VecV, Device> const& v) {
	typedef typename VecV::value_type value_type;
	vector_fold<functors::scalar_binary_max<typename VecV::value_type> > kernel;
	auto const& elem_result = eval_block(v);
	return kernel(elem_result,elem_result(0));
}

/// \brief min v = min_i v_i
template<class VecV, class Device>
typename VecV::value_type
min(vector_expression<VecV, Device> const& v) {
	typedef typename VecV::value_type value_type;
	vector_fold<functors::scalar_binary_min<typename VecV::value_type> > kernel;
	auto const& elem_result = eval_block(v);
	return kernel(elem_result,elem_result(0));
}

/// \brief arg_max v = arg max_i v_i
template<class VecV, class Device>
std::size_t arg_max(vector_expression<VecV, Device> const& v) {
	SIZE_CHECK(v().size() > 0);
	auto const& elem_result = eval_block(v);
	return std::max_element(elem_result.begin(),elem_result.end()).index();
}

/// \brief arg_min v = arg min_i v_i
template<class VecV, class Device>
std::size_t arg_min(vector_expression<VecV, Device> const& v) {
	SIZE_CHECK(v().size() > 0);
	return arg_max(-v);
}

/// \brief soft_max v = ln(sum(exp(v)))
///
/// Be aware that this is NOT the same function as used in machine learning: exp(v)/sum(exp(v))
///
/// The function is computed in an numerically stable way to prevent that too high values of v_i produce inf or nan.
/// The name of the function comes from the fact that it behaves like a continuous version of max in the respect that soft_max v <= v.size()*max(v)
/// max is reached in the limit as the gap between the biggest value and the rest grows to infinity.
template<class VecV, class Device>
typename VecV::value_type
soft_max(vector_expression<VecV, Device> const& v) {
	typename VecV::value_type maximum = max(v);
	return std::log(sum(exp(v - maximum))) + maximum;
}


////implement all the norms based on sum!

/// \brief norm_1 v = sum_i |v_i|
template<class VecV, class Device>
typename real_traits<typename VecV::value_type >::type
norm_1(vector_expression<VecV, Device> const& v) {
	return sum(abs(eval_block(v)));
}

/// \brief norm_2 v = sum_i |v_i|^2
template<class VecV, class Device>
typename real_traits<typename VecV::value_type >::type
norm_sqr(vector_expression<VecV, Device> const& v) {
	return sum(sqr(eval_block(v)));
}

/// \brief norm_2 v = sqrt (sum_i |v_i|^2 )
template<class VecV, class Device>
typename real_traits<typename VecV::value_type >::type
norm_2(vector_expression<VecV, Device> const& v) {
	using std::sqrt;
	return sqrt(norm_sqr(v));
}

/// \brief norm_inf v = max_i |v_i|
template<class VecV, class Device>
typename real_traits<typename VecV::value_type >::type
norm_inf(vector_expression<VecV, Device> const& v){
	return max(abs(eval_block(v)));
}

/// \brief index_norm_inf v = arg max_i |v_i|
template<class VecV, class Device>
std::size_t index_norm_inf(vector_expression<VecV, Device> const& v){
	return arg_max(abs(eval_block(v)));
}

// inner_prod (v1, v2) = sum_i v1_i * v2_i
template<class VecV1, class VecV2, class Device>
decltype(
	typename VecV1::value_type() * typename VecV2::value_type()
)
inner_prod(
	vector_expression<VecV1, Device> const& v1,
	vector_expression<VecV2, Device> const& v2
) {
	SIZE_CHECK(v1().size() == v2().size());
	typedef decltype(
		typename VecV1::value_type() * typename VecV2::value_type()
	) value_type;
	value_type result = value_type();
	kernels::dot(eval_block(v1),eval_block(v2),result);
	return result;
}

}

}

#endif
