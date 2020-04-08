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

#include "../kernels/device_traits.hpp"
#include "../detail/check.hpp"

#include <boost/serialization/collection_size_type.hpp>
#include <boost/serialization/array.hpp>
#include <boost/serialization/nvp.hpp>
#include <boost/serialization/vector.hpp>


namespace remora{
	
//forward declaration of assignment operations
template<std::size_t N, class TensorX, class TensorV, class Device>
TensorX& assign(tensor_expression<N, TensorX, Device>& x, tensor_expression<N, TensorV, Device> const& v);

namespace kernels{
	template<std::size_t N, class Functor, class TensorA, class Device>
	void apply(tensor_expression<N, TensorA, Device>&, Functor const&);
}


	
template<class T, class Axis, class Tag>
class dense_tensor_adaptor<T, Axis, Tag, cpu_tag>: public tensor_expression<Axis::num_dims, dense_tensor_adaptor<T, Axis, Tag, cpu_tag>, cpu_tag > {
public:
	static_assert(Tag::num_dims == Axis::num_dims);
	static constexpr std::size_t num_dims = Axis::num_dims;
	typedef std::size_t size_type;
	typedef typename std::remove_const<T>::type value_type;
	typedef value_type const& const_reference;
	typedef T&  reference;

	typedef dense_tensor_adaptor<value_type const, Axis, Tag, cpu_tag> const_closure_type;
	typedef dense_tensor_adaptor closure_type;
	typedef dense_tensor_storage<T, Tag> storage_type;
	typedef dense_tensor_storage<value_type const, Tag> const_storage_type;
	typedef elementwise<dense_tag> evaluation_category;
	typedef Axis axis;
	
	bool check_storage_invariants()const{
		auto dense_tags = Tag::to_array();
		auto axis_id = axis::to_array();
		auto axis_id_pos = axis::inverse_t::to_array();
		
		for(std::size_t i = 0; i != axis::num_dims; ++i){
			//nothing to do if axis is not dense
			if(dense_tags[i] == 0)
				continue;
			//check if we are at the minor axis
			unsigned next_id = axis_id[i] + 1;
			if(next_id == axis::num_dims){
				//dense minor axis must have stride 1
				if (m_storage.strides[i] != 1)
					return false;
				continue;
			}
			
			std::size_t pos = axis_id_pos[next_id];
			if(m_storage.strides[i] != m_shape[pos] * m_storage.strides[pos])
				return false;
		}
		return true;
	}

	// Construction and destruction
	dense_tensor_adaptor(dense_tensor_adaptor const&) = default;
	dense_tensor_adaptor(dense_tensor_adaptor&&) = default;
	
	// construction from tensor expressions
	template<class Tensor>
	dense_tensor_adaptor(tensor_expression<num_dims, Tensor, cpu_tag> const& expression)
	: m_storage(expression().raw_storage())
	, m_shape(expression().shape()){
		static_assert(std::is_same<axis, typename Tensor::axis>::value, "Can only create adaptors from Tensors with same axis");
		REMORA_SIZE_CHECK(check_storage_invariants());
	}
	
	template<class Tensor>
	dense_tensor_adaptor(tensor_expression<num_dims, Tensor, cpu_tag>& expression)
	: m_storage(expression().raw_storage())
	, m_shape(expression().shape()){
		static_assert(std::is_same<axis, typename Tensor::axis>::value, "Can only create adaptors from Tensors with same axis");
		REMORA_SIZE_CHECK(check_storage_invariants());
	}

	dense_tensor_adaptor(storage_type const& storage, no_queue, tensor_shape<num_dims> const& size)
	:m_storage(storage),m_shape(size){
		REMORA_SIZE_CHECK(check_storage_invariants());
	}	

	dense_tensor_adaptor& operator = (dense_tensor_adaptor const& e){
		REMORA_SIZE_CHECK(shape() == e().shape());
		return assign(*this, typename tensor_temporary<dense_tensor_adaptor>::type(e));
	}
	template<class E>
	dense_tensor_adaptor& operator = (tensor_expression<num_dims, E, cpu_tag> const& e){
		REMORA_SIZE_CHECK(shape() == e().shape());
		return assign(*this, typename tensor_temporary<E>::type(e));
	}
	
	// --------------
	// Accessors
	// --------------
	
	/// \brief Return the size of the vector.
	tensor_shape<num_dims> shape() const {
		return m_shape;
	}
	
	///\brief Returns the underlying storage structure for low level access
	storage_type raw_storage() const{
		return m_storage;
	}
	
	no_queue queue() const{
		return no_queue();
	}
	// --------------
	// Evaluation
	// --------------
	
	struct tensor_element{
		typedef const_reference result_type;
		template<typename... Indices, class = typename std::enable_if<sizeof...(Indices) == num_dims,void>::type>
		result_type operator()(Indices... idx) const{
			std::size_t elem = axis::element(std::array<std::size_t,axis::num_dims>{idx...}, storage.strides);
			return storage.values[elem];
		}
		const_storage_type storage;
	};
	
	auto elements() const{
		return tensor_element{raw_storage()};
	}

	/// \brief Return a const reference to the element \f$i\f$
	/// \param i index of the element
	template<class... Indices, class = typename std::enable_if<sizeof...(Indices) == num_dims,void>::type>
	reference operator()(Indices... idx) const {
		static_assert(sizeof...(idx) == axis::num_dims, "Must pass same amount of parameters as dimensions in the tensor");
		std::size_t elem = axis::element(std::array<std::size_t,axis::num_dims>{idx...}, m_storage.strides);
		return m_storage.values[elem];
	}
	
	/// \brief Return a reference to the element \f$i\f$
	/// \param i index of the element
	template<class... Indices, class = typename std::enable_if<sizeof...(Indices) == num_dims,void>::type>
	reference operator()(Indices... idx){
		static_assert(sizeof...(idx) == axis::num_dims, "Must pass same amount of parameters as dimensions in the tensor");
		std::size_t elem = axis::element(std::array<std::size_t,axis::num_dims>{idx...}, m_storage.strides);
		return m_storage.values[elem];
	}
	
	
	void clear(){
		typename device_traits<cpu_tag>:: template constant<value_type> Constant;
		kernels::apply(*this, Constant(value_type/*zero*/()));
	}
	
private:
	storage_type m_storage;
	tensor_shape<num_dims> m_shape;
};


	



template<class T, class Axis>
class tensor<T, Axis, cpu_tag>: public tensor_expression<Axis::num_dims, tensor<T, Axis, cpu_tag>, cpu_tag > {
public:
	typedef std::vector<T> array_type;
	typedef typename array_type::value_type value_type;
	typedef typename array_type::const_reference const_reference;
	typedef typename array_type::reference reference;
	typedef typename array_type::size_type size_type;
	
	static constexpr std::size_t num_dims = Axis::num_dims;
	typedef continuous_tensor_storage<num_dims, T> storage_type;
	typedef continuous_tensor_storage<num_dims, T const> const_storage_type;
	typedef dense_tensor_adaptor<T const, Axis, typename storage_type::dense_axis_tag, cpu_tag> const_closure_type;
	typedef dense_tensor_adaptor<T, Axis, typename storage_type::dense_axis_tag, cpu_tag> closure_type;
	
	typedef elementwise<dense_tag> evaluation_category;
	typedef Axis axis;
	

	tensor() = default;
	tensor(tensor&&) = default;
	tensor(tensor const&) = default;
	tensor& operator=(tensor const&) = default;
	tensor& operator=(tensor&&) = default;

	/// \brief Dense tensor constructor with defined shape
	/// \param shape sizes of each dimension
	tensor(tensor_shape<num_dims> const& shape)
	: m_shape(shape)
	, m_values(shape.num_elements())
	, m_strides(axis::compute_dense_strides(shape).shape_array){}

	/// \brief  Dense tensor constructor with defined shape and initial value for all the matrix elements
	/// \param shape sizes of each dimension
	/// \param init initial value assigned to all elements
	tensor(tensor_shape<num_dims> const& shape, value_type const& init)
	: m_shape(shape)
	, m_values(shape.num_elements(), init)
	, m_strides(axis::compute_dense_strides(shape).shape_array){}

	/// \brief Creates a tensor as a copy of the result of E
	/// 
	/// evaluates the expression and assign the
	/// results to this using a call to assign.
	/// A temporary is created to prevent aliasing.
	///
	/// \param e is a tensor expression
	template<class E>
	tensor(tensor_expression<num_dims, E, cpu_tag> const& e)
	: m_shape(e().shape())
	, m_values(m_shape.num_elements())
	, m_strides(axis::compute_dense_strides(m_shape).shape_array){
		assign(*this,e);
	}
	
	/// \brief Assigns e to this
	/// 
	/// evaluates the expression and assign the
	/// results to this using a call to assign.
	/// A temporary is created to prevent aliasing.
	///
	/// \param e is a tensor expression
	template<class E>
	tensor& operator = (tensor_expression<num_dims, E, cpu_tag> const& e) {
		tensor temporary(e);
		swap(temporary);
		return *this;
	}
	
	/// \brief Assigns e to this
	/// 
	/// evaluates the expression and assign the
	/// results to this using a call to assign.
	///
	/// \param e is a tensor expression
	template<class E>
	tensor& operator = (tensor_container<num_dims, E, cpu_tag> const& e){
		resize(e().shape());
		return assign(*this, e);
	}
	
	/// \brief Return the shape of the tensor
	tensor_shape<num_dims> shape() const {
		return m_shape;
	}
	
	///\brief Returns the underlying storage structure for low level access
	storage_type raw_storage(){
		return {m_values.data(), m_strides};
	}
	
	///\brief Returns the underlying storage structure for low level access
	const_storage_type raw_storage() const{
		return {m_values.data(), m_strides};
	}
	
	no_queue queue() const{
		return no_queue();
	}
	
	/// \brief Swaps contents of this tensor with the given tensor
	void swap(tensor& m){
		std::swap(m_shape, m.shape);
		std::swap(m_strides, m.m_strides);
		m_values.swap(m.m_values);
	}
	
	/// \brief Resize a tensor to a new shape. If shape is different from previous, the data is not preserved.
	/// \param shape new 
	void resize(tensor_shape<num_dims> const& shape) {
		m_shape = shape;
		m_strides = axis::compute_dense_strides(shape).shape_array;
		m_values.resize(shape.num_elements());
	}
	
	// --------------
	// Element access
	// --------------
	
	struct tensor_element{
		typedef const_reference result_type;
		template<typename... Indices, class = typename std::enable_if<sizeof...(Indices) == num_dims,void>::type>
		result_type operator()(Indices... idx) const{
			std::size_t elem = axis::element(std::array<std::size_t,axis::num_dims>{idx...}, storage.strides);
			return storage.values[elem];
		}
		const_storage_type storage;
	};
	
	auto elements() const{
		return tensor_element{raw_storage()};
	}

	/// \brief Return a const reference to the element \f$i\f$
	/// \param i index of the element
	template<class... Indices, class = typename std::enable_if<sizeof...(Indices) == num_dims,void>::type>
	const_reference operator()(Indices... idx) const {
		static_assert(sizeof...(idx) == axis::num_dims, "Must pass same amount of parameters as dimensions in the tensor");
		std::size_t elem = axis::element(std::array<std::size_t,axis::num_dims>{idx...}, m_strides);
		return m_values[elem];
	}
	
	/// \brief Return a reference to the element \f$i\f$
	/// \param i index of the element
	template<class... Indices, class = typename std::enable_if<sizeof...(Indices) == num_dims,void>::type>
	reference operator()(Indices... idx){
		static_assert(sizeof...(idx) == axis::num_dims, "Must pass same amount of parameters as dimensions in the tensor");
		std::size_t elem = axis::element(std::array<std::size_t,axis::num_dims>{idx...}, m_strides);
		return m_values[elem];
	}
	
	void clear(){
		std::fill(m_values.begin(), m_values.end(), value_type/*zero*/());
	}
	
	// Serialization
	template<class Archive>
	void serialize(Archive& ar, const unsigned int /* file_version */) {
		// serialize the sizes
		ar& boost::serialization::make_nvp("shape",m_shape);
		ar& boost::serialization::make_nvp("values",m_values);
		
		// copy the values back if loading
		if (Archive::is_loading::value) {
			m_strides = axis::compute_dense_strides(shape);
		}
	}
private:
	tensor_shape<num_dims> m_shape;
	array_type m_values;
	std::array<std::size_t, num_dims> m_strides;
};
	
	// template<class E>
	// matrix(vector_set_expression<E, cpu_tag> const& e)
	// : m_size1(e().shape()[0])
	// , m_size2(e().shape()[1])
	// , m_values(m_size1 * m_size2) {
		// assign(*this,e().expression());
	// }

	// friend void swap_rows(matrix& a, size_type i, matrix& b, size_type j){
		// REMORA_SIZE_CHECK(i < a.size1());
		// REMORA_SIZE_CHECK(j < b.size1());
		// REMORA_SIZE_CHECK(a.size2() == b.size2());
		// for(std::size_t k = 0; k != a.size2(); ++k){
			// std::swap(a(i,k),b(j,k));
		// }
	// }
	
	// void swap_rows(size_type i, size_type j) {
		// if(i == j) return;
		// for(std::size_t k = 0; k != size2(); ++k){
			// std::swap((*this)(i,k),(*this)(j,k));
		// }
	// }
	
	
	// friend void swap_columns(matrix& a, size_type i, matrix& b, size_type j){
		// REMORA_SIZE_CHECK(i < a.size2());
		// REMORA_SIZE_CHECK(j < b.size2());
		// REMORA_SIZE_CHECK(a.size1() == b.size1());
		// for(std::size_t k = 0; k != a.size1(); ++k){
			// std::swap(a(k,i),b(k,j));
		// }
	// }
	
	// void swap_columns(size_type i, size_type j) {
		// if(i == j) return;
		// for(std::size_t k = 0; k != size1(); ++k){
			// std::swap((*this)(k,i),(*this)(k,j));
		// }
	// }

	// void swap_rows(size_type i, size_type j){
		// for(std::size_t k = 0; k != size2(); ++k){
			// std::swap((*this)(i,k),(*this)(j,k));
		// }
	// }
	
	// void swap_columns(size_type i, size_type j){
		// for(std::size_t k = 0; k != size1(); ++k){
			// std::swap((*this)(k,i),(*this)(k,j));
		// }
	// }
// }
}

#endif
