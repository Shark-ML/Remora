//===========================================================================
/*!
 * 
 *
 * \brief       Axis descriptor Traits-class
 *
 * \author      O. Krause
 * \date        2020
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

#ifndef REMORA_DETAIL_AXIS_HPP
#define REMORA_DETAIL_AXIS_HPP

#include "integer_list.hpp"
#include <utility> // index_sequence
namespace remora{


template<unsigned... Seq>
using axis_set = integer_list<unsigned, Seq...>;


template<unsigned... Seq>
struct axis: public axis_set<Seq...>{
public: 
	typedef typename axis_set<Seq...>::array_type array_type;
private:
	struct invert_helper{
		static constexpr array_type apply(array_type seq0){
			array_type seq={0};
			for(std::size_t i = 0; i != sizeof...(Seq); ++i)
				seq.values[seq0.values[i]] = i;
			return seq;
		}
	};
	
	template<std::size_t Axis>
	struct slice_helper{
		static constexpr array_type apply(array_type seq0){
			array_type seq={0};
			for(std::size_t i = 0, j = 0; i != sizeof...(Seq); ++i){
				if (i == Axis)
					continue;
				seq.values[j] = seq0.values[i] - (seq0.values[i] > seq0.values[Axis]);
				++j;
			}
			return seq;
		}
	};
	
	template<std::size_t Axis>
	struct split_helper{
		static constexpr array_type apply(array_type seq0){
			array_type seq={0};
			for(std::size_t i = 0, j = 0; i != sizeof...(Seq); ++i, ++j){
				if (i == Axis){
					//copy axis and insert new element after the position
					seq.values[j] = seq0.values[Axis];
					++j;
					seq.values[j] = seq0.values[Axis] + 1;
				}else{
					// all axes with value larger or equal the new axis id will be increased by one
					seq.values[j] = seq0.values[i] + (seq0.values[i] > seq0.values[Axis]);
				}
			}
			return seq;
		}
	};
	
	//taking a constrexpr functor F with a static apply function, computes an axis transformation as array
	//and converts the array to an axis<...> object.
	template<class F, std::size_t... Inds>
	static constexpr axis<F::apply({Seq...}).values[Inds]...> apply(std::index_sequence<Inds...>);
		
public:
	typedef std::size_t size_type;
	typedef std::ptrdiff_t difference_type;

	/// \brief Compute the axis order of a tensor with the Nth axis sliced away.
	/// This removes the Nth element and decrements all elements which have a larger value.
	template<std::size_t Axis>
	using slice_t = decltype(apply<slice_helper<Axis> >(std::make_index_sequence<sizeof...(Seq)-1>()));
	
	/// \brief Splits an Axis into two
	/// Split adds a new element (Value+1) in the sequence behind the Value at position Axis and increments all existing elements 
	/// that have larger value than Value by 1. This is to ensure that slice<Axis+1> afterwards is the inverse operation.
	/// Example:
	/// axis<2,3,1,0>::split_t<2>=axis<3,4,1,2,0> where 2 is the newly inserted element.
	template<std::size_t Axis>
	using split_t = decltype(apply<split_helper<Axis> >(std::make_index_sequence<sizeof...(Seq)+1>()));
	
	/// \brief Permutes an axis set
	/// the outcome is the mapping Selection_i -> axis[Selection[i]]
	/// The difference to select_t is that the result is an axis and the permutation must have same number of elements as axis.
	/// e.g.: axis<1,3,0,2>::permute_t<2,0,3,1> = axis<0,1,2,3>
	template<std::size_t... Selection>
	using permute_t = axis<axis::template element_v<Selection>...>;
	
	
	/// \brief Represents the inverted axis object. 
	typedef decltype(apply<invert_helper>(std::make_index_sequence<sizeof...(Seq)>())) inverse_t;
	
	template<class Array>
	static Array compute_dense_strides(Array const& shape){
		constexpr auto map = inverse_t::to_array();
		Array result;
		size_type stride = 1;
		for(std::size_t i = sizeof...(Seq); i > 0; --i){
			size_type axis = map[i - 1];
			result[axis] = stride;
			stride *= shape[axis];
		}
		return result;
	}
	
	/// \brief Returns the stride of the n-th leading dimension
	///
	/// N= 0 is equivalent to the major direction, i.e. the largest stride in the tensor.
	template<std::size_t N, class Array>
	static size_type leading(Array const& strides){
		return strides[inverse_t::template element_v<sizeof...(Seq) - 1 - N>];
	}
	
	// Indexing conversion to storage element
	template<class Array1, class Array2>
	static size_type element(Array1 const& indices, Array2 const& strides) {
		size_type elem = 0;
		for (std::size_t i = 0; i != sizeof...(Seq); ++i)
			elem += strides[i] * indices[i];
		return elem;
	}
	
	template<class Array>
	static Array to_axis(Array const& arr){
		
		Array result;
		auto map = axis::to_array();
		for(std::size_t i = 0; i != sizeof...(Seq); ++i){
			size_type axis = map[i];
			result[i] = arr[axis];
		}
		return result;
	}
	
	template<class Array>
	static Array from_axis(Array const& arr){
		return inverse_t::to_axis(arr);
	}

};



namespace detail{
	template<unsigned... Ns>
	axis<Ns...> make_default_axis_helper(std::integer_sequence<unsigned, Ns...>);
	
	template<class Axis, class FilterList, std::size_t N = 0>
	struct filter_slice;
	
	template<class Axis, bool Tag, bool... FilterTags, std::size_t N>
	struct filter_slice<Axis, integer_list<bool, Tag, FilterTags...>, N >{
		typedef typename std::conditional<Tag, typename Axis::template slice_t<N>, Axis>::type sliced_axis;
		typedef typename filter_slice<sliced_axis, integer_list<bool, FilterTags...>, N + 1 - Tag>::type type;
	};
	template<class Axis, std::size_t N>
	struct filter_slice<Axis, integer_list<bool>, N >{
		typedef Axis type;
	};
}

template<unsigned N>
using default_axis = decltype(detail::make_default_axis_helper(std::make_integer_sequence<unsigned, N>()));
typedef default_axis<2> row_major;
typedef axis<1,0> column_major;

template<std::size_t NumDims>
struct unknown_axis{
	typedef std::size_t size_type;
	typedef std::ptrdiff_t difference_type;
	
	/// \brief number of dimensions
	static constexpr std::size_t num_dims = NumDims;

	/// \brief Represents the inverted axis object. 
	typedef unknown_axis inverse_t;

};

}

#endif