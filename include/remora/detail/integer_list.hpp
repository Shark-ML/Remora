//===========================================================================
/*!
 * 
 *
 * \brief       Compile-Time list of integers
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

#ifndef REMORA_DETAIL_INTEGER_LIST_HPP
#define REMORA_DETAIL_INTEGER_LIST_HPP

#include <utility> // index_sequence
namespace remora{


template<class T, T... Seq>
struct integer_list{
public:
	struct array_type{
		constexpr T& operator[](std::size_t i){
			return values[i];
		}
		constexpr T operator[](std::size_t i)const{
			return values[i];
		}			
		T values[sizeof...(Seq) + 1];// for insert
	
	}; 
private:
	static constexpr T Nth_integer_helper(std::size_t N){
		T seq[] = {Seq...};
		return seq[N];
	}
	static constexpr unsigned min_integer_helper(){
		T array[]={Seq...};
		T minv = array[0];
		for(std::size_t i = 1; i != sizeof...(Seq); ++i){
			minv = array[i] < minv? array[i]: minv;
		}
		return minv;
	}
	
	static constexpr std::size_t index_of_helper(T V, array_type const& seq){
		std::size_t j = 0;
		for(std::size_t i = 0; i != sizeof...(Seq); ++i){
			if (seq.values[i] == V)
				j = i;
		}
		return j;
	}
	
	template<std::size_t N>
	struct remove_helper{
		static constexpr array_type apply(array_type seq0){
			array_type seq={0};
			for(std::size_t i = 0, j = 0; i != sizeof...(Seq); ++i){
				if (i == N)
					continue;
				seq.values[j] = seq0.values[i];
				++j;
			}
			return seq;
		}
	};
	
	template<std::size_t N, T V>
	struct insert_helper{
		static constexpr array_type apply(array_type seq0){
			array_type seq={0};
			seq.values[N] = V;
			for(std::size_t i = 0, j = 0; i != sizeof...(Seq); ++i){
				if (i == N){
					++j;
				}
				seq.values[j] = seq0.values[i];
				++j;
			}
			
			return seq;
		}
	};
	
	struct identity_helper{
		static constexpr array_type apply(array_type seq){
			return seq;
		}
	};
	
	//taking a constrexpr functor F with a static apply function, computes a transformation as array
	//and converts the array to an integer_list<...> object.
	template<class F, std::size_t... Inds>
	static constexpr integer_list<T, F::apply(array_type{Seq...}).values[Inds]...> apply(std::index_sequence<Inds...>);
public:
	/// \brief number of dimensions
	static constexpr std::size_t num_dims = sizeof...(Seq);
	/// \brief Returns the nth element.
	/// Note: if N= num_dims, the result is unspecified but will not lead to compile-errors.
	template<std::size_t N>
	static constexpr T element_v = Nth_integer_helper(N);
	
	/// \brief Returns the index of the element V.
	template<T V>
	static constexpr std::size_t index_of_v = index_of_helper(V, {Seq...});
	
	/// \brief Value of the minimum element
	static constexpr T min_element = min_integer_helper();
	
	/// \brief This removes the Nth element.
	template<std::size_t N>
	using remove_t = decltype(apply<remove_helper<N> >(std::make_index_sequence<num_dims-1>()));
	
	/// \brief Insert a new element at position N-1
	/// All current elements at position [N, N+1,...,num_dims - 1] are moved by one to the right
	template<std::size_t N, T V>
	using insert_t = decltype(apply<insert_helper<N, V> >(std::make_index_sequence<num_dims+1>()));

	/// \brief selects a subrange based on the given arguments
	/// the outcome is the mapping Selection_i -> list[Selection[i]]
	/// e.g.: integer_list<1,3,0,2>::select_t<2,0,3> = integer_list<0,1,2>
	template<std::size_t... Selection>
	using select_t = integer_list<T, element_v<Selection>...>;
	
	/// \brief Selects the first N elements
	/// the outcome is equivalent to select_t<0,1,2,...,N-1> 
	/// e.g.: integer_list<1,3,0,2>::front_t<2> = integer_list<1,3>
	template<std::size_t N >
	using front_t = decltype(apply<identity_helper >(std::make_index_sequence<N>()));
	
	template<class F> 
	using transform_t = decltype(apply<F>(std::make_index_sequence<num_dims>()));
	
	static constexpr array_type to_array(){
		return {Seq...};
	}
};

}

#endif