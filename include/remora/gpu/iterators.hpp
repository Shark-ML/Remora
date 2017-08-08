//===========================================================================
/*!
 * 
 *
 * \brief       Iterators for gpu expressions
 *
 * \author      O. Krause
 * \date        2017
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

#ifndef REMORA_GPU_ITERATORS_HPP
#define REMORA_GPU_ITERATORS_HPP

#include <boost/compute/functional/detail/unpack.hpp>
#include <boost/compute/iterator/zip_iterator.hpp>
#include <boost/compute/iterator/constant_iterator.hpp>
#include <boost/compute/iterator/transform_iterator.hpp>
#include <tuple>

namespace remora{namespace gpu{ namespace detail{
	
template<class Iterator1, class Iterator2, class Functor>
struct binary_transform_iterator
: public boost::compute::transform_iterator<
	boost::compute::zip_iterator<boost::tuple<Iterator1, Iterator2> >,
	boost::compute::detail::unpacked<Functor>
>{
	typedef boost::compute::transform_iterator<
		boost::compute::zip_iterator<boost::tuple<Iterator1, Iterator2> >,
		boost::compute::detail::unpacked<Functor>
	> self_type;
	binary_transform_iterator(){}
	binary_transform_iterator(
		Functor const& f,
		Iterator1 const& iter1, Iterator1 const& iter1_end,
		Iterator2 const& iter2, Iterator2 const& iter2_end
	): self_type(boost::compute::make_zip_iterator(boost::make_tuple(iter1,iter2)), boost::compute::detail::unpack(f)){}
};

template<class Closure>
class indexed_iterator : public boost::iterator_facade<
	indexed_iterator<Closure>,
        typename Closure::value_type,
        std::random_access_iterator_tag,
	typename Closure::value_type
>{
public:
	indexed_iterator() = default;
	indexed_iterator(Closure const& closure, std::size_t index)
	: m_closure(closure)
	, m_index(index){}
		
	template<class C>
	indexed_iterator(indexed_iterator<C> const& other)
	: m_closure (other.m_closure)
	, m_index(other.m_index){}

	template<class C>
	indexed_iterator& operator=(indexed_iterator<C> const& other){
		m_closure = other.m_closure;
		m_index = other.m_index;
		return *this;
	}

	size_t get_index() const{
		return m_index;
	}

	/// \internal_
	template<class Expr>
	auto operator[](Expr const& expr) const-> decltype(std::declval<Closure>()(expr)){
		return m_closure(expr);
	}

private:
	friend class ::boost::iterator_core_access;

	/// \internal_
	typename Closure::value_type dereference() const
	{
		return typename Closure::value_type();
	}

	/// \internal_
	template<class C>
	bool equal(indexed_iterator<C> const& other) const
	{
		return m_index == other.m_index;
	}

	/// \internal_
	void increment()
	{
		m_index++;
	}

	/// \internal_
	void decrement()
	{
		m_index--;
	}

	/// \internal_
	void advance(std::ptrdiff_t n)
	{
		m_index = static_cast<size_t>(static_cast<std::ptrdiff_t>(m_index) + n);
	}

	/// \internal_
	template<class C>
	std::ptrdiff_t distance_to(indexed_iterator<C> const& other) const
	{
		return static_cast<std::ptrdiff_t>(other.m_index - m_index);
	}

private:
	Closure m_closure;
	std::size_t m_index;
	template<class> friend class indexed_iterator;
};


template<class Iterator>
class subrange_iterator : public boost::iterator_facade<
	subrange_iterator<Iterator>,
        typename Iterator::value_type,
        std::random_access_iterator_tag,
	typename Iterator::value_type
>{
public:
	subrange_iterator() = default;
	subrange_iterator(Iterator const &it, Iterator const& /*end*/, std::size_t startIterIndex,std::size_t /*startIndex*/)
	: m_iterator(it+startIterIndex){}
		
	template<class I>
	subrange_iterator(subrange_iterator<I> other):m_iterator(other.m_iterator){}

	template<class I>
	subrange_iterator& operator=(subrange_iterator<I> const& other){
		m_iterator = other.m_iterator;
		return *this;
	}

	size_t get_index() const{
		return m_iterator.index();
	}

	/// \internal_
	template<class Expr>
	auto operator[](Expr const& expr) const-> decltype(std::declval<Iterator>()[expr]){
		return m_iterator[expr];
	}

private:
	friend class ::boost::iterator_core_access;

	/// \internal_
	typename Iterator::value_type dereference() const
	{
		return typename Iterator::value_type();
	}

	/// \internal_
	template<class I>
	bool equal(subrange_iterator<I> const& other) const
	{
		return m_iterator == other.m_iterator;
	}

	/// \internal_
	void increment()
	{
		++m_iterator;
	}

	/// \internal_
	void decrement()
	{
		--m_iterator;
	}

	/// \internal_
	void advance(std::ptrdiff_t n)
	{
		m_iterator +=n;
	}

	/// \internal_
	template<class I>
	std::ptrdiff_t distance_to(subrange_iterator<I> const& other) const
	{
		return static_cast<std::ptrdiff_t>(other.m_iterator - m_iterator);
	}

private:
	Iterator m_iterator;
	template<class> friend class subrange_iterator;
};

}}}


namespace boost{namespace compute{
template<class I1, class I2, class F>
struct is_device_iterator<remora::gpu::detail::binary_transform_iterator<I1,I2, F> > : boost::true_type {};
template<class Closure>
struct is_device_iterator<remora::gpu::detail::indexed_iterator<Closure> > : boost::true_type {};
template<class Iterator>
struct is_device_iterator<remora::gpu::detail::subrange_iterator<Iterator> > : boost::true_type {};
}}

#endif