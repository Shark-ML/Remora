/*!
 * 
 *
 * \brief       -
 *
 * \author      -
 * \date        -
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

/*
 *  Copyright (c) 2001-2003 Joel de Guzman
 *
 *  Use, modification and distribution is subject to the Boost Software
 *  License, Version 1.0. (See accompanying file LICENSE_1_0.txt or copy at
 *    http://www.boost.org/LICENSE_1_0.txt)
 */

#ifndef SHARK_LINALG_BLAS_DETAIL_RETURNTYPE_DEDUCTION_HPP
#define SHARK_LINALG_BLAS_DETAIL_RETURNTYPE_DEDUCTION_HPP

// See original in boost-sandbox/boost/utility/type_deduction.hpp for comments

#include <boost/mpl/vector/vector20.hpp>
#include <boost/mpl/at.hpp>
#include <boost/mpl/or.hpp>
#include <boost/mpl/identity.hpp>
#include <boost/type_traits/remove_cv.hpp>
#include <boost/type_traits/is_same.hpp>
#include <boost/utility/enable_if.hpp>

namespace shark{ namespace blas{

struct error_cant_deduce_type {};

  namespace type_deduction_detail
  {
    typedef char(&bool_value_type)[1];
    typedef char(&float_value_type)[2];
    typedef char(&double_value_type)[3];
    typedef char(&long_double_value_type)[4];
    typedef char(&char_value_type)[5];
    typedef char(&schar_value_type)[6];
    typedef char(&uchar_value_type)[7];
    typedef char(&short_value_type)[8];
    typedef char(&ushort_value_type)[9];
    typedef char(&int_value_type)[10];
    typedef char(&uint_value_type)[11];
    typedef char(&long_value_type)[12];
    typedef char(&ulong_value_type)[13];
    
    typedef char(&x_value_type)[14];
    typedef char(&y_value_type)[15];

    typedef char(&cant_deduce_type)[16];

    template <typename T, typename PlainT = typename boost::remove_cv<T>::type>
    struct is_basic
        : boost::mpl::or_<
          typename boost::mpl::or_<
              boost::is_same<PlainT, bool>
            , boost::is_same<PlainT, float>
            , boost::is_same<PlainT, double>
            , boost::is_same<PlainT, long double>
          >::type,
          typename boost::mpl::or_<
              boost::is_same<PlainT, char>
            , boost::is_same<PlainT, signed char>
            , boost::is_same<PlainT, unsigned char>
            , boost::is_same<PlainT, short>
            , boost::is_same<PlainT, unsigned short>
            >::type,
          typename boost::mpl::or_<
              boost::is_same<PlainT, int>
            , boost::is_same<PlainT, unsigned int>
            , boost::is_same<PlainT, long>
            , boost::is_same<PlainT, unsigned long>
            >::type
        > {};

    struct asymmetric;

    template <typename X, typename Y>
    cant_deduce_type
    test(...); // The black hole !!!

    template <typename X, typename Y>
    bool_value_type
    test(bool const&);

    template <typename X, typename Y>
    float_value_type
    test(float const&);
    
    template <typename X, typename Y>
    double_value_type
    test(double const&);

    template <typename X, typename Y>
    long_double_value_type
    test(long double const&);

    template <typename X, typename Y>
    char_value_type
    test(char const&);

    template <typename X, typename Y>
    schar_value_type
    test(signed char const&);

    template <typename X, typename Y>
    uchar_value_type
    test(unsigned char const&);

    template <typename X, typename Y>
    short_value_type
    test(short const&);

    template <typename X, typename Y>
    ushort_value_type
    test(unsigned short const&);

    template <typename X, typename Y>
    int_value_type
    test(int const&);

    template <typename X, typename Y>
    uint_value_type
    test(unsigned int const&);

    template <typename X, typename Y>
    long_value_type
    test(long const&);

    template <typename X, typename Y>
    ulong_value_type
    test(unsigned long const&);

    template <typename X, typename Y>
    typename boost::disable_if<
        is_basic<X>, x_value_type
    >::type
    test(X const&);

    template <typename X, typename Y>
    typename boost::disable_if<
        boost::mpl::or_<
            is_basic<Y>
          , boost::is_same<Y, asymmetric>
          , boost::is_same<const X, const Y>
        >
      , y_value_type
    >::type
    test(Y const&);

    template <typename X, typename Y>
    struct base_result_of
    {
        typedef typename boost::remove_cv<X>::type x_type;
        typedef typename boost::remove_cv<Y>::type y_type;

        typedef boost::mpl::vector16<
            boost::mpl::identity<bool>
          , boost::mpl::identity<float>
          , boost::mpl::identity<double>
          , boost::mpl::identity<long double>
          , boost::mpl::identity<char>
          , boost::mpl::identity<signed char>
          , boost::mpl::identity<unsigned char>
          , boost::mpl::identity<short>
          , boost::mpl::identity<unsigned short>
          , boost::mpl::identity<int>
          , boost::mpl::identity<unsigned int>
          , boost::mpl::identity<long>
          , boost::mpl::identity<unsigned long>
          , boost::mpl::identity<x_type>
          , boost::mpl::identity<y_type>
          , boost::mpl::identity<error_cant_deduce_type>
        >
        types;
    };

}} } // namespace shark::blas ::type_deduction_detail

#endif
