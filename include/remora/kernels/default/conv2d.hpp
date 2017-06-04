/*!
 *
 *
 * \brief       Implements the 2D convolution kernel for cpus
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

#ifndef REMORA_KERNELS_DEFAULT_Conv2D_HPP
#define REMORA_KERNELS_DEFAULT_Conv2D_HPP

#include "simd.hpp"
#include "../gemm.hpp"
#include <type_traits> //for std::common_type and aligned storage
namespace remora{namespace bindings {
	

/// \brief Transforms the given image into a row-major format for convolution 
///
/// The resulting matrix has one row for each output of the convolution.
/// each row contains the patch used for computing the result.	
template<class E1, class E2>
void im2mat(
	vector_expression<E1, cpu_tag> const& image,
	matrix_expression<E2, cpu_tag>& output,
	std::size_t num_channels,
	std::size_t image_height,
	std::size_t image_width,
	std::size_t filter_height,
	std::size_t filter_width
){
	//the order of loops is chosen such, that only very little changes of rows are performed
	for(std::size_t c = 0; c != num_channels; ++c){//iterate over the channels
		for(std::size_t i = 0; i != image_height - filter_height +1; ++i){// iterate over row-positions in the image
			for(std::size_t i1 = 0; i1 != filter_height; ++i1){//iterate over the the rows of the current filter
				for(std::size_t j = 0; j != image_width - filter_width +1; ++j){//iterate over the column-position in the image
					std::size_t row_start = i * (image_width - filter_width +1) +j;
					std::size_t image_start = c * image_width * image_height + (i+i1) * image_width + j;
					std::size_t col_start = c * filter_width * filter_height + i1 * filter_width;
					for(std::size_t j1 = 0; j1 != filter_width; ++j1){
						output()(row_start, col_start +j1) = image()(image_start + j1);
					}
				}
			}
		}
	}
}


template<class E1, class E2, class M>
void conv2d(
	vector_expression<E1, cpu_tag> const& image,
	vector_expression<E2, cpu_tag> const& filter,
	vector_expression<M, cpu_tag>& output,
	std::size_t num_channels,
	std::size_t num_filters,
	std::size_t image_height,
	std::size_t image_width,
	std::size_t filter_height,
	std::size_t filter_width
){
	typedef typename std::common_type<
		typename E1::value_type, typename E2::value_type, typename M::value_type
	>::type value_type;
	std::size_t output_rows_per_filter = (image_height  - filter_height +1) * (image_width - filter_width +1);
	std::size_t filter_size = filter_width * filter_height * num_channels;
	
	REMORA_SIZE_CHECK(output().size() == num_filters * output_rows_per_filter);
	REMORA_SIZE_CHECK(image().size() == num_channels * image_width * image_height);
	REMORA_SIZE_CHECK(filter().size() == num_filters * filter_size);
	
	//allocate storage and create temporary matrices
	boost::alignment::aligned_allocator<value_type,64> allocator;
	value_type* image_storage = allocator.allocate( output_rows_per_filter * filter_size);
	value_type* filter_storage = allocator.allocate(num_filters * filter_size);
	dense_matrix_adaptor<value_type, row_major, cpu_tag> image_transformed(image_storage,output_rows_per_filter, filter_size);
	dense_matrix_adaptor<value_type, row_major, cpu_tag> filter_transformed(filter_storage, num_filters, filter_size);
	dense_matrix_adaptor<value_type, row_major, cpu_tag> output_transformed(output(), num_filters, output_rows_per_filter);
	//copy image to temporary storage
	im2mat(image,image_transformed, num_channels, image_height, image_width, filter_height, filter_width);
	//copy filters to temporary storage
	for(std::size_t f = 0; f != num_filters; ++f){
		for(std::size_t i = 0; i != filter_size; ++i){
			filter_transformed(f,i) = filter()(f * filter_size + i);
		}
	}
	
	//do the computation
	kernels::gemm(filter_transformed, trans(image_transformed), output_transformed, value_type(1.0));
	
	//deallocate storage
	allocator.deallocate(image_storage,output_rows_per_filter * filter_size);
	allocator.deallocate(filter_storage, num_filters * filter_size);
}

}}

#endif
