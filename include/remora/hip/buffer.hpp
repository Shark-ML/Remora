//===========================================================================
/*!
 * 
 *
 * \brief       Helper for allocating memory on the Device
 *
 * \author      O. Krause
 * \date        2018
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

#ifndef REMORA_HIP_BUFFER_HPP
#define REMORA_HIP_BUFFER_HPP

#include "exception.hpp"
#include "device.hpp"
namespace remora{namespace hip{
	

template<class T>
class buffer{
public:
	buffer(std::size_t num_elems, device& device)
	:m_ptr(nullptr), m_size(num_elems), m_device(&device){
		m_device->set_device();
		check_hip(hipMalloc(&m_ptr, num_elems * sizeof(T)));
	}
	buffer(buffer && other){
		m_device = other.m_device;
		check_hip(hipFree(m_ptr));
		m_size = other.size();
		m_ptr = other.m_ptr;
		other.m_ptr = nullptr;
	}
	buffer(buffer const& other){
		m_size = other.m_size;
		m_device = other.m_device;
		m_device->set_device();
		check_hip(hipMalloc(&m_ptr, m_size * sizeof(T)));
		check_hip(hipMemcpy(m_ptr, other.m_ptr, other.m_size, hipMemcpyDeviceToDevice)); 
	}
	
	buffer operator = (buffer && other){
		m_device = other.m_device;
		check_hip(hipFree(m_ptr));
		m_size = other.size();
		m_ptr = other.m_ptr;
		other.m_ptr = nullptr;
		return *this;
	}
	buffer& operator = (buffer const& other){
		*this = buffer<T>(other);
		return *this;
	}
	
	
	__host__ __device__ hip::device& device() const {
		return *m_device;
	}
	__host__ __device__ T* get(){
		return m_ptr;
	}
	__host__ __device__ T const* get() const{
		return m_ptr;
	}
	__host__ __device__ std::size_t size()const{
		return m_size;
	}
	void resize(std::size_t newSize){
		if(m_size != newSize){
			*this = buffer<T>(newSize, *m_device);
		}
	}
	
	~buffer(){
		check_hip(hipFree(m_ptr));
	}
private:
	T* m_ptr;
	std::size_t m_size;
	hip::device* m_device;
};


template<class T>
__global__ void fill_buffer_kernel(hipLaunchParm lp, T* buffer, T value, std::size_t size, std::size_t stride_elem){
	std::size_t offset = (hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x) * stride_elem;
	std::size_t stride = hipBlockDim_x * hipGridDim_x * stride_elem;

	for (std::size_t i = offset; i < size; i += stride) {
		buffer[i] = value;
	}
}
template<class T>
__global__ void fill_buffer2d_kernel(hipLaunchParm lp, T* buffer, T value, std::size_t size1, std::size_t size2, std::size_t stride1, std::size_t stride2){
	std::size_t row = (hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x);
	if(row >= size1) return;
	
	std::size_t column = (hipBlockIdx_y * hipBlockDim_y + hipThreadIdx_y);
	std::size_t offset = row * stride1 + column * stride2;
	std::size_t stride = hipBlockDim_y * hipGridDim_y * stride2;
	for (std::size_t i = offset; i < size2; i += stride) {
		buffer[i] = value;
	}
}
template<class T>
void fill_buffer(T* buffer, T value, std::size_t size, std::size_t stride_elem, device& device){
	device.set_device();
	std::size_t blockSize = 8 * device.warp_size();
	std::size_t numBlocks = (size + blockSize - 1)/ blockSize;
	hipLaunchKernel(
		fill_buffer_kernel, 
		dim3(numBlocks), dim3(blockSize), 0, get_stream(device).handle(), 
		buffer, value, size, stride_elem
	);
}
template<class T>
void fill_buffer2d(T* buffer, T value, std::size_t size1, std::size_t size2, std::size_t stride1, std::size_t stride2, device& device){
	device.set_device();
	std::size_t blockSize1 = 4;
	std::size_t blockSize2 = device.warp_size()/4;
	std::size_t numBlocks1 = (size1 + blockSize1 - 1) / blockSize1;
	std::size_t numBlocks2 = 1;
	hipLaunchKernel(
		fill_buffer2d_kernel, 
		dim3(numBlocks1, numBlocks2), dim3(blockSize1,blockSize2), 0, get_stream(device).handle(), 
		buffer, value, size1, size2, stride1, stride2
	);
}
}}
#endif