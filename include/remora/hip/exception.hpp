//===========================================================================
/*!
 * 
 *
 * \brief       Error handling for the HIP runtime
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

#ifndef REMORA_HIP_EXCEPTION_HPP
#define REMORA_HIP_EXCEPTION_HPP

#include <exception>
#include <system_error>
#include <hip/hip_runtime.h>
//#include <hipblas.h>
namespace remora{namespace hip{
	
class hip_error_category: public std::error_category{
public:
	const char* name() const noexcept{
		return "HIP";
	}
	std::string message( int error ) const{
		return hipGetErrorString(static_cast<hipError_t>(error));
	}
	static hip_error_category& category(){
		static hip_error_category cat;
		return cat;
	}
};

//~ class hipblas_error_category: public std::error_category{
//~ public:
	//~ const char* name() const noexcept{
		//~ return "hipBLAS";
	//~ }
	//~ std::string message( int condition ) const{
		//~ switch(condition){
			//~ case HIPBLAS_STATUS_SUCCESS:
				//~ return "Success";
			//~ case HIPBLAS_STATUS_NOT_INITIALIZED:
				//~ return "HIPBLAS library not initialized";
			//~ case HIPBLAS_STATUS_ALLOC_FAILED:
				//~ return "Resource allocation failed";
			//~ case HIPBLAS_STATUS_INVALID_VALUE: 
				//~ return "Unsupported numerical value was passed to function";
			//~ case HIPBLAS_STATUS_MAPPING_ERROR:
				//~ return "Access to OPENCL memory space failed";
			//~ case HIPBLAS_STATUS_EXECUTION_FAILED:
				//~ return "OPENCL program failed to execute";
			//~ case HIPBLAS_STATUS_INTERNAL_ERROR:
				//~ return "An internal HIPBLAS operation failed";
			//~ case HIPBLAS_STATUS_NOT_SUPPORTED:
				//~ return "Function not implemented";
			//~ case HIPBLAS_STATUS_ARCH_MISMATCH:
				//~ return "Arch mismatch";
			//~ default:
				//~ return "Unknown error code: "+std::to_string(condition);
		//~ }
	//~ }
	//~ static hipblas_error_category&() category(){
		//~ static hipblas_error_category cat;
		//~ return cat;
	//~ }
//~ };

class hip_exception:public std::system_error{
public:
	hip_exception(hipError_t code): std::system_error(code, hip_error_category::category()){}
};
//~ class hipblas_exception:public std::system_error{
//~ public:
	//~ hipblas_exception(hipblasStatus_t code): std::system_error(code, hipblas_error_category::category()){}
//~ };


inline void check_hip(hipError_t code){
	if(code != hipSuccess)
		throw hip_exception(code);
}

//~ inline void check_hipblas(hipblasStatus_t code){
	//~ if(code != HIPBLAS_STATUS_SUCCESS)
		//~ throw hipblas_exception(code);
//~ }

}}
#endif