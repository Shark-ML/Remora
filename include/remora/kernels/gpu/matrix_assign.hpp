/*!
 * \brief       Kernels for matrix-expression assignments on the gpu
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
#ifndef REMORA_KERNELS_CLBLAS_MATRIX_ASSIGN_HPP
#define REMORA_KERNELS_CLBLAS_MATRIX_ASSIGN_HPP

#include "../../expression_types.hpp"
#include "../../detail/traits.hpp"
#include <boost/compute/kernel.hpp>
#include <boost/compute/detail/meta_kernel.hpp>

namespace remora{namespace bindings{
	
	
//////////////////////////////////////////////////////
////Apply function elementwise to Matrix
/////////////////////////////////////////////////////

// Explicitly iterating row major
template<class F, class M, class Orientation>
void matrix_apply(
	matrix_expression<M, gpu_tag>& m, 
	F const& f,
	Orientation
){
	static_assert(std::is_base_of<dense_tag, typename M::storage_type::storage_tag>::value, "target must have dense storage for assignment");
	
	auto storage = m.raw_storage();
	typedef typename M::value_type value_type;
	boost::compute::detail::meta_kernel k("blas_matrix_apply_dense");
	std::string buffer = k.get_buffer_identifier<value_type>(storage.buffer);
	std::size_t lda_index = k.add_arg<std::size_t>("lda");
	std::size_t offset_index = k.add_arg<std::size_t>("offset");
	
	//create source
	if(std::is_same<Orientation, row_major>::value){
		auto elem=k.expr<value_type>(buffer+"[lda * get_global_id(0) + get_global_id(1) +offset]");
		k<<elem<<" = " << F()(elem)<<";";
	}else{
		auto elem=k.expr<value_type>(buffer+"[lda * get_global_id(1) + get_global_id(0) +offset]");
		k<<elem<<" = " << F()(elem)<<";";
	}
	boost::compute::kernel kernel = k.compile(m().queue().get_context());
	//enqueue kernel
	kernel.set_arg(lda_index, storage.leading_dimension);
	kernel.set_arg(offset_index, storage.offset);
	std::size_t global_work_size[2] = {m().size1(), m().size2()};
	m().queue().enqueue_nd_range_kernel(kernel, 2,nullptr, global_work_size, nullptr);
}
	
//////////////////////////////////////////////////////
////Scalar Assignment to Matrix
/////////////////////////////////////////////////////

// Explicitly iterating row major
template<class F, class M, class Orientation>
void matrix_assign(
	matrix_expression<M, gpu_tag>& m, 
	typename M::value_type t, 
	Orientation
){
	static_assert(std::is_base_of<dense_tag, typename M::storage_type::storage_tag>::value, "target must have dense storage for assignment");
	
	auto storage = m().raw_storage();
	typedef typename M::value_type value_type;
	boost::compute::detail::meta_kernel k("blas_matrix_assign_constant_dense");
	std::size_t t_index = k.add_arg<value_type>("t");
	std::string buffer = k.get_buffer_identifier<value_type>(storage.buffer);
	std::size_t lda_index = k.add_arg<std::size_t>("lda");
	std::size_t offset_index = k.add_arg<std::size_t>("offset");
	
	//create source
	if(std::is_same<Orientation, row_major>::value){
		auto elem=k.expr<value_type>(buffer+"[lda * get_global_id(0) + get_global_id(1) +offset]");
		k<<elem<<" = " << F()(elem,k.var<value_type>("t"))<<";";
	}else{
		auto elem=k.expr<value_type>(buffer+"[lda * get_global_id(1) + get_global_id(0) +offset]");
		k<<elem<<" = " << F()(elem,k.var<value_type>("t"))<<";";
	}
	boost::compute::kernel kernel = k.compile(m().queue().get_context());
	//enqueue kernel
	kernel.set_arg(t_index, t);
	kernel.set_arg(lda_index, storage.leading_dimension);
	kernel.set_arg(offset_index, storage.offset);
	std::size_t global_work_size[2] = {m().size1(), m().size2()};
	m().queue().enqueue_nd_range_kernel(kernel, 2,nullptr, global_work_size, nullptr);
}


///////////////////////////////////////////////////////////////////////////////////////////
//////Matrix Assignment With Functor implementing +=,-=...
///////////////////////////////////////////////////////////////////////////////////////////

template<class F, class M, class E>
void matrix_assign_functor_gpu_impl(
	matrix_expression<M, gpu_tag>& m, 
	matrix_expression<E, gpu_tag> const& e,
	F f,
	row_major, dense_tag
){
	auto storageM = m().raw_storage();
	auto storageE = e().raw_storage();
	typedef typename M::value_type value_type;
	boost::compute::detail::meta_kernel k("blas_matrix_assign_dense");
	std::string bufferM = k.get_buffer_identifier<value_type>(storageM.buffer);
	std::size_t ldM_index = k.add_arg<std::size_t>("ldM");
	std::size_t offsetM_index = k.add_arg<std::size_t>("offsetM");
	
	std::string bufferE = k.get_buffer_identifier<value_type>(storageE.buffer);
	std::size_t ldE_index = k.add_arg<std::size_t>("ldE");
	std::size_t offsetE_index = k.add_arg<std::size_t>("offsetE");
	
	//create source
	auto elemM=k.expr<value_type>(bufferM+"[ldM * get_global_id(0) + get_global_id(1) +offsetM]");
	auto elemE=k.expr<value_type>(bufferE+"[ldE * get_global_id(0) + get_global_id(1) +offsetE]");
	k<< elemM << " = " <<f(elemM,elemE)<<";\n";
	boost::compute::kernel kernel = k.compile(m().queue().get_context());
	//enqueue kernel
	kernel.set_arg(ldM_index, storageM.leading_dimension);
	kernel.set_arg(offsetM_index, storageM.offset);
	kernel.set_arg(ldE_index, storageE.leading_dimension);
	kernel.set_arg(offsetE_index, storageE.offset);
	std::size_t global_work_size[2] = {m().size1(), m().size2()};
	m().queue().enqueue_nd_range_kernel(kernel, 2,nullptr, global_work_size, nullptr);
}

template<class F, class M, class E>
void matrix_assign_functor_gpu_impl(
	matrix_expression<M, gpu_tag>& m, 
	matrix_expression<E, gpu_tag> const& e,
	F f,
	row_major, unknown_tag
){
	//create source
	auto storageM = m().raw_storage();
	typedef typename M::value_type value_type;
	boost::compute::detail::meta_kernel k("blas_matrix_assign");
	std::string bufferM = k.get_buffer_identifier<value_type>(storageM.buffer);
	std::size_t ldM_index = k.add_arg<std::size_t>("ldM");
	std::size_t offsetM_index = k.add_arg<std::size_t>("offsetM");
	
	auto exprRow=k.expr<cl_uint>("get_global_id(0)");
	auto exprCol=k.expr<cl_uint>("get_global_id(1)");
	auto elemM=k.expr<value_type>(bufferM+"[ldM * get_global_id(0) + get_global_id(1) + offsetM]");
	k<< elemM << "=" << f(elemM,e()(exprRow,exprCol))<<";\n";
	//enqueue kernel
	boost::compute::kernel kernel = k.compile(m().queue().get_context());
	kernel.set_arg(ldM_index, storageM.leading_dimension);
	kernel.set_arg(offsetM_index, storageM.offset);
	std::size_t global_work_size[2] = {m().size1(), m().size2()};
	m().queue().enqueue_nd_range_kernel(kernel, 2,nullptr, global_work_size, nullptr);
}


//dense-dense case row-major, column-major
template<class F,class M, class E>
void matrix_assign_functor_gpu_impl(
	matrix_expression<M, gpu_tag> &m, 
	matrix_expression<E, gpu_tag> const& e,
	F f,
	column_major,dense_tag
) {
	//Kernel is based on boost/compute/examples/matrix_transpose.cpp
	typedef typename M::value_type value_type;
	auto storageM = m().raw_storage();
	auto storageE = e().raw_storage();
	std::size_t TILE_DIM = 32;
	char const* options ="-DTILE_DIM=32ul";
	//There are usually not enough parallel worker threads in a local group
	//to fill a tile. Therefore every parallel threads reads several elements.
	//BLOCK_COLS are the number of parallel threads reading a column
	//and must be a divisor of TILE_DIM
	std::size_t BLOCK_COLS = 8; 
	
	//create source
	boost::compute::detail::meta_kernel k("blas_matrix_assign_row_col_dense");
	std::string bufferM = k.get_buffer_identifier<value_type>(storageM.buffer);
	std::size_t ldM_index = k.add_arg<std::size_t>("ldM");
	std::size_t offsetM_index = k.add_arg<std::size_t>("offsetM");
	
	std::string bufferE = k.get_buffer_identifier<value_type>(storageE.buffer);
	std::size_t ldE_index = k.add_arg<std::size_t>("ldE");
	std::size_t offsetE_index = k.add_arg<std::size_t>("offsetE");
	
	//create local memory. we first copy a tile in local
	// memory which gets the orientation right. Then we copy the tile
	//to the target
	// TILE_DIM+1 is here to avoid bank conflicts in local memory
	std::size_t size1_index = k.add_arg<std::size_t>("size1");
	std::size_t size2_index = k.add_arg<std::size_t>("size2");
	k << "__local " <<k.decl<value_type>("tile")<< "[TILE_DIM][TILE_DIM+2];\n";
	k << "uint base_row = get_group_id(0) * TILE_DIM;\n";
	k << "uint base_col = get_group_id(1) * TILE_DIM;\n";
	//copy indices, into local memory, note the change of direction
	//also note that if we are on the boundary, the tile
	// might be largerthan the amount of values to read
	k << "uint maxDim1 = min(size1-base_row,TILE_DIM);\n";
	k << "uint maxDim2 = min(size2-base_col,TILE_DIM);\n";
	k << "for(uint i = get_local_id(1) ; i < maxDim2 && get_local_id(0) < maxDim1; i += get_local_size(1)){\n";
	auto row_exp = k.expr<cl_uint>("(base_row+get_local_id(0))");
	auto col_exp = k.expr<cl_uint>("(base_col+i)");
	k << "    tile[get_local_id(0)][i] =" << bufferE<<"[ldE * (base_col+i) + base_row+get_local_id(0) + offsetE];\n";
	k << "}\n";
	k << "barrier(CLK_LOCAL_MEM_FENCE);\n";//wait until all threads are done with copying
	// write output from local memory, again be sure that 
	// we do not write outside the feasible area
	k << "for(uint i = get_local_id(1); i < maxDim1 && get_local_id(0) < maxDim2; i += get_local_size(1)){\n"; 
	auto target = k.expr<cl_uint>(bufferM+"[ldM * (base_row+i) + base_col+get_local_id(0) + offsetM]");
	k << target << " = " <<f(target, k.expr<cl_uint>("tile[i][get_local_id(0)]"))<<";\n";
	k << "}\n";
	
	//compile kernel
	
	boost::compute::kernel kernel = k.compile(m().queue().get_context(), options);
	
	//enqueue kernel
	kernel.set_arg(ldM_index, storageM.leading_dimension);
	kernel.set_arg(offsetM_index, storageM.offset);
	kernel.set_arg(ldE_index, storageE.leading_dimension);
	kernel.set_arg(offsetE_index, storageE.offset);
	kernel.set_arg(size1_index, m().size1());
	kernel.set_arg(size2_index, m().size2());
	std::size_t global_work_size[2] = {(m().size1()+TILE_DIM-1) / TILE_DIM * TILE_DIM, (m().size2()+TILE_DIM-1) / TILE_DIM * BLOCK_COLS };
	std::size_t local_work_size[2] = {TILE_DIM, BLOCK_COLS};
	m().queue().enqueue_nd_range_kernel(kernel, 2,nullptr, global_work_size, local_work_size);
}

//dense-dense case row-major, column-major
template<class F,class M, class E>
void matrix_assign_functor_gpu_impl(
	matrix_expression<M, gpu_tag> &m, 
	matrix_expression<E, gpu_tag> const& e,
	F f,
	column_major,unknown_tag
) {
	//Kernel is based on boost/compute/examples/matrix_transpose.cpp
	typedef typename M::value_type value_type;
	auto storageM = m().raw_storage();
	std::size_t TILE_DIM = 32;
	char const* options ="-DTILE_DIM=32ul";
	//There are usually not enough parallel worker threads in a local group
	//to fill a tile. Therefore every parallel threads reads several elements.
	//BLOCK_COLS are the number of parallel threads reading a column
	//and must be a divisor of TILE_DIM
	std::size_t BLOCK_COLS = 8; 
	
	
	//create source
	boost::compute::detail::meta_kernel k("blas_matrix_assign_row_col");
	//create local memory. we first copy a tile in local
	// memory which gets the orientation right. Then we copy the tile
	//to the target
	// TILE_DIM+1 is here to avoid bank conflicts in local memory
	std::size_t size1_index = k.add_arg<std::size_t>("size1");
	std::size_t size2_index = k.add_arg<std::size_t>("size2");
	k << "__local " <<k.decl<value_type>("tile")<< "[TILE_DIM][TILE_DIM+2];\n";
	k << "uint base_row = get_group_id(0) * TILE_DIM;\n";
	k << "uint base_col = get_group_id(1) * TILE_DIM;\n";
	//copy indices, into local memory, note the change of direction
	//also note that if we are on the boundary, the tile
	// might be largerthan the amount of values to read
	k << "uint maxDim1 = min(size1-base_row,TILE_DIM);\n";
	k << "uint maxDim2 = min(size2-base_col,TILE_DIM);\n";
	k << "for(uint i = get_local_id(1) ; i < maxDim2 && get_local_id(0) < maxDim1; i += get_local_size(1)){\n";
	auto row_exp = k.expr<cl_uint>("(base_row+get_local_id(0))");
	auto col_exp = k.expr<cl_uint>("(base_col+i)");
	k << "    tile[get_local_id(0)][i] =" << e()(row_exp, col_exp)<<";\n";
	k << "}\n";
	k << "barrier(CLK_LOCAL_MEM_FENCE);\n";//wait until all threads are done with copying
	// write output from local memory, again be sure that 
	// we do not write outside the feasible area
	k << "for(uint i = get_local_id(1); i < maxDim1 && get_local_id(0) < maxDim2; i += get_local_size(1)){\n"; 
	auto target = m()(k.expr<cl_uint>("(base_row + i)"), k.expr<cl_uint>("(base_col + get_local_id(0))"));
	k << target << " = " <<f(target, k.expr<cl_uint>("tile[i][get_local_id(0)]"))<<";\n";
	k << "}\n";
	
	//compile kernel
	
	boost::compute::kernel kernel = k.compile(m().queue().get_context(), options);
	
	//enqueue kernel
	kernel.set_arg(size1_index, m().size1());
	kernel.set_arg(size2_index, m().size2());
	std::size_t global_work_size[2] = {(m().size1()+TILE_DIM-1) / TILE_DIM * TILE_DIM, (m().size2()+TILE_DIM-1) / TILE_DIM * BLOCK_COLS };
	std::size_t local_work_size[2] = {TILE_DIM, BLOCK_COLS};
	m().queue().enqueue_nd_range_kernel(kernel, 2,nullptr, global_work_size, local_work_size);
}

//dense-dense case row-major, row-major
template<class F, class M, class E, class Orientation>
void matrix_assign_functor(
	matrix_expression<M, gpu_tag> &m, 
	matrix_expression<E, gpu_tag> const& e,
	F f,
	row_major, Orientation,dense_tag, dense_tag
) {
	static_assert(std::is_base_of<dense_tag, typename M::storage_type::storage_tag>::value, "target must have dense storage for assignment");
	//Dispatch based on whether the matrices themselves are densly stored
	matrix_assign_functor_gpu_impl(m, e, f, Orientation(), typename E::storage_type::storage_tag());
}

/////////////////////////////////////////////////////////////////
//////Matrix Assignment implementing op=
////////////////////////////////////////////////////////////////

//implement by using the assigner function below and call the functions above

namespace detail {
struct assigner{
	template<class Arg1, class Arg2>
	Arg2 operator()(Arg1 const&, Arg2 const& y) const{
		return y;
	}
};
}

template<class M, class E>
void matrix_assign(
	matrix_expression<M, gpu_tag> &m, 
	matrix_expression<E, gpu_tag> const& e,
	row_major o, row_major,dense_tag t, dense_tag
) {
	matrix_assign_functor(m, e, detail::assigner(), o, o, t, t);
}

//dense-dense case
template<class M, class E>
void matrix_assign(
	matrix_expression<M, gpu_tag> &m, 
	matrix_expression<E, gpu_tag> const& e,
	row_major o1, column_major o2,dense_tag t, dense_tag
) {
	matrix_assign_functor(m, e, detail::assigner(), o1, o2, t, t);
}


}}

#endif
