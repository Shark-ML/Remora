#define BOOST_TEST_MODULE Remora_VectorAssign
#include <boost/test/unit_test.hpp>
#include <boost/test/floating_point_comparison.hpp>

#include <remora/kernels/matrix_assign.hpp>
//~ #include <remora/triangular_matrix.hpp>
#include <remora/dense.hpp>
#include <remora/sparse.hpp>
#include <remora/io.hpp>
#include <remora/proxy_expressions.hpp>
#include <iostream>

using namespace remora;

template<class M1, class M2>
void checkMatrixEqual(M1 const& m1, M2 const& m2){
	BOOST_REQUIRE_EQUAL(m1.size1(),m2.size1());
	BOOST_REQUIRE_EQUAL(m1.size2(),m2.size2());
	for(std::size_t i = 0; i != m2.size1(); ++i){
		for(std::size_t j = 0; j != m2.size2(); ++j){
			BOOST_CHECK_EQUAL(m1(i,j),m2(i,j));
		}
	}
}

template<class M1, class M2>
void checkSparseMatrixEqual(M1 const& m1, M2 const& m2){
	BOOST_REQUIRE_EQUAL(m1.size1(),m2.size1());
	BOOST_REQUIRE_EQUAL(m1.size2(),m2.size2());
	matrix<typename M1::value_type, typename M1::orientation> m1t(m2.size1(),m2.size2(),0);
	matrix<typename M1::value_type, typename M2::orientation> m2t(m2.size1(),m2.size2(),0);
	typedef typename M1::orientation orientation1;
	typedef typename M2::orientation orientation2;
	for(std::size_t i = 0; i != major_size(m1); ++i){
		for(auto pos = m1.major_begin(i); pos != m1.major_end(i); ++pos){
			m1t(orientation1::index_M(i,pos.index()),orientation1::index_m(i,pos.index())) = *pos;
		}
	}
	for(std::size_t i = 0; i != major_size(m2); ++i){
		for(auto pos = m2.major_begin(i); pos != m2.major_end(i); ++pos){
			m2t(orientation2::index_M(i,pos.index()),orientation2::index_m(i,pos.index())) = *pos;
		}			
	}
	checkMatrixEqual(m1t,m2t);
}


BOOST_AUTO_TEST_SUITE (Remora_matrix_assign)


///////////////////////////////////////////////////////////////////////////////
//////FUNCTOR CONSTANT ASSIGNMENT
//////////////////////////////////////////////////////////////////////////////

//test for assign of the form m_ij=f(m_ij,t) for constant t
BOOST_AUTO_TEST_CASE( Remora_Dense_Matrix_Constant_Functor_Assign ){
	std::cout<<"testing dense functor-constant assignment."<<std::endl;
	matrix<unsigned int,row_major> input_row_major(10,20);
	matrix<unsigned int,row_major> input_column_major(10,20);
	matrix<unsigned int,row_major> target(10,20);
	typedef device_traits<cpu_tag>::multiply<unsigned int> functor;
	const unsigned int t = 2;
	for(std::size_t i = 0; i != 10; ++i){
		for(std::size_t j = 0; j != 20; ++j){
			input_row_major(i,j) = 2*(20*i+1)+1.0;
			input_column_major(i,j) = 2*(20*i+1)+1.0;
			target(i,j)=t*input_row_major(i,j);
		}
	}

	std::cout<<"testing row-major"<<std::endl;
	kernels::assign<functor> (input_row_major, t);
	checkMatrixEqual(target,input_row_major);
	
	std::cout<<"testing column-major"<<std::endl;
	kernels::assign<functor> (input_column_major, t);
	checkMatrixEqual(target,input_column_major);
	
	std::cout<<"\n";
}

//~ BOOST_AUTO_TEST_CASE( Remora_Packed_Matrix_Constant_Functor_Assign ){
	//~ std::cout<<"testing packed functor-constant assignment."<<std::endl;
	//~ triangular_matrix<unsigned int,row_major,lower> input_row_lower(20);
	//~ triangular_matrix<unsigned int,row_major,upper> input_row_upper(20);
	//~ triangular_matrix<unsigned int,column_major,lower> input_column_lower(20);
	//~ triangular_matrix<unsigned int,column_major,upper> input_column_upper(20);
	//~ triangular_matrix<unsigned int,row_major,lower> target_lower(20);
	//~ triangular_matrix<unsigned int,row_major,upper> target_upper(20);
	//~ typedef device_traits<cpu_tag>::multiply<unsigned int> functor;
	//~ const unsigned int t = 2;
	//~ for(unsigned int i = 0; i != 20; ++i){
		//~ for(unsigned int j = 0; j <= i; ++j){
			//~ input_row_lower.set_element(i,j, 2*(20*i+1)+1);
			//~ input_column_lower.set_element(i,j, 2*(20*i+1)+1);
			//~ target_lower.set_element(i,j,t*input_row_lower(i,j));
		//~ }
	//~ }
	
	//~ for(unsigned int i = 0; i != 20; ++i){
		//~ for(unsigned int j = i; j != 20; ++j){
			//~ input_row_upper.set_element(i,j,2*(20*i+1)+1);
			//~ input_column_upper.set_element(i,j,2*(20*i+1)+1);
			//~ target_upper.set_element(i,j,t*input_row_upper(i,j));
		//~ }
	//~ }

	//~ std::cout<<"testing row-major lower"<<std::endl;
	//~ kernels::assign<functor> (input_row_lower, t);
	//~ checkMatrixEqual(target_lower,input_row_lower);
	
	//~ std::cout<<"testing column-major lower"<<std::endl;
	//~ kernels::assign<functor> (input_column_lower, t);
	//~ checkMatrixEqual(target_lower,input_column_lower);
	
	//~ std::cout<<"testing row-major upper"<<std::endl;
	//~ kernels::assign<functor> (input_row_upper, t);
	//~ checkMatrixEqual(target_upper,input_row_upper);
	
	//~ std::cout<<"testing column-major upper"<<std::endl;
	//~ kernels::assign<functor> (input_column_upper, t);
	//~ checkMatrixEqual(target_upper,input_column_upper);
	
	//~ std::cout<<"\n";
//~ }

BOOST_AUTO_TEST_CASE( Remora_Sparse_Matrix_Constant_Functor_Assign ){
	std::cout<<"testing sparse functor-constant assignment."<<std::endl;
	compressed_matrix<unsigned int> input_row_major(10,20,0);
	compressed_matrix<unsigned int,std::size_t, column_major> input_column_major(20,10,0);
	matrix<unsigned int> result_row_major(10,20,0);
	matrix<unsigned int> result_column_major(20,10,0);
	typedef device_traits<cpu_tag>::multiply<unsigned int> functor;
	const unsigned int t = 2;
	for(unsigned int i = 0; i != 10; ++i){
		auto in_row_pos = input_row_major.major_begin(i);
		auto in_col_pos = input_column_major.major_begin(i);
		for(unsigned int j = 1; j < 20; j += (i + 1)){
			in_row_pos = input_row_major.set_element(in_row_pos, j, 2*(20*i+1)+j);
			in_col_pos = input_column_major.set_element(in_col_pos, j, 3*(20*i+1)+j);
			result_row_major(i,j) = t*(2*(20*i+1)+j);
			result_column_major(j,i) = t*(3*(20*i+1)+j);
		}
	}

	std::cout<<"testing row-major"<<std::endl;
	kernels::assign<functor> (input_row_major, t);
	checkSparseMatrixEqual(result_row_major, input_row_major);
	
	std::cout<<"testing column-major"<<std::endl;
	kernels::assign<functor> (input_column_major, t);
	checkSparseMatrixEqual(result_column_major, input_column_major);
	
	std::cout<<"\n";
}

//////////////////////////////////////////////////////
//////SIMPLE ASSIGNMENT
//////////////////////////////////////////////////////

BOOST_AUTO_TEST_CASE( Remora_Dense_Dense_Matrix_Assign ){
	std::cout<<"testing direct dense-dense assignment"<<std::endl;
	matrix<unsigned int,row_major> source_row_major(10,20);
	matrix<unsigned int,column_major> source_column_major(10,20);

	for(unsigned int i = 0; i != 10; ++i){
		for(unsigned int j = 0; j != 20; ++j){
			source_row_major(i,j) = 2*(20*i+1)+1;
			source_column_major(i,j) =  source_row_major(i,j)+2;
		}
	}

	//test all 4 combinations of row/column major
	{
		matrix<unsigned int,row_major> target(10,20);
		for(unsigned int i = 0; i != 10; ++i){
			for(unsigned int j = 0; j != 20; ++j){
				target(i,j) = 3*(20*i+1)+2;
			}
		}
		std::cout<<"testing row-row"<<std::endl;
		kernels::assign(target,source_row_major);
		checkMatrixEqual(target,source_row_major);
	}
	
	{
		matrix<unsigned int,row_major> target(10,20);
		for(unsigned int i = 0; i != 10; ++i){
			for(unsigned int j = 0; j != 20; ++j){
				target(i,j) = 3*(20*i+1)+2;
			}
		}
		std::cout<<"testing row-column"<<std::endl;
		kernels::assign(target,source_column_major);
		checkMatrixEqual(target,source_column_major);
	}
	
	{
		matrix<unsigned int,column_major> target(10,20);
		for(unsigned int i = 0; i != 10; ++i){
			for(unsigned int j = 0; j != 20; ++j){
				target(i,j) = 3*(20*i+1)+2;
			}
		}
		std::cout<<"testing column-row"<<std::endl;
		kernels::assign(target,source_row_major);
		checkMatrixEqual(target,source_row_major);
	}
	
	{
		matrix<unsigned int,column_major> target(10,20);
		for(unsigned int i = 0; i != 10; ++i){
			for(unsigned int j = 0; j != 20; ++j){
				target(i,j) = 3*(20*i+1)+2;
			}
		}
		std::cout<<"testing column-column"<<std::endl;
		kernels::assign(target,source_column_major);
		checkMatrixEqual(target,source_column_major);
	}
}

//not implemented yet!
//~ //Dense-Packed
//~ BOOST_AUTO_TEST_CASE( Remora_Dense_Packed_Matrix_Assign ){
	//~ std::cout<<"testing direct dense-packed assignment"<<std::endl;
	
	//~ //create the 4 different source matrices
	//~ typedef triangular_matrix<unsigned int,row_major,upper> MRU;
	//~ typedef triangular_matrix<unsigned int,column_major,upper> MCU;
	//~ typedef triangular_matrix<unsigned int,row_major,lower> MRL;
	//~ typedef triangular_matrix<unsigned int,column_major,lower> MCL;
	//~ MRU source_upper_row_major(20);
	//~ MRL  source_lower_row_major(20);
	//~ MCU source_upper_column_major(20);
	//~ MCL source_lower_column_major(20);

	//~ for(std::size_t i = 0; i != 20; ++i){
		//~ MRU::row_iterator pos1=source_upper_row_major.row_begin(i);
		//~ MRL::row_iterator pos2=source_lower_row_major.row_begin(i);
		//~ MCU::column_iterator pos3=source_upper_column_major.column_begin(i);
		//~ MCL::column_iterator pos4=source_lower_column_major.column_begin(i);
		//~ for(; pos1 != source_upper_row_major.row_end(i);++pos1){
			//~ *pos1 = i*20+pos1.index()+1;
		//~ }
		//~ for(; pos2 != source_lower_row_major.row_end(i);++pos2){
			//~ *pos2 = i*20+pos2.index()+1;
		//~ }
		//~ for(; pos3 != source_upper_column_major.column_end(i);++pos3){
			//~ *pos3 = i*20+pos3.index()+1;
		//~ }
		//~ for(; pos4 != source_lower_column_major.column_end(i);++pos4){
			//~ *pos4 = i*20+pos4.index()+1;
		//~ }
	//~ }

	//~ //test all 8 combinations of row/column major  target and the four sources
	//~ {
		//~ matrix<unsigned int,row_major> target(20,20);
		//~ for(std::size_t i = 0; i != 20; ++i){
			//~ for(std::size_t j = 0; j != 20; ++j){
				//~ target(i,j) = 3*(20*i+1)+2;
			//~ }
		//~ }
		//~ std::cout<<"testing row-row/upper"<<std::endl;
		//~ kernels::assign(target,source_upper_row_major);
		//~ checkMatrixEqual(target,source_upper_row_major);
	//~ }
	//~ {
		//~ matrix<unsigned int,row_major> target(20,20);
		//~ for(std::size_t i = 0; i != 20; ++i){
			//~ for(std::size_t j = 0; j != 20; ++j){
				//~ target(i,j) = 3*(20*i+1)+2;
			//~ }
		//~ }
		//~ std::cout<<"testing row-row/lower"<<std::endl;
		//~ kernels::assign(target,source_lower_row_major);
		//~ checkMatrixEqual(target,source_lower_row_major);
	//~ }
	
	//~ {
		//~ matrix<unsigned int,row_major> target(20,20);
		//~ for(std::size_t i = 0; i != 20; ++i){
			//~ for(std::size_t j = 0; j != 20; ++j){
				//~ target(i,j) = 3*(20*i+1)+2;
			//~ }
		//~ }
		//~ std::cout<<"testing row-column/upper"<<std::endl;
		//~ kernels::assign(target,source_upper_column_major);
		//~ checkMatrixEqual(target,source_upper_column_major);
	//~ }
	//~ {
		//~ matrix<unsigned int,row_major> target(20,20);
		//~ for(std::size_t i = 0; i != 20; ++i){
			//~ for(std::size_t j = 0; j != 20; ++j){
				//~ target(i,j) = 3*(20*i+1)+2;
			//~ }
		//~ }
		//~ std::cout<<"testing row-column/lower"<<std::endl;
		//~ kernels::assign(target,source_lower_column_major);
		//~ checkMatrixEqual(target,source_lower_column_major);
	//~ }
	
	//~ {
		//~ matrix<unsigned int,column_major> target(20,20);
		//~ for(std::size_t i = 0; i != 20; ++i){
			//~ for(std::size_t j = 0; j != 20; ++j){
				//~ target(i,j) = 3*(20*i+1)+2;
			//~ }
		//~ }
		//~ std::cout<<"testing column-row/upper"<<std::endl;
		//~ kernels::assign(target,source_upper_row_major);
		//~ checkMatrixEqual(target,source_upper_row_major);
	//~ }
	//~ {
		//~ matrix<unsigned int,column_major> target(20,20);
		//~ for(std::size_t i = 0; i != 20; ++i){
			//~ for(std::size_t j = 0; j != 20; ++j){
				//~ target(i,j) = 3*(20*i+1)+2;
			//~ }
		//~ }
		//~ std::cout<<"testing column-row/lower"<<std::endl;
		//~ kernels::assign(target,source_lower_row_major);
		//~ checkMatrixEqual(target,source_lower_row_major);
	//~ }
	
	//~ {
		//~ matrix<unsigned int,column_major> target(20,20);
		//~ for(std::size_t i = 0; i != 20; ++i){
			//~ for(std::size_t j = 0; j != 20; ++j){
				//~ target(i,j) = 3*(20*i+1)+2;
			//~ }
		//~ }
		//~ std::cout<<"testing column-column/upper"<<std::endl;
		//~ kernels::assign(target,source_upper_column_major);
		//~ checkMatrixEqual(target,source_upper_column_major);
	//~ }
	//~ {
		//~ matrix<unsigned int,column_major> target(20,20);
		//~ for(std::size_t i = 0; i != 20; ++i){
			//~ for(std::size_t j = 0; j != 20; ++j){
				//~ target(i,j) = 3*(20*i+1)+2;
			//~ }
		//~ }
		//~ std::cout<<"testing column-column/lower"<<std::endl;
		//~ kernels::assign(target,source_lower_column_major);
		//~ checkMatrixEqual(target,source_lower_column_major);
	//~ }
//~ }


//~ BOOST_AUTO_TEST_CASE( Remora_Packed_Packed_Matrix_Assign ){
	//~ std::cout<<"testing direct packed-packed assignment"<<std::endl;
	
	//~ //create the 4 different source matrices
	//~ typedef triangular_matrix<unsigned int,row_major,upper> MRU;
	//~ typedef triangular_matrix<unsigned int,column_major,upper> MCU;
	//~ typedef triangular_matrix<unsigned int,row_major,lower> MRL;
	//~ typedef triangular_matrix<unsigned int,column_major,lower> MCL;
	//~ MRU source_upper_row_major(20);
	//~ MRL  source_lower_row_major(20);
	//~ MCU source_upper_column_major(20);
	//~ MCL source_lower_column_major(20);

	//~ for(unsigned int i = 0; i != 20; ++i){
		//~ MRU::row_iterator pos1=source_upper_row_major.row_begin(i);
		//~ MRL::row_iterator pos2=source_lower_row_major.row_begin(i);
		//~ MCU::column_iterator pos3=source_upper_column_major.column_begin(i);
		//~ MCL::column_iterator pos4=source_lower_column_major.column_begin(i);
		//~ for(; pos1 != source_upper_row_major.row_end(i);++pos1){
			//~ *pos1 = i*20+pos1.index()+1;
		//~ }
		//~ for(; pos2 != source_lower_row_major.row_end(i);++pos2){
			//~ *pos2 = i*20+pos2.index()+1;
		//~ }
		//~ for(; pos3 != source_upper_column_major.column_end(i);++pos3){
			//~ *pos3 = i*20+pos3.index()+1;
		//~ }
		//~ for(; pos4 != source_lower_column_major.column_end(i);++pos4){
			//~ *pos4 = i*20+pos4.index()+1;
		//~ }
	//~ }

	//~ //test all 8 combinations of row/column major  target and the four sources
	//~ //for simplicitely we just assign the targets to be 1...
	//~ {
		//~ MRU target(20,1);
		//~ std::cout<<"testing row-row/upper"<<std::endl;
		//~ kernels::assign(target,source_upper_row_major);
		//~ checkMatrixEqual(target,source_upper_row_major);
	//~ }
	//~ {
		//~ MRL target(20,1);
		//~ std::cout<<"testing row-row/lower"<<std::endl;
		//~ kernels::assign(target,source_lower_row_major);
		//~ checkMatrixEqual(target,source_lower_row_major);
	//~ }
	//~ {
		//~ MRU target(20,1);
		//~ std::cout<<"testing row-column/upper"<<std::endl;
		//~ kernels::assign(target,source_upper_column_major);
		//~ checkMatrixEqual(target,source_upper_column_major);
	//~ }
	//~ {
		//~ MRL target(20,1);
		//~ std::cout<<"testing row-column/lower"<<std::endl;
		//~ kernels::assign(target,source_lower_column_major);
		//~ checkMatrixEqual(target,source_lower_column_major);
	//~ }
	//~ {
		//~ MCU target(20,1);
		//~ std::cout<<"testing column-row/upper"<<std::endl;
		//~ kernels::assign(target,source_upper_row_major);
		//~ checkMatrixEqual(target,source_upper_row_major);
	//~ }
	//~ {
		//~ MCL target(20,1);
		//~ std::cout<<"testing column-row/lower"<<std::endl;
		//~ kernels::assign(target,source_lower_row_major);
		//~ checkMatrixEqual(target,source_lower_row_major);
	//~ }
	//~ {
		//~ MCU target(20,1);
		//~ std::cout<<"testing column-column/upper"<<std::endl;
		//~ kernels::assign(target,source_upper_column_major);
		//~ checkMatrixEqual(target,source_upper_column_major);
	//~ }
	//~ {
		//~ MCL target(20,1);
		//~ std::cout<<"testing column-column/lower"<<std::endl;
		//~ kernels::assign(target,source_lower_column_major);
		//~ checkMatrixEqual(target,source_lower_column_major);
	//~ }
//~ }


BOOST_AUTO_TEST_CASE( Remora_Dense_Sparse_Matrix_Assign ){
	std::cout<<"\ntesting direct dense-sparse assignment"<<std::endl;
	compressed_matrix<unsigned int> source_row_major(10,20);
	for(unsigned int i = 0; i != 10; ++i){	
		auto in_row_pos = source_row_major.major_begin(i);
		for(unsigned int j = 1; j < 20; j += (i + 1)){
			in_row_pos = source_row_major.set_element(in_row_pos, j,2*(20*i+1)+1);
		}
	}
	compressed_matrix<unsigned int, std::size_t, column_major> source_column_major = source_row_major;
	
	//test all 4 combinations of row/column major
	{
		matrix<unsigned int,row_major> target(10,20);
		for(unsigned int i = 0; i != 10; ++i){
			for(unsigned int j = 0; j != 20; ++j){
				target(i,j) = 3*(20*i+1)+2;
			}
		}
		std::cout<<"testing row-row"<<std::endl;
		kernels::assign(target,source_row_major);
		checkSparseMatrixEqual(target,source_row_major);
	}
	
	{
		matrix<unsigned int,row_major> target(10,20);
		for(unsigned int i = 0; i != 10; ++i){
			for(unsigned int j = 0; j != 20; ++j){
				target(i,j) = 3*(20*i+1)+2;
			}
		}
		std::cout<<"testing row-column"<<std::endl;
		kernels::assign(target,source_column_major);
		checkSparseMatrixEqual(target,source_column_major);
	}
	
	{
		matrix<unsigned int,column_major> target(10,20);
		for(unsigned int i = 0; i != 10; ++i){
			for(unsigned int j = 0; j != 20; ++j){
				target(i,j) = 3*(20*i+1)+2;
			}
		}
		std::cout<<"testing column-row"<<std::endl;
		kernels::assign(target,source_row_major);
		checkSparseMatrixEqual(target,source_row_major);
	}
	
	{
		matrix<unsigned int,column_major> target(10,20);
		for(unsigned int i = 0; i != 10; ++i){
			for(unsigned int j = 0; j != 20; ++j){
				target(i,j) = 3*(20*i+1)+2;
			}
		}
		std::cout<<"testing column-column"<<std::endl;
		kernels::assign(target,source_column_major);
		checkSparseMatrixEqual(target,source_column_major);
	}
	
}

BOOST_AUTO_TEST_CASE( Remora_Sparse_Sparse_Matrix_Assign ){
	std::cout<<"\ntesting direct sparse-sparse assignment"<<std::endl;
	compressed_matrix<unsigned int, std::size_t, row_major> source_row_major(10,20,0);
	compressed_matrix<unsigned int, std::size_t, column_major> source_column_major(20,10);
	for(unsigned int i = 0; i != 10; ++i){	
		auto in_row_pos = source_row_major.major_begin(i);
		auto in_col_pos = source_column_major.major_begin(i);
		for(unsigned int j = 1; j < 20; j += (i + 1)){
			in_row_pos = source_row_major.set_element(in_row_pos, j,2*(20*i+1)+1);
			in_col_pos = source_column_major.set_element(in_col_pos, j,2*(20*i+1)+1);
		}
	}
	
	//test all 4 combinations of row/column major
	{
		compressed_matrix<unsigned int> target(10,20,0);
		for(unsigned int i = 0; i != 10; ++i){
			auto tar_pos = target.major_begin(i);
			for(unsigned int j = 1; j < 20; j += (i + 1)){
				tar_pos =  target.set_element(tar_pos,j,4*(20*i+1)+9);
			}
		}
		std::cout<<"testing row-row"<<std::endl;
		kernels::assign(target,source_row_major);
		checkSparseMatrixEqual(target,source_row_major);
	}
	
	{
		compressed_matrix<unsigned int> target(20,10,0);
		for(unsigned int i = 0; i != 20; ++i){
			auto tar_pos = target.major_begin(i);
			for(unsigned int j = 1; j < 10; j += (i + 1)){
				tar_pos =  target.set_element(tar_pos,j,4*(20*i+1)+9);
			}
		}
		std::cout<<"testing row-column"<<std::endl;
		kernels::assign(target,source_column_major);
		checkSparseMatrixEqual(target,source_column_major);
	}
	
	{
		compressed_matrix<unsigned int, std::size_t, column_major> target(10,20);
		for(unsigned int i = 0; i != 20; ++i){
			auto tar_pos = target.major_begin(i);
			for(unsigned int j = 1; j < 10; j += (i + 1)){
				tar_pos =  target.set_element(tar_pos,j,4*(20*i+1)+9);
			}
		}
		std::cout<<"testing column-row"<<std::endl;
		kernels::assign(target,source_row_major);
		checkSparseMatrixEqual(target,source_row_major);
	}
	
	{
		compressed_matrix<unsigned int, std::size_t, column_major> target(20,10);
		for(unsigned int i = 0; i != 10; ++i){
			auto tar_pos = target.major_begin(i);
			for(unsigned int j = 1; j < 20; j += (i + 1)){
				tar_pos =  target.set_element(tar_pos,j,4*(20*i+1)+9);
			}
		}
		std::cout<<"testing column-column"<<std::endl;
		kernels::assign(target,source_column_major);
		checkSparseMatrixEqual(target,source_column_major);
	}
}



//////////////////////////////////////////////////////
//////PLUS ASSIGNMENT
//////////////////////////////////////////////////////

BOOST_AUTO_TEST_CASE( Remora_Dense_Dense_Matrix_Plus_Assign ){
	std::cout<<"\ntesting dense-dense functor assignment"<<std::endl;
	matrix<unsigned int,row_major> source_row_major(10,20);
	matrix<unsigned int,column_major> source_column_major(10,20);
	matrix<unsigned int,row_major> preinit(10,20);
	matrix<unsigned int,row_major> result(10,20);
	typedef device_traits<cpu_tag>::add<unsigned int> functor;
	for(unsigned int i = 0; i != 10; ++i){
		for(unsigned int j = 0; j != 20; ++j){
			source_row_major(i,j) = 2*(20*i+1)+1;
			source_column_major(i,j) =  source_row_major(i,j);
			preinit(i,j) = 3*(20*i+1)+2;
			result(i,j) = preinit(i,j)+source_row_major(i,j);
		}
	}

	//test all 4 combinations of row/column major
	{
		matrix<unsigned int,row_major> target = preinit;
		std::cout<<"testing row-row"<<std::endl;
		kernels::assign(target,source_row_major, functor());
		checkMatrixEqual(target,result);
	}
	
	{
		matrix<unsigned int,row_major> target = preinit;
		std::cout<<"testing row-column"<<std::endl;
		kernels::assign(target,source_column_major, functor());
		checkMatrixEqual(target,result);
	}
	
	{
		matrix<unsigned int,column_major> target = preinit;
		std::cout<<"testing column-row"<<std::endl;
		kernels::assign(target,source_row_major, functor());
		checkMatrixEqual(target,result);
	}
	
	{
		matrix<unsigned int,column_major> target = preinit;
		std::cout<<"testing column-column"<<std::endl;
		kernels::assign(target,source_column_major, functor());
		checkMatrixEqual(target,result);
	}
}

BOOST_AUTO_TEST_CASE( Remora_Dense_Sparse_Matrix_Plus_Assign ){
	std::cout<<"\ntesting dense-sparse functor assignment"<<std::endl;
	typedef device_traits<cpu_tag>::add<unsigned int> functor;
	
	matrix<unsigned int,row_major> preinit(10,20);
	for(unsigned int i = 0; i != 10; ++i){
		for(unsigned int j = 0; j != 20; ++j){
			preinit(i,j) = 3*(20*i+1)+2;
		}
	}
	
	compressed_matrix<unsigned int> source_row_major(10,20);
	matrix<unsigned int,row_major> result = preinit;
	for(unsigned int i = 0; i != 10; ++i){	
		auto in_row_pos = source_row_major.major_begin(i);
		for(unsigned int j = 1; j < 20; j += (i + 1)){
			in_row_pos = source_row_major.set_element(in_row_pos, j,2*(20*i+1)+1);
			result(i,j) += 2*(20*i+1)+1;
		}
	}
	compressed_matrix<unsigned int, std::size_t, column_major> source_column_major = source_row_major;

	//test all 4 combinations of row/column major
	{
		matrix<unsigned int,row_major> target = preinit;
		std::cout<<"testing row-row"<<std::endl;
		kernels::assign(target,source_row_major, functor());
		checkMatrixEqual(target,result);
	}
	
	{
		matrix<unsigned int,row_major> target = preinit;
		std::cout<<"testing row-column"<<std::endl;
		kernels::assign(target,source_column_major, functor());
		checkMatrixEqual(target,result);
	}
	
	{
		matrix<unsigned int,column_major> target = preinit;
		std::cout<<"testing column-row"<<std::endl;
		kernels::assign(target,source_row_major, functor());
		checkMatrixEqual(target,result);
	}
	
	{
		matrix<unsigned int,column_major> target = preinit;
		std::cout<<"testing column-column"<<std::endl;
		kernels::assign(target,source_column_major, functor());
		checkMatrixEqual(target,result);
	}
}


BOOST_AUTO_TEST_CASE( Remora_Sparse_Sparse_Matrix_Plus_Assign ){
	std::cout<<"\ntesting sparse-sparse functor assignment"<<std::endl;
	
	
	compressed_matrix<unsigned int> preinit(10,20);
	matrix<unsigned int> result_dense(10,20,0);
	compressed_matrix<unsigned int> result(10,20);
	typedef device_traits<cpu_tag>::add<unsigned int> functor;
	
	compressed_matrix<unsigned int> source_row_major(10,20);
	for(unsigned int i = 0; i != 10; ++i){	
		auto in_row_pos = source_row_major.major_begin(i);
		for(unsigned int j = 1; j < 20; j += (i + 1)){
			in_row_pos = source_row_major.set_element(in_row_pos, j,2*(20*i+1)+1);
			result_dense(i,j) += 2*(20*i+1)+1;
		}
	}
	compressed_matrix<unsigned int, std::size_t, column_major> source_column_major = source_row_major;
	
	for(unsigned int i = 0; i != 10; ++i){
		auto pos = preinit.major_begin(i);
		for(unsigned int j = 0; j < 20; j += (i + 2) / 2){
			pos = preinit.set_element(pos, j,3*(20*i+1)+2);
			result_dense(i,j) += 3*(20*i+1)+2;
		}
	}
	compressed_matrix<unsigned int> preinit_c = preinit;
	checkSparseMatrixEqual(source_column_major,source_row_major);
	
	for(unsigned int i = 0; i != 10; ++i){
		auto pos = result.major_begin(i);
		for(unsigned int j = 0; j < 20; ++j){
			int r = result_dense(i,j);
			if(r != 0)
				pos = result.set_element(pos, j,r);
		}
	}

	//test all 4 combinations of row/column major
	{
		compressed_matrix<unsigned int, std::size_t, row_major> target = preinit;
		std::cout<<"testing row-row"<<std::endl;
		kernels::assign(target,source_row_major, functor());
		checkSparseMatrixEqual(target,result);
	}
	{
		compressed_matrix<unsigned int, std::size_t, row_major> target = preinit;
		std::cout<<"testing row-column"<<std::endl;
		kernels::assign(target,source_column_major, functor());
		checkSparseMatrixEqual(target,result);
	}
	{
		compressed_matrix<unsigned int, std::size_t, column_major> target = preinit;
		checkSparseMatrixEqual(target,preinit);
		std::cout<<"testing column-row"<<std::endl;
		kernels::assign(target,source_row_major, functor());
		checkSparseMatrixEqual(target,result);
	}
	
	{
		compressed_matrix<unsigned int, std::size_t, column_major> target = preinit;
		checkSparseMatrixEqual(target,preinit);
		std::cout<<"testing column-column"<<std::endl;
		kernels::assign(target,source_column_major, functor());
		checkSparseMatrixEqual(target,result);
	}
}
BOOST_AUTO_TEST_SUITE_END()
