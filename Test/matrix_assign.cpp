#define BOOST_TEST_MODULE LinAlg_BLAS_VectorAssign
#include <boost/test/unit_test.hpp>
#include <boost/test/floating_point_comparison.hpp>

#include <shark/LinAlg/BLAS/blas.h>
#include <shark/LinAlg/BLAS/triangular_matrix.hpp>

using namespace shark;

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


BOOST_AUTO_TEST_SUITE (LinAlg_BLAS_matrix_assign)


///////////////////////////////////////////////////////////////////////////////
//////FUNCTOR CONSTANT ASSIGNMENT
//////////////////////////////////////////////////////////////////////////////

//test for assign of the form m_ij=f(m_ij,t) for constant t
BOOST_AUTO_TEST_CASE( LinAlg_BLAS_Dense_Matrix_Constant_Functor_Assign ){
	std::cout<<"testing dense functor-constant assignment."<<std::endl;
	blas::matrix<unsigned int,blas::row_major> input_row_major(10,20);
	blas::matrix<unsigned int,blas::row_major> input_column_major(10,20);
	blas::matrix<unsigned int,blas::row_major> target(10,20);
	const unsigned int t = 2;
	for(std::size_t i = 0; i != 10; ++i){
		for(std::size_t j = 0; j != 20; ++j){
			input_row_major(i,j) = 2*(20*i+1)+1.0;
			input_column_major(i,j) = 2*(20*i+1)+1.0;
			target(i,j)=t*input_row_major(i,j);
		}
	}

	std::cout<<"testing row-major"<<std::endl;
	blas::kernels::assign<blas::scalar_multiply_assign> (input_row_major, t);
	checkMatrixEqual(target,input_row_major);
	
	std::cout<<"testing column-major"<<std::endl;
	blas::kernels::assign<blas::scalar_multiply_assign> (input_column_major, t);
	checkMatrixEqual(target,input_column_major);
	
	std::cout<<"\n";
}

BOOST_AUTO_TEST_CASE( LinAlg_BLAS_Packed_Matrix_Constant_Functor_Assign ){
	std::cout<<"testing packed functor-constant assignment."<<std::endl;
	blas::triangular_matrix<unsigned int,blas::row_major,blas::lower> input_row_lower(20);
	blas::triangular_matrix<unsigned int,blas::row_major,blas::upper> input_row_upper(20);
	blas::triangular_matrix<unsigned int,blas::column_major,blas::lower> input_column_lower(20);
	blas::triangular_matrix<unsigned int,blas::column_major,blas::upper> input_column_upper(20);
	blas::triangular_matrix<unsigned int,blas::row_major,blas::lower> target_lower(20);
	blas::triangular_matrix<unsigned int,blas::row_major,blas::upper> target_upper(20);
	const unsigned int t = 2;
	for(unsigned int i = 0; i != 20; ++i){
		for(unsigned int j = 0; j <= i; ++j){
			input_row_lower.set_element(i,j, 2*(20*i+1)+1);
			input_column_lower.set_element(i,j, 2*(20*i+1)+1);
			target_lower.set_element(i,j,t*input_row_lower(i,j));
		}
	}
	
	for(unsigned int i = 0; i != 20; ++i){
		for(unsigned int j = i; j != 20; ++j){
			input_row_upper.set_element(i,j,2*(20*i+1)+1);
			input_column_upper.set_element(i,j,2*(20*i+1)+1);
			target_upper.set_element(i,j,t*input_row_upper(i,j));
		}
	}

	std::cout<<"testing row-major lower"<<std::endl;
	blas::kernels::assign<blas::scalar_multiply_assign> (input_row_lower, t);
	checkMatrixEqual(target_lower,input_row_lower);
	
	std::cout<<"testing column-major lower"<<std::endl;
	blas::kernels::assign<blas::scalar_multiply_assign> (input_column_lower, t);
	checkMatrixEqual(target_lower,input_column_lower);
	
	std::cout<<"testing row-major upper"<<std::endl;
	blas::kernels::assign<blas::scalar_multiply_assign> (input_row_upper, t);
	checkMatrixEqual(target_upper,input_row_upper);
	
	std::cout<<"testing column-major upper"<<std::endl;
	blas::kernels::assign<blas::scalar_multiply_assign> (input_column_upper, t);
	checkMatrixEqual(target_upper,input_column_upper);
	
	std::cout<<"\n";
}

BOOST_AUTO_TEST_CASE( LinAlg_BLAS_Sparse_Matrix_Constant_Functor_Assign ){
	std::cout<<"testing sparse functor-constant assignment."<<std::endl;
	blas::compressed_matrix<unsigned int> input_row_major(10,20,0);
	blas::compressed_matrix<unsigned int> target_row_major(10,20,0);
	blas::compressed_matrix<unsigned int> input_column_major_base(20,10,0);
	blas::compressed_matrix<unsigned int> target_column_major_base(20,10,0);
	blas::matrix_transpose<blas::compressed_matrix<unsigned int> > input_column_major(input_column_major_base);
	blas::matrix_transpose<blas::compressed_matrix<unsigned int> > target_column_major(target_column_major_base);
	const unsigned int t = 2;
	for(unsigned int i = 0; i != 10; ++i){
		for(unsigned int j = 1; j < 20; j += (i + 1)){
			input_row_major(i,j) = 2*(20*i+1)+1;
			input_column_major(i,j) =  2*(20*i+1)+1;//source_row_major(i,j)+2;
			target_row_major(i,j) = t*input_row_major(i,j);
			target_column_major(i,j) = t*input_column_major(i,j);
		}
	}

	std::cout<<"testing row-major"<<std::endl;
	blas::kernels::assign<blas::scalar_multiply_assign> (input_row_major, t);
	checkMatrixEqual(target_row_major,input_row_major);
	
	std::cout<<"testing column-major"<<std::endl;
	blas::kernels::assign<blas::scalar_multiply_assign> (input_column_major, t);
	checkMatrixEqual(target_column_major,input_column_major);
	
	std::cout<<"\n";
}

//////////////////////////////////////////////////////
//////SIMPLE ASSIGNMENT
//////////////////////////////////////////////////////

BOOST_AUTO_TEST_CASE( LinAlg_BLAS_Dense_Dense_Matrix_Assign ){
	std::cout<<"testing direct dense-dense assignment"<<std::endl;
	blas::matrix<unsigned int,blas::row_major> source_row_major(10,20);
	blas::matrix<unsigned int,blas::column_major> source_column_major(10,20);

	for(unsigned int i = 0; i != 10; ++i){
		for(unsigned int j = 0; j != 20; ++j){
			source_row_major(i,j) = 2*(20*i+1)+1;
			source_column_major(i,j) =  source_row_major(i,j)+2;
		}
	}

	//test all 4 combinations of row/column major
	{
		blas::matrix<unsigned int,blas::row_major> target(10,20);
		for(unsigned int i = 0; i != 10; ++i){
			for(unsigned int j = 0; j != 20; ++j){
				target(i,j) = 3*(20*i+1)+2;
			}
		}
		std::cout<<"testing row-row"<<std::endl;
		blas::kernels::assign(target,source_row_major);
		checkMatrixEqual(target,source_row_major);
	}
	
	{
		blas::matrix<unsigned int,blas::row_major> target(10,20);
		for(unsigned int i = 0; i != 10; ++i){
			for(unsigned int j = 0; j != 20; ++j){
				target(i,j) = 3*(20*i+1)+2;
			}
		}
		std::cout<<"testing row-column"<<std::endl;
		blas::kernels::assign(target,source_column_major);
		checkMatrixEqual(target,source_column_major);
	}
	
	{
		blas::matrix<unsigned int,blas::column_major> target(10,20);
		for(unsigned int i = 0; i != 10; ++i){
			for(unsigned int j = 0; j != 20; ++j){
				target(i,j) = 3*(20*i+1)+2;
			}
		}
		std::cout<<"testing column-row"<<std::endl;
		blas::kernels::assign(target,source_row_major);
		checkMatrixEqual(target,source_row_major);
	}
	
	{
		blas::matrix<unsigned int,blas::column_major> target(10,20);
		for(unsigned int i = 0; i != 10; ++i){
			for(unsigned int j = 0; j != 20; ++j){
				target(i,j) = 3*(20*i+1)+2;
			}
		}
		std::cout<<"testing column-column"<<std::endl;
		blas::kernels::assign(target,source_column_major);
		checkMatrixEqual(target,source_column_major);
	}
}

//not implemented yet!
//~ //Dense-Packed
//~ BOOST_AUTO_TEST_CASE( LinAlg_BLAS_Dense_Packed_Matrix_Assign ){
	//~ std::cout<<"testing direct dense-packed assignment"<<std::endl;
	
	//~ //create the 4 different source matrices
	//~ typedef blas::triangular_matrix<unsigned int,blas::row_major,blas::upper> MRU;
	//~ typedef blas::triangular_matrix<unsigned int,blas::column_major,blas::upper> MCU;
	//~ typedef blas::triangular_matrix<unsigned int,blas::row_major,blas::lower> MRL;
	//~ typedef blas::triangular_matrix<unsigned int,blas::column_major,blas::lower> MCL;
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
		//~ blas::matrix<unsigned int,blas::row_major> target(20,20);
		//~ for(std::size_t i = 0; i != 20; ++i){
			//~ for(std::size_t j = 0; j != 20; ++j){
				//~ target(i,j) = 3*(20*i+1)+2;
			//~ }
		//~ }
		//~ std::cout<<"testing row-row/upper"<<std::endl;
		//~ blas::kernels::assign(target,source_upper_row_major);
		//~ checkMatrixEqual(target,source_upper_row_major);
	//~ }
	//~ {
		//~ blas::matrix<unsigned int,blas::row_major> target(20,20);
		//~ for(std::size_t i = 0; i != 20; ++i){
			//~ for(std::size_t j = 0; j != 20; ++j){
				//~ target(i,j) = 3*(20*i+1)+2;
			//~ }
		//~ }
		//~ std::cout<<"testing row-row/lower"<<std::endl;
		//~ blas::kernels::assign(target,source_lower_row_major);
		//~ checkMatrixEqual(target,source_lower_row_major);
	//~ }
	
	//~ {
		//~ blas::matrix<unsigned int,blas::row_major> target(20,20);
		//~ for(std::size_t i = 0; i != 20; ++i){
			//~ for(std::size_t j = 0; j != 20; ++j){
				//~ target(i,j) = 3*(20*i+1)+2;
			//~ }
		//~ }
		//~ std::cout<<"testing row-column/upper"<<std::endl;
		//~ blas::kernels::assign(target,source_upper_column_major);
		//~ checkMatrixEqual(target,source_upper_column_major);
	//~ }
	//~ {
		//~ blas::matrix<unsigned int,blas::row_major> target(20,20);
		//~ for(std::size_t i = 0; i != 20; ++i){
			//~ for(std::size_t j = 0; j != 20; ++j){
				//~ target(i,j) = 3*(20*i+1)+2;
			//~ }
		//~ }
		//~ std::cout<<"testing row-column/lower"<<std::endl;
		//~ blas::kernels::assign(target,source_lower_column_major);
		//~ checkMatrixEqual(target,source_lower_column_major);
	//~ }
	
	//~ {
		//~ blas::matrix<unsigned int,blas::column_major> target(20,20);
		//~ for(std::size_t i = 0; i != 20; ++i){
			//~ for(std::size_t j = 0; j != 20; ++j){
				//~ target(i,j) = 3*(20*i+1)+2;
			//~ }
		//~ }
		//~ std::cout<<"testing column-row/upper"<<std::endl;
		//~ blas::kernels::assign(target,source_upper_row_major);
		//~ checkMatrixEqual(target,source_upper_row_major);
	//~ }
	//~ {
		//~ blas::matrix<unsigned int,blas::column_major> target(20,20);
		//~ for(std::size_t i = 0; i != 20; ++i){
			//~ for(std::size_t j = 0; j != 20; ++j){
				//~ target(i,j) = 3*(20*i+1)+2;
			//~ }
		//~ }
		//~ std::cout<<"testing column-row/lower"<<std::endl;
		//~ blas::kernels::assign(target,source_lower_row_major);
		//~ checkMatrixEqual(target,source_lower_row_major);
	//~ }
	
	//~ {
		//~ blas::matrix<unsigned int,blas::column_major> target(20,20);
		//~ for(std::size_t i = 0; i != 20; ++i){
			//~ for(std::size_t j = 0; j != 20; ++j){
				//~ target(i,j) = 3*(20*i+1)+2;
			//~ }
		//~ }
		//~ std::cout<<"testing column-column/upper"<<std::endl;
		//~ blas::kernels::assign(target,source_upper_column_major);
		//~ checkMatrixEqual(target,source_upper_column_major);
	//~ }
	//~ {
		//~ blas::matrix<unsigned int,blas::column_major> target(20,20);
		//~ for(std::size_t i = 0; i != 20; ++i){
			//~ for(std::size_t j = 0; j != 20; ++j){
				//~ target(i,j) = 3*(20*i+1)+2;
			//~ }
		//~ }
		//~ std::cout<<"testing column-column/lower"<<std::endl;
		//~ blas::kernels::assign(target,source_lower_column_major);
		//~ checkMatrixEqual(target,source_lower_column_major);
	//~ }
//~ }


BOOST_AUTO_TEST_CASE( LinAlg_BLAS_Packed_Packed_Matrix_Assign ){
	std::cout<<"testing direct packed-packed assignment"<<std::endl;
	
	//create the 4 different source matrices
	typedef blas::triangular_matrix<unsigned int,blas::row_major,blas::upper> MRU;
	typedef blas::triangular_matrix<unsigned int,blas::column_major,blas::upper> MCU;
	typedef blas::triangular_matrix<unsigned int,blas::row_major,blas::lower> MRL;
	typedef blas::triangular_matrix<unsigned int,blas::column_major,blas::lower> MCL;
	MRU source_upper_row_major(20);
	MRL  source_lower_row_major(20);
	MCU source_upper_column_major(20);
	MCL source_lower_column_major(20);

	for(unsigned int i = 0; i != 20; ++i){
		MRU::row_iterator pos1=source_upper_row_major.row_begin(i);
		MRL::row_iterator pos2=source_lower_row_major.row_begin(i);
		MCU::column_iterator pos3=source_upper_column_major.column_begin(i);
		MCL::column_iterator pos4=source_lower_column_major.column_begin(i);
		for(; pos1 != source_upper_row_major.row_end(i);++pos1){
			*pos1 = i*20+pos1.index()+1;
		}
		for(; pos2 != source_lower_row_major.row_end(i);++pos2){
			*pos2 = i*20+pos2.index()+1;
		}
		for(; pos3 != source_upper_column_major.column_end(i);++pos3){
			*pos3 = i*20+pos3.index()+1;
		}
		for(; pos4 != source_lower_column_major.column_end(i);++pos4){
			*pos4 = i*20+pos4.index()+1;
		}
	}

	//test all 8 combinations of row/column major  target and the four sources
	//for simplicitely we just assign the targets to be 1...
	{
		MRU target(20,1);
		std::cout<<"testing row-row/upper"<<std::endl;
		blas::kernels::assign(target,source_upper_row_major);
		checkMatrixEqual(target,source_upper_row_major);
	}
	{
		MRL target(20,1);
		std::cout<<"testing row-row/lower"<<std::endl;
		blas::kernels::assign(target,source_lower_row_major);
		checkMatrixEqual(target,source_lower_row_major);
	}
	{
		MRU target(20,1);
		std::cout<<"testing row-column/upper"<<std::endl;
		blas::kernels::assign(target,source_upper_column_major);
		checkMatrixEqual(target,source_upper_column_major);
	}
	{
		MRL target(20,1);
		std::cout<<"testing row-column/lower"<<std::endl;
		blas::kernels::assign(target,source_lower_column_major);
		checkMatrixEqual(target,source_lower_column_major);
	}
	{
		MCU target(20,1);
		std::cout<<"testing column-row/upper"<<std::endl;
		blas::kernels::assign(target,source_upper_row_major);
		checkMatrixEqual(target,source_upper_row_major);
	}
	{
		MCL target(20,1);
		std::cout<<"testing column-row/lower"<<std::endl;
		blas::kernels::assign(target,source_lower_row_major);
		checkMatrixEqual(target,source_lower_row_major);
	}
	{
		MCU target(20,1);
		std::cout<<"testing column-column/upper"<<std::endl;
		blas::kernels::assign(target,source_upper_column_major);
		checkMatrixEqual(target,source_upper_column_major);
	}
	{
		MCL target(20,1);
		std::cout<<"testing column-column/lower"<<std::endl;
		blas::kernels::assign(target,source_lower_column_major);
		checkMatrixEqual(target,source_lower_column_major);
	}
}


BOOST_AUTO_TEST_CASE( LinAlg_BLAS_Dense_Sparse_Matrix_Assign ){
	std::cout<<"\ntesting direct dense-sparse assignment"<<std::endl;
	blas::compressed_matrix<unsigned int> source_row_major(10,20,0);
	blas::compressed_matrix<unsigned int> source_column_major_base(20,10);
	blas::matrix_transpose<blas::compressed_matrix<unsigned int> > source_column_major(source_column_major_base);
	
	for(unsigned int i = 0; i != 10; ++i){
		for(unsigned int j = 1; j < 20; j += (i + 1)){
			source_row_major(i,j) = 2*(20*i+1)+1;
			source_column_major(i,j) =  2*(20*i+1)+1;//source_row_major(i,j)+2;
		}
	}
	//test all 4 combinations of row/column major
	{
		blas::matrix<unsigned int,blas::row_major> target(10,20);
		for(unsigned int i = 0; i != 10; ++i){
			for(unsigned int j = 0; j != 20; ++j){
				target(i,j) = 3*(20*i+1)+2;
			}
		}
		std::cout<<"testing row-row"<<std::endl;
		blas::kernels::assign(target,source_row_major);
		checkMatrixEqual(target,source_row_major);
	}
	
	{
		blas::matrix<unsigned int,blas::row_major> target(10,20);
		for(unsigned int i = 0; i != 10; ++i){
			for(unsigned int j = 0; j != 20; ++j){
				target(i,j) = 3*(20*i+1)+2;
			}
		}
		std::cout<<"testing row-column"<<std::endl;
		blas::kernels::assign(target,source_column_major);
		checkMatrixEqual(target,source_column_major);
	}
	
	{
		blas::matrix<unsigned int,blas::column_major> target(10,20);
		for(unsigned int i = 0; i != 10; ++i){
			for(unsigned int j = 0; j != 20; ++j){
				target(i,j) = 3*(20*i+1)+2;
			}
		}
		std::cout<<"testing column-row"<<std::endl;
		blas::kernels::assign(target,source_row_major);
		checkMatrixEqual(target,source_row_major);
	}
	
	{
		blas::matrix<unsigned int,blas::column_major> target(10,20);
		for(unsigned int i = 0; i != 10; ++i){
			for(unsigned int j = 0; j != 20; ++j){
				target(i,j) = 3*(20*i+1)+2;
			}
		}
		std::cout<<"testing column-column"<<std::endl;
		blas::kernels::assign(target,source_column_major);
		checkMatrixEqual(target,source_column_major);
	}
	
}

BOOST_AUTO_TEST_CASE( LinAlg_BLAS_Sparse_Dense_Matrix_Assign ){
	std::cout<<"\ntesting direct sparse-dense assignment"<<std::endl;
	blas::matrix<unsigned int,blas::row_major> source_row_major(10,20);
	blas::matrix<unsigned int,blas::column_major> source_column_major(10,20);

	for(unsigned int i = 0; i != 10; ++i){
		for(unsigned int j = 0; j != 20; ++j){
			source_row_major(i,j) = 2*(20*i+1)+1;
			source_column_major(i,j) =  source_row_major(i,j)+2;
		}
	}
	
	//test all 4 combinations of row/column major
	{
		blas::compressed_matrix<unsigned int> target(10,20,0);
		for(unsigned int i = 0; i != 10; ++i){
			for(unsigned int j = 1; j < 20; j += (i + 1)){
				target(i,j) = 4*(20*i+1)+9;
			}
		}
		std::cout<<"testing row-row"<<std::endl;
		blas::kernels::assign(target,source_row_major);
		checkMatrixEqual(target,source_row_major);
	}
	
	{
		blas::compressed_matrix<unsigned int> target_base(20,10);
		blas::matrix_transpose<blas::compressed_matrix<unsigned int> > target(target_base);
		for(unsigned int i = 0; i != 10; ++i){
			for(unsigned int j = 1; j < 20; j += (i + 1)){
				target(i,j) =  4*(20*i+1)+9;
			}
		}
		std::cout<<"testing row-column"<<std::endl;
		blas::kernels::assign(target,source_column_major);
		checkMatrixEqual(target,source_column_major);
	}
	
	{
		blas::compressed_matrix<unsigned int> target(10,20,0);
		for(unsigned int i = 0; i != 10; ++i){
			for(unsigned int j = 1; j < 20; j += (i + 1)){
				target(i,j) = 4*(20*i+1)+9;
			}
		}
		std::cout<<"testing column-row"<<std::endl;
		blas::kernels::assign(target,source_row_major);
		checkMatrixEqual(target,source_row_major);
	}
	
	{
		blas::compressed_matrix<unsigned int> target_base(20,10);
		blas::matrix_transpose<blas::compressed_matrix<unsigned int> > target(target_base);
		for(unsigned int i = 0; i != 10; ++i){
			for(unsigned int j = 1; j < 20; j += (i + 1)){
				target(i,j) =  4*(20*i+1)+9;
			}
		}
		std::cout<<"testing column-column"<<std::endl;
		blas::kernels::assign(target,source_column_major);
		checkMatrixEqual(target,source_column_major);
	}
}


BOOST_AUTO_TEST_CASE( LinAlg_BLAS_Sparse_Sparse_Matrix_Assign ){
	std::cout<<"\ntesting direct sparse-sparse assignment"<<std::endl;
	blas::compressed_matrix<unsigned int> source_row_major(10,20,0);
	blas::compressed_matrix<unsigned int> source_column_major_base(20,10);
	blas::matrix_transpose<blas::compressed_matrix<unsigned int> > source_column_major(source_column_major_base);
	
	for(unsigned int i = 0; i != 10; ++i){
		for(unsigned int j = 1; j < 20; j += (i + 1)){
			source_row_major(i,j) = 2*(20*i+1)+1;
			source_column_major(i,j) =  source_row_major(i,j);
		}
	}
	
	//test all 4 combinations of row/column major
	{
		blas::compressed_matrix<unsigned int> target(10,20,0);
		//~ for(std::size_t i = 0; i != 10; ++i){
			//~ for(std::size_t j = 1; j < 20; j+=(i+1)){
				//~ target(i,j) = 4*(20*i+1)+9;
			//~ }
		//~ }
		std::cout<<"testing row-row"<<std::endl;
		blas::kernels::assign(target,source_row_major);
		checkMatrixEqual(target,source_row_major);
	}
	
	{
		blas::compressed_matrix<unsigned int> target(10,20,0);
		//~ for(std::size_t i = 0; i != 10; ++i){
			//~ for(std::size_t j = 1; j < 20; j+=(i+1)){
				//~ target(i,j) = 4*(20*i+1)+9;
			//~ }
		//~ }
		std::cout<<"testing row-column"<<std::endl;
		blas::kernels::assign(target,source_column_major);
		checkMatrixEqual(target,source_column_major);
	}
	
	{
		blas::compressed_matrix<unsigned int> target_base(20,10);
		blas::matrix_transpose<blas::compressed_matrix<unsigned int> > target(target_base);
		//~ for(std::size_t i = 0; i != 10; ++i){
			//~ for(std::size_t j = 1; j < 20; j+=(i+1)){
				//~ target(i,j) = 4*(20*i+1)+9;
			//~ }
		//~ }
		std::cout<<"testing column-row"<<std::endl;
		blas::kernels::assign(target,source_row_major);
		checkMatrixEqual(target,source_row_major);
	}
	
	{
		blas::compressed_matrix<unsigned int> target_base(20,10);
		blas::matrix_transpose<blas::compressed_matrix<unsigned int> > target(target_base);
		for(unsigned int i = 0; i != 10; ++i){
			for(unsigned int j = 1; j < 20; j += (i + 1)){
				target(i,j) =  4*(20*i+1)+9;
			}
		}
		std::cout<<"testing column-column"<<std::endl;
		blas::kernels::assign(target,source_column_major);
		checkMatrixEqual(target,source_column_major);
	}
}



//////////////////////////////////////////////////////
//////PLUS ASSIGNMENT
//////////////////////////////////////////////////////

BOOST_AUTO_TEST_CASE( LinAlg_BLAS_Dense_Dense_Matrix_Plus_Assign ){
	std::cout<<"\ntesting dense-dense functor assignment"<<std::endl;
	blas::matrix<unsigned int,blas::row_major> source_row_major(10,20);
	blas::matrix<unsigned int,blas::column_major> source_column_major(10,20);
	blas::matrix<unsigned int,blas::row_major> preinit(10,20);
	blas::matrix<unsigned int,blas::row_major> result(10,20);

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
		blas::matrix<unsigned int,blas::row_major> target = preinit;
		std::cout<<"testing row-row"<<std::endl;
		blas::kernels::assign<blas::scalar_plus_assign>(target,source_row_major);
		checkMatrixEqual(target,result);
	}
	
	{
		blas::matrix<unsigned int,blas::row_major> target = preinit;
		std::cout<<"testing row-column"<<std::endl;
		blas::kernels::assign<blas::scalar_plus_assign>(target,source_column_major);
		checkMatrixEqual(target,result);
	}
	
	{
		blas::matrix<unsigned int,blas::column_major> target = preinit;
		std::cout<<"testing column-row"<<std::endl;
		blas::kernels::assign<blas::scalar_plus_assign>(target,source_row_major);
		checkMatrixEqual(target,result);
	}
	
	{
		blas::matrix<unsigned int,blas::column_major> target = preinit;
		std::cout<<"testing column-column"<<std::endl;
		blas::kernels::assign<blas::scalar_plus_assign>(target,source_column_major);
		checkMatrixEqual(target,result);
	}
}

BOOST_AUTO_TEST_CASE( LinAlg_BLAS_Dense_Sparse_Matrix_Plus_Assign ){
	std::cout<<"\ntesting dense-sparse functor assignment"<<std::endl;
	blas::compressed_matrix<unsigned int> source_row_major(10,20);
	blas::compressed_matrix<unsigned int> source_column_major_base(20,10);
	blas::matrix_transpose<blas::compressed_matrix<unsigned int> > source_column_major(source_column_major_base);
	blas::matrix<unsigned int,blas::row_major> preinit(10,20);
	blas::matrix<unsigned int,blas::row_major> result(10,20);
	
	for(unsigned int i = 0; i != 10; ++i){
		for(unsigned int j = 1; j < 20; j += (i + 1)){
			source_row_major(i,j) = 2*(20*i+1)+1;
			source_column_major(i,j) = 2*(20*i+1)+1;
		}
	}

	for(unsigned int i = 0; i != 10; ++i){
		for(unsigned int j = 0; j != 20; ++j){
			preinit(i,j) = 3*(20*i+1)+2;
			result(i,j) = preinit(i,j)+source_row_major(i,j);
		}
	}

	//test all 4 combinations of row/column major
	{
		blas::matrix<unsigned int,blas::row_major> target = preinit;
		std::cout<<"testing row-row"<<std::endl;
		blas::kernels::assign<blas::scalar_plus_assign>(target,source_row_major);
		checkMatrixEqual(target,result);
	}
	
	{
		blas::matrix<unsigned int,blas::row_major> target = preinit;
		std::cout<<"testing row-column"<<std::endl;
		blas::kernels::assign<blas::scalar_plus_assign>(target,source_column_major);
		checkMatrixEqual(target,result);
	}
	
	{
		blas::matrix<unsigned int,blas::column_major> target = preinit;
		std::cout<<"testing column-row"<<std::endl;
		blas::kernels::assign<blas::scalar_plus_assign>(target,source_row_major);
		checkMatrixEqual(target,result);
	}
	
	{
		blas::matrix<unsigned int,blas::column_major> target = preinit;
		std::cout<<"testing column-column"<<std::endl;
		blas::kernels::assign<blas::scalar_plus_assign>(target,source_column_major);
		checkMatrixEqual(target,result);
	}
}


BOOST_AUTO_TEST_CASE( LinAlg_BLAS_Sparse_Sparse_Matrix_Plus_Assign ){
	std::cout<<"\ntesting sparse-sparse functor assignment"<<std::endl;
	blas::compressed_matrix<unsigned int> source_row_major(10,20);
	blas::compressed_matrix<unsigned int> source_column_major_base(20,10);
	blas::matrix_transpose<blas::compressed_matrix<unsigned int> > source_column_major(source_column_major_base);
	blas::compressed_matrix<unsigned int> preinit(10,20);
	blas::compressed_matrix<unsigned int> result(10,20);
	
	for(unsigned int i = 0; i != 10; ++i){
		for(unsigned int j = 1; j < 20; j += (i + 1)){
			source_row_major(i,j) = 2*(20*i+1)+1;
			source_column_major(i,j) = 2*(20*i+1)+1;
		}
	}

	for(unsigned int i = 0; i != 10; ++i){
		for(unsigned int j = 0; j < 20; j += (i + 2) / 2){
			preinit(i,j) = 3*(20*i+1)+2;
		}
	}
	
	for(unsigned int i = 0; i != 10; ++i){
		for(unsigned int j = 0; j < 20; ++j){
			int r = preinit(i,j)+source_row_major(i,j);
			if(r != 0)
				result(i,j) = r;
		}
	}

	//test all 4 combinations of row/column major
	{
		blas::compressed_matrix<unsigned int> target = preinit;
		std::cout<<"testing row-row"<<std::endl;
		blas::kernels::assign<blas::scalar_plus_assign>(target,source_row_major);
		checkMatrixEqual(target,result);
	}
	
	{
		blas::matrix<unsigned int,blas::row_major> target = preinit;
		std::cout<<"testing row-column"<<std::endl;
		blas::kernels::assign<blas::scalar_plus_assign>(target,source_column_major);
		checkMatrixEqual(target,result);
	}
	{
		blas::compressed_matrix<unsigned int> target_base(20,10);
		blas::matrix_transpose<blas::compressed_matrix<unsigned int> > target(target_base);
		target = preinit;
		std::cout<<"testing column-row"<<std::endl;
		blas::kernels::assign<blas::scalar_plus_assign>(target,source_row_major);
		checkMatrixEqual(target,result);
	}
	
	{
		blas::compressed_matrix<unsigned int> target_base(20,10);
		blas::matrix_transpose<blas::compressed_matrix<unsigned int> > target(target_base);
		target = preinit;
		std::cout<<"testing column-column"<<std::endl;
		blas::kernels::assign<blas::scalar_plus_assign>(target,source_column_major);
		checkMatrixEqual(target,result);
	}
}
BOOST_AUTO_TEST_SUITE_END()
