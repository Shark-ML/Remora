#define BOOST_TEST_MODULE Remora_GPU_MatrixProxy
#include <boost/test/unit_test.hpp>
#include <boost/test/floating_point_comparison.hpp>


#include <remora/matrix_proxy.hpp>
#include <remora/vector.hpp>
#include <remora/matrix.hpp>
#include <remora/vector_expression.hpp>
#include <remora/matrix_expression.hpp>

using namespace remora;

template<class Operation, class Result>
void checkDenseMatrixEquality(Operation op_gpu, Result const& result){
	BOOST_REQUIRE_EQUAL(op_gpu.size1(), result.size1());
	BOOST_REQUIRE_EQUAL(op_gpu.size2(), result.size2());
	
	//test copy to cpu, this tests the buffer
	matrix<float> op = copy_to_cpu(op_gpu);
	for(std::size_t i = 0; i != op.size1(); ++i){
		for(std::size_t j = 0; j != op.size2(); ++j){
			BOOST_CHECK_CLOSE(result(i,j), op(i,j),1.e-8);
		}
	}
	
	//test row iterators
	{
		vector<float, gpu_tag> opcopy_gpu(op.size2());
		for(std::size_t i = 0; i != op.size1(); ++i){
			boost::compute::copy(op_gpu.row_begin(i),op_gpu.row_end(i),opcopy_gpu.begin());
			vector<float> opcopy = copy_to_cpu(opcopy_gpu);
			for(std::size_t j = 0; j != op.size2(); ++j){
				BOOST_CHECK_CLOSE(result(i,j), opcopy(j),1.e-8);
			}
		}
	}
	
	//test column iterators
	{
		vector<float, gpu_tag> opcopy_gpu(op.size1());
		for(std::size_t j = 0; j != op.size2(); ++j){
			boost::compute::copy(op_gpu.column_begin(j),op_gpu.column_end(j),opcopy_gpu.begin());
			vector<float> opcopy = copy_to_cpu(opcopy_gpu);
			for(std::size_t i = 0; i != op.size1(); ++i){
				BOOST_CHECK_CLOSE(result(i,j), opcopy(i),1.e-8);
			}
		}
	}
}
template<class Operation, class Result>
void checkDenseVectorEquality(Operation op_gpu, Result const& result){
	BOOST_REQUIRE_EQUAL(op_gpu.size(), result.size());
	
	//test copy to cpu, this tests the buffer
	vector<float> op = copy_to_cpu(op_gpu);
	BOOST_REQUIRE_EQUAL(op.size(), result.size());
	for(std::size_t i = 0; i != op.size(); ++i){
		BOOST_CHECK_CLOSE(result(i), op(i),1.e-8);
	}
	
	//test iterators
	BOOST_REQUIRE_EQUAL(op_gpu.end() - op_gpu.begin(), op.size());
	vector<float, gpu_tag> opcopy_gpu(op.size());
	boost::compute::copy(op_gpu.begin(),op_gpu.end(),opcopy_gpu.begin());
	vector<float> opcopy = copy_to_cpu(opcopy_gpu);
	for(std::size_t i = 0; i != result.size(); ++i){
		BOOST_CHECK_CLOSE(result(i), opcopy(i),1.e-8);
	}
}


std::size_t Dimensions1 = 20;
std::size_t Dimensions2 = 10;
struct MatrixProxyFixture
{
	matrix<float> denseData_cpu;
	matrix<float, row_major, gpu_tag> denseData;
	matrix<float, column_major, gpu_tag> denseDataColMajor;
	
	MatrixProxyFixture():denseData_cpu(Dimensions1,Dimensions2){
		for(std::size_t row=0;row!= Dimensions1;++row){
			for(std::size_t col=0;col!=Dimensions2;++col){
				denseData_cpu(row,col) = row*Dimensions2+col+5.0;
			}
		}
		denseData = copy_to_gpu(denseData_cpu);
		denseDataColMajor = copy_to_gpu(denseData_cpu);
	}
};

BOOST_FIXTURE_TEST_SUITE (Remora_matrix_proxy, MatrixProxyFixture);

//~ BOOST_AUTO_TEST_CASE( Remora_Dense_Subrange ){
	//~ //all possible combinations of ranges on the data matrix
	//~ for(std::size_t rowEnd=0;rowEnd!= Dimensions1;++rowEnd){
		//~ for(std::size_t rowBegin =0;rowBegin <= rowEnd;++rowBegin){//<= for 0 range
			//~ for(std::size_t colEnd=0;colEnd!=Dimensions2;++colEnd){
				//~ for(std::size_t colBegin=0;colBegin != colEnd;++colBegin){
					//~ //obtain ground truth
					//~ std::size_t size1= rowEnd-rowBegin;
					//~ std::size_t size2= colEnd-colBegin;
					//~ matrix<float> mTest(size1,size2);
					//~ for(std::size_t i = 0; i != size1; ++i){
						//~ for(std::size_t j = 0; j != size2; ++j){
							//~ mTest(i,j) = denseData_cpu(i+rowBegin,j+colBegin);
						//~ }
					//~ }
					//~ //check whether the subrange has the right values
					//~ checkDenseMatrixEquality(subrange(denseData,rowBegin,rowEnd,colBegin,colEnd),mTest);
					//~ checkDenseMatrixEquality(subrange(denseDataColMajor,rowBegin,rowEnd,colBegin,colEnd),mTest);

					//~ //now test whether we can assign to a range like this.
					//~ matrix<float, row_major, gpu_tag> newData(Dimensions1,Dimensions2,1.0);
					//~ matrix<float, column_major, gpu_tag> newDataColMajor(Dimensions1,Dimensions2,1.0);
					//~ auto rangeTest = subrange(newData,rowBegin,rowEnd,colBegin,colEnd);
					//~ auto rangeTestColMajor = subrange(newDataColMajor,rowBegin,rowEnd,colBegin,colEnd);
					//~ noalias(rangeTest) = subrange(denseData,rowBegin,rowEnd,colBegin,colEnd);
					//~ noalias(rangeTestColMajor) = subrange(denseDataColMajor,rowBegin,rowEnd,colBegin,colEnd);
					//~ //check that the assignment has been carried out correctly
					//~ checkDenseMatrixEquality(rangeTest,mTest);
					//~ checkDenseMatrixEquality(rangeTestColMajor,mTest);

					//~ //check that after assignment all elements outside the range are still intact
					//~ //generate ground truth
					//~ matrix<float> truth(Dimensions1,Dimensions2,1.0);
					//~ for(std::size_t i = 0; i != size1; ++i){
						//~ for(std::size_t j = 0; j != size2; ++j){
							//~ truth(i+rowBegin,j+colBegin) = denseData_cpu(i+rowBegin,j+colBegin);
						//~ }
					//~ }	
					//~ matrix<float> data = copy_to_cpu(newData);
					//~ matrix<float> dataColMajor = copy_to_cpu(newDataColMajor);
					//~ for(std::size_t i = 0; i != Dimensions1; ++i){
						//~ for(std::size_t j = 0; j != Dimensions2; ++j){
							//~ BOOST_CHECK_EQUAL(data(i,j),truth(i,j));
							//~ BOOST_CHECK_EQUAL(dataColMajor(i,j),truth(i,j));
						//~ }
					//~ }
				//~ }
			//~ }
		//~ }
	//~ }
//~ }

BOOST_AUTO_TEST_CASE( Remora_Dense_row){
	for(std::size_t r = 0;r != Dimensions1;++r){
		vector<float> vTest(Dimensions2);
		for(std::size_t j = 0; j != Dimensions2; ++j)
			vTest(j) = denseData_cpu(r,j);
		checkDenseVectorEquality(row(denseData,r),vTest);
		
		//now test whether we can assign to a range like this.
		matrix<float, row_major, gpu_tag> newData(Dimensions1, Dimensions2,1.0);
		vector<float, gpu_tag> vTest_gpu = copy_to_gpu(vTest);
		auto rowTest = row(newData,r);
		noalias(rowTest) = vTest_gpu;
		//check that the assignment has been carried out correctly
		checkDenseVectorEquality(rowTest,vTest);

		//check that after assignment all elements outside the range are still intact
		matrix<float> truth(Dimensions1,Dimensions2,1.0);
		for(std::size_t j = 0; j != Dimensions2; ++j){
			truth(r,j) = denseData_cpu(r,j);
		}
		matrix<float> data = copy_to_cpu(newData);
		for(std::size_t i = 0; i != Dimensions1; ++i){
			for(std::size_t j = 0; j != Dimensions2; ++j){
				BOOST_CHECK_EQUAL(data(i,j),truth(i,j));
			}
		}
	}
}
BOOST_AUTO_TEST_CASE( Remora_Dense_column){
	for(std::size_t c = 0;c != Dimensions2;++c){
		vector<float> vTest(Dimensions1);
		for(std::size_t i = 0; i != Dimensions1; ++i)
			vTest(i) = denseData_cpu(i,c);
		checkDenseVectorEquality(column(denseData,c),vTest);
		
		//now test whether we can assign to a range like this.
		matrix<float, row_major, gpu_tag> newData(Dimensions1, Dimensions2,1.0);
		vector<float, gpu_tag> vTest_gpu = copy_to_gpu(vTest);
		auto columnTest = column(newData,c);
		noalias(columnTest) = vTest_gpu;
		//check that the assignment has been carried out correctly
		checkDenseVectorEquality(columnTest,vTest);

		//check that after assignment all elements outside the range are still intact
		matrix<float> truth(Dimensions1,Dimensions2,1.0);
		for(std::size_t i = 0; i != Dimensions1; ++i){
			truth(i,c) = denseData_cpu(i,c);
		}
		matrix<float> data = copy_to_cpu(newData);
		for(std::size_t i = 0; i != Dimensions1; ++i){
			for(std::size_t j = 0; j != Dimensions2; ++j){
				BOOST_CHECK_EQUAL(data(i,j),truth(i,j));
			}
		}
	}
}

BOOST_AUTO_TEST_CASE( Remora_Dense_diagonal){
	matrix<float, row_major, gpu_tag> square = subrange(denseData,0,Dimensions2,0,Dimensions2);
	vector<float> vTest(Dimensions2);
	for(std::size_t i = 0; i != Dimensions2; ++i)
		vTest(i) = denseData_cpu(i,i);
	checkDenseVectorEquality(diag(square),vTest);
	
	//now test whether we can assign to a range like this.
	matrix<float, row_major, gpu_tag> newData(Dimensions2, Dimensions2,1.0);
	vector<float, gpu_tag> vTest_gpu = copy_to_gpu(vTest);
	auto diagTest = diag(newData);
	noalias(diagTest) = vTest_gpu;
	//check that the assignment has been carried out correctly
	checkDenseVectorEquality(diagTest,vTest);

	//check that after assignment all elements outside the range are still intact
	matrix<float> truth(Dimensions1,Dimensions2,1.0);
	for(std::size_t i = 0; i != Dimensions2; ++i){
		truth(i,i) = denseData_cpu(i,i);
	}
	matrix<float> data = copy_to_cpu(newData);
	for(std::size_t i = 0; i != Dimensions2; ++i){
		for(std::size_t j = 0; j != Dimensions2; ++j){
			BOOST_CHECK_EQUAL(data(i,j),truth(i,j));
		}
	}
}

BOOST_AUTO_TEST_SUITE_END();