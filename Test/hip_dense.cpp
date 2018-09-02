#define BOOST_PP_VARIADICS 0
#define BOOST_TEST_MODULE Remora_HIP_Dense
#include <boost/test/unit_test.hpp>
#include <boost/test/floating_point_comparison.hpp>

#include <iostream>
#include <remora/dense.hpp>
#include <remora/io.hpp>
#include <remora/device_copy.hpp>
#include <boost/mpl/vector.hpp>

using namespace remora;

std::size_t Dimensions1 = 20;
std::size_t Dimensions2 = 10;
struct MatrixProxyFixture
{
	matrix<float> denseData_cpu;
	vector<float> denseData_cpu_vec;
	matrix<float, row_major, hip_tag> denseData;
	
	matrix<float, column_major, hip_tag> denseDataColMajor;
	vector<float, hip_tag> denseDataVec;
	MatrixProxyFixture():denseData_cpu(Dimensions1,Dimensions2),denseData_cpu_vec(Dimensions1){
		for(std::size_t row=0;row!= Dimensions1;++row){
			for(std::size_t col=0;col!=Dimensions2;++col){
				denseData_cpu(row,col) = row*Dimensions2+col+5.0;
			}
			denseData_cpu_vec(row) = row + 3.0;
		}
		denseData = copy_to_device(denseData_cpu, hip_tag());
		denseDataColMajor = copy_to_device(denseData_cpu, hip_tag());
		denseDataVec = copy_to_device(denseData_cpu_vec, hip_tag());
	}
};

BOOST_FIXTURE_TEST_SUITE (Remora_matrix_proxy, MatrixProxyFixture);

BOOST_AUTO_TEST_CASE( Vector_Proxy){
	vector<float, hip_tag> data = denseDataVec;
	auto storageVec = data.raw_storage();
	storageVec.stride = 2;
	
	dense_vector_adaptor<float, continuous_dense_tag, hip_tag> proxy(storageVec,denseDataVec.queue(), 6);
	BOOST_CHECK_EQUAL(proxy.size(),6);
	BOOST_CHECK_EQUAL(proxy.raw_storage().stride,storageVec.stride);
	BOOST_CHECK_EQUAL(proxy.raw_storage().values,storageVec.values);
	
	//test whether the referenced values are correct
	vector<float> op = copy_to_cpu(proxy);
	BOOST_REQUIRE_EQUAL(op.size(), 6);
	for(std::size_t i = 0; i != op.size(); ++i){
		BOOST_CHECK_EQUAL(denseData_cpu_vec(storageVec.stride*i), op(i));
	}
	
	//test whether clearing works
	proxy.clear();
	vector<float> result = denseData_cpu_vec;
	for(std::size_t i = 0; i != proxy.size(); ++i){
		result(storageVec.stride*i) = 0;
	}
	op = copy_to_cpu(data);
	for(std::size_t i = 0; i != data.size(); ++i){
		BOOST_CHECK_EQUAL(result(i), op(i));
	}
	
	//Test assignment
	{
		vector<float, hip_tag> dataConst(proxy.size(),1.0);
		for(std::size_t i = 0; i != proxy.size(); ++i){
			result(storageVec.stride*i) = 1;
		}
		proxy = dataConst;
		op = copy_to_cpu(data);
		for(std::size_t i = 0; i != data.size(); ++i){
			BOOST_CHECK_EQUAL(result(i), op(i));
		}
	}
	{
		vector<float, hip_tag> dataConst(2 * proxy.size(),3.0,data.queue());
		for(std::size_t i = 0; i != proxy.size(); ++i){
			result(storageVec.stride*i) = 3.0;
		}
		auto storage = dataConst.raw_storage();
		dense_vector_adaptor<float const, continuous_dense_tag, hip_tag> proxyTest(storage,data.queue(), proxy.size());
		proxy = proxyTest;
		op = copy_to_cpu(data);
		for(std::size_t i = 0; i != data.size(); ++i){
			BOOST_CHECK_EQUAL(result(i), op(i));
		}
	}
}

template<class Orientation>
void matrix_proxy_test(Orientation, matrix<float, row_major, hip_tag>& denseData, matrix<float, row_major>& denseData_cpu){
	matrix<float, Orientation, hip_tag> data = denseData;
	auto storage = data.raw_storage();
	storage.values += Orientation::element(3,2,storage.leading_dimension);
	dense_matrix_adaptor<float,Orientation, continuous_dense_tag, hip_tag> proxy(storage,data.queue(), Dimensions1 - 6, Dimensions2 - 4);
	BOOST_CHECK_EQUAL(proxy.size1(),Dimensions1 - 6);
	BOOST_CHECK_EQUAL(proxy.size2(),Dimensions2 - 4);
	BOOST_CHECK_EQUAL(proxy.raw_storage().leading_dimension,storage.leading_dimension);
	BOOST_CHECK_EQUAL(proxy.raw_storage().values,storage.values);
	
	//test whether the referenced values are correct
	matrix<float> op = copy_to_cpu(proxy);
	for(std::size_t i = 0; i != proxy.size1(); ++i){
		for(std::size_t j = 0; j != proxy.size2(); ++j){
			BOOST_CHECK_EQUAL(op(i,j),denseData_cpu(i + 3,j+2));
		}
	}
	
	//test whether clearing works
	proxy.clear();
	op = copy_to_cpu(data);
	matrix<float> result = denseData_cpu;
	for(std::size_t i = 0; i != proxy.size1(); ++i){
		for(std::size_t j = 0; j != proxy.size2(); ++j){
			result(i+3,j+2) = 0;
		}
	}
	for(std::size_t i = 0; i != data.size1(); ++i){
		for(std::size_t j = 0; j != data.size2(); ++j){
			BOOST_CHECK_EQUAL(result(i,j),op(i,j));
		}
	}
}
BOOST_AUTO_TEST_CASE( Matrix_Proxy){
	matrix_proxy_test(row_major(), denseData, denseData_cpu);
	matrix_proxy_test(column_major(), denseData, denseData_cpu);
}


//check that vectors are correctly transformed into their closures
BOOST_AUTO_TEST_CASE( Vector_Closure){
	auto storageVec = denseDataVec.raw_storage();
	BOOST_CHECK_EQUAL(storageVec.stride,1);
	{
		vector<float, hip_tag>::closure_type closure = denseDataVec;
		BOOST_CHECK_EQUAL(closure.size(),denseDataVec.size());
		auto storage = closure.raw_storage();
		BOOST_CHECK_EQUAL(storage.stride,1);
		BOOST_CHECK_EQUAL(storage.values,storageVec.values);
	}
	{
		vector<float, hip_tag>::const_closure_type closure = (vector<float, hip_tag> const&) denseDataVec;
		BOOST_CHECK_EQUAL(closure.size(),denseDataVec.size());
		auto storage = closure.raw_storage();
		BOOST_CHECK_EQUAL(storage.stride,1);
		BOOST_CHECK_EQUAL(storage.values,storageVec.values);
	}
	{
		vector<float, hip_tag>::const_closure_type closure = denseDataVec;
		BOOST_CHECK_EQUAL(closure.size(),denseDataVec.size());
		auto storage = closure.raw_storage();
		BOOST_CHECK_EQUAL(storage.stride,1);
		BOOST_CHECK_EQUAL(storage.values,storageVec.values);
	}
}
//check that matrices are correctly transformed into their closures
BOOST_AUTO_TEST_CASE( Matrix_Closure_Row_Major){
	auto storageR = denseData.raw_storage();
	BOOST_CHECK_EQUAL(storageR.leading_dimension,Dimensions2);
	
	{
		matrix<float, row_major, hip_tag>::closure_type closure = denseData;
		BOOST_CHECK_EQUAL(closure.size1(),denseData.size1());
		BOOST_CHECK_EQUAL(closure.size2(),denseData.size2());
		auto storage = closure.raw_storage();
		BOOST_CHECK_EQUAL(storage.leading_dimension,Dimensions2);
		BOOST_CHECK_EQUAL(storage.values,storageR.values);
	}
	{
		matrix<float, row_major, hip_tag>::const_closure_type closure = (matrix<float, row_major, hip_tag> const&)denseData;
		BOOST_CHECK_EQUAL(closure.size1(),denseData.size1());
		BOOST_CHECK_EQUAL(closure.size2(),denseData.size2());
		auto storage = closure.raw_storage();
		BOOST_CHECK_EQUAL(storage.leading_dimension,Dimensions2);
		BOOST_CHECK_EQUAL(storage.values,storageR.values);
	}
	{
		matrix<float, row_major, hip_tag>::const_closure_type closure = denseData;
		BOOST_CHECK_EQUAL(closure.size1(),denseData.size1());
		BOOST_CHECK_EQUAL(closure.size2(),denseData.size2());
		auto storage = closure.raw_storage();
		BOOST_CHECK_EQUAL(storage.leading_dimension,Dimensions2);
		BOOST_CHECK_EQUAL(storage.values,storageR.values);
	}
}

BOOST_AUTO_TEST_CASE( Matrix_Closure_Column_Major){
	auto storageC = denseDataColMajor.raw_storage();
	BOOST_CHECK_EQUAL(storageC.leading_dimension,Dimensions1);
	
	{
		matrix<float, column_major, hip_tag>::closure_type closure = denseDataColMajor;
		BOOST_CHECK_EQUAL(closure.size1(),denseDataColMajor.size1());
		BOOST_CHECK_EQUAL(closure.size2(),denseDataColMajor.size2());
		auto storage = closure.raw_storage();
		BOOST_CHECK_EQUAL(storage.leading_dimension,Dimensions1);
		BOOST_CHECK_EQUAL(storage.values,storageC.values);
	}
	{
		matrix<float, column_major, hip_tag>::const_closure_type closure = (matrix<float, column_major, hip_tag> const&)denseDataColMajor;
		BOOST_CHECK_EQUAL(closure.size1(),denseDataColMajor.size1());
		BOOST_CHECK_EQUAL(closure.size2(),denseDataColMajor.size2());
		auto storage = closure.raw_storage();
		BOOST_CHECK_EQUAL(storage.leading_dimension,Dimensions1);
		BOOST_CHECK_EQUAL(storage.values,storageC.values);
	}
	{
		matrix<float, column_major, hip_tag>::const_closure_type closure = denseDataColMajor;
		BOOST_CHECK_EQUAL(closure.size1(),denseDataColMajor.size1());
		BOOST_CHECK_EQUAL(closure.size2(),denseDataColMajor.size2());
		auto storage = closure.raw_storage();
		BOOST_CHECK_EQUAL(storage.leading_dimension,Dimensions1);
		BOOST_CHECK_EQUAL(storage.values,storageC.values);
	}
}


// vector subrange
BOOST_AUTO_TEST_CASE( Vector_Subrange){
	auto storageVec = denseDataVec.raw_storage();
	{
		vector<float, hip_tag>::closure_type closure = subrange(denseDataVec,3,7);
		BOOST_CHECK_EQUAL(closure.size(),4);
		auto storage = closure.raw_storage();
		BOOST_CHECK_EQUAL(storage.stride,1);
		BOOST_CHECK_EQUAL(storage.values,storageVec.values + 3);
	}
	{
		vector<float, hip_tag>::const_closure_type closure = subrange((vector<float, hip_tag> const&) denseDataVec, 3, 7);
		BOOST_CHECK_EQUAL(closure.size(),4);
		auto storage = closure.raw_storage();
		BOOST_CHECK_EQUAL(storage.stride,1);
		BOOST_CHECK_EQUAL(storage.values,storageVec.values + 3);
	}
	{
		vector<float, hip_tag>::const_closure_type closure = subrange(denseDataVec,3,7);
		BOOST_CHECK_EQUAL(closure.size(),4);
		auto storage = closure.raw_storage();
		BOOST_CHECK_EQUAL(storage.stride,1);
		BOOST_CHECK_EQUAL(storage.values,storageVec.values + 3);
	}
	//now also check from non-trivial proxy
	storageVec.stride = 2;
	vector<float, hip_tag>::closure_type proxy(storageVec, denseDataVec.queue(), (denseDataVec.size()-2)/2);
	vector<float, hip_tag>::const_closure_type const_proxy = proxy;
	{
		vector<float, hip_tag>::closure_type closure = subrange(proxy,3,7);
		BOOST_CHECK_EQUAL(closure.size(),4);
		auto storage = closure.raw_storage();
		BOOST_CHECK_EQUAL(storage.stride,2);
		BOOST_CHECK_EQUAL(storage.values,storageVec.values + 6);
	}
	{
		vector<float, hip_tag>::const_closure_type closure = subrange(const_proxy,3,7);
		BOOST_CHECK_EQUAL(closure.size(),4);
		auto storage = closure.raw_storage();
		BOOST_CHECK_EQUAL(storage.stride,2);
		BOOST_CHECK_EQUAL(storage.values,storageVec.values + 6);
	}
}

template<class Orientation>
void matrix_transpose_test(Orientation, matrix<float, row_major, hip_tag>& denseData, matrix<float, row_major>& denseData_cpu){
	matrix<float, Orientation, hip_tag> data = denseData;
	
	auto storageR = data.raw_storage();
	if(std::is_same<Orientation,column_major>::value){
		BOOST_REQUIRE_EQUAL(storageR.leading_dimension,Dimensions1);
	}else{
		BOOST_REQUIRE_EQUAL(storageR.leading_dimension,Dimensions2);
	}
	typedef typename Orientation::transposed_orientation Transposed;
	{
		typename matrix<float, Transposed, hip_tag>::closure_type closure = trans(data);
		BOOST_CHECK_EQUAL(closure.size1(),data.size2());
		BOOST_CHECK_EQUAL(closure.size2(),data.size1());
		auto storage = closure.raw_storage();
		BOOST_CHECK_EQUAL(storage.leading_dimension,storageR.leading_dimension);
		BOOST_CHECK_EQUAL(storage.values,storageR.values);
	}
	{
		typename matrix<float, Transposed, hip_tag>::const_closure_type closure = trans((matrix<float, Orientation, hip_tag> const&)data);
		BOOST_CHECK_EQUAL(closure.size1(),data.size2());
		BOOST_CHECK_EQUAL(closure.size2(),data.size1());
		auto storage = closure.raw_storage();
		BOOST_CHECK_EQUAL(storage.leading_dimension,storageR.leading_dimension);
		BOOST_CHECK_EQUAL(storage.values,storageR.values);
	}
	{
		typename matrix<float, Transposed, hip_tag>::const_closure_type closure = trans(data);
		BOOST_CHECK_EQUAL(closure.size1(),data.size2());
		BOOST_CHECK_EQUAL(closure.size2(),data.size1());
		auto storage = closure.raw_storage();
		BOOST_CHECK_EQUAL(storage.leading_dimension,storageR.leading_dimension);
		BOOST_CHECK_EQUAL(storage.values,storageR.values);
	}
}

BOOST_AUTO_TEST_CASE( Matrix_Transpose){
	matrix_transpose_test(row_major(), denseData, denseData_cpu);
	matrix_transpose_test(column_major(), denseData, denseData_cpu);
}

template<class Orientation>
void matrix_row_test(Orientation, matrix<float, row_major, hip_tag>& denseData, matrix<float, row_major>& denseData_cpu){
	matrix<float, Orientation, hip_tag> data = denseData;
	typedef typename std::conditional<
		std::is_same<Orientation,column_major>::value,
		dense_tag,
		continuous_dense_tag
	>::type Tag;
	std::size_t stride1 = Dimensions2;
	std::size_t stride2 = 1;
	if(std::is_same<Orientation,column_major>::value){
		stride1 = 1;
		stride2 = Dimensions1;
	}
	auto storageR = data.raw_storage();
	{
		dense_vector_adaptor<float, Tag, hip_tag> closure = row(data,3);
		BOOST_CHECK_EQUAL(closure.size(),data.size2());
		auto storage = closure.raw_storage();
		BOOST_CHECK_EQUAL(storage.stride,stride2);
		BOOST_CHECK_EQUAL(storage.values,storageR.values + stride1 * 3);
	}
	{
		dense_vector_adaptor<float const, Tag, hip_tag> closure = row((matrix<float, Orientation, hip_tag> const&)data,3);
		BOOST_CHECK_EQUAL(closure.size(),data.size2());
		auto storage = closure.raw_storage();
		BOOST_CHECK_EQUAL(storage.stride,stride2);
		BOOST_CHECK_EQUAL(storage.values,storageR.values + stride1 * 3);
	}
	{
		dense_vector_adaptor<float const, Tag, hip_tag> closure = row(data,3);
		BOOST_CHECK_EQUAL(closure.size(),data.size2());
		auto storage = closure.raw_storage();
		BOOST_CHECK_EQUAL(storage.stride,stride2);
		BOOST_CHECK_EQUAL(storage.values,storageR.values + stride1 * 3);
	}
}
BOOST_AUTO_TEST_CASE( Matrix_Row){
	matrix_row_test(row_major(), denseData, denseData_cpu);
	matrix_row_test(column_major(), denseData, denseData_cpu);
}

template<class Orientation>
void matrix_subrange_test(Orientation, matrix<float, row_major, hip_tag>& denseData, matrix<float, row_major>& denseData_cpu){
	matrix<float, Orientation, hip_tag> data = denseData;
	auto storageR = data.raw_storage();
	{
		dense_matrix_adaptor<float, Orientation, dense_tag, hip_tag> closure = subrange(data,2,5,3,8);
		BOOST_CHECK_EQUAL(closure.size1(),3);
		BOOST_CHECK_EQUAL(closure.size2(),5);
		auto storage = closure.raw_storage();
		BOOST_CHECK_EQUAL(storage.leading_dimension,storageR.leading_dimension);
		BOOST_CHECK_EQUAL(storage.values,storageR.values + Orientation::element(2, 3, storageR.leading_dimension));
	}
	{
		dense_matrix_adaptor<float const, Orientation, dense_tag, hip_tag> closure = subrange((matrix<float, Orientation, hip_tag> const&)data,2,5,3,8);
		BOOST_CHECK_EQUAL(closure.size1(),3);
		BOOST_CHECK_EQUAL(closure.size2(),5);
		auto storage = closure.raw_storage();
		BOOST_CHECK_EQUAL(storage.leading_dimension,storageR.leading_dimension);
		BOOST_CHECK_EQUAL(storage.values,storageR.values + Orientation::element(2, 3, storageR.leading_dimension));
	}
	{
		dense_matrix_adaptor<float const, Orientation, dense_tag, hip_tag> closure = subrange(data,2,5,3,8);
		BOOST_CHECK_EQUAL(closure.size1(),3);
		BOOST_CHECK_EQUAL(closure.size2(),5);
		auto storage = closure.raw_storage();
		BOOST_CHECK_EQUAL(storage.leading_dimension,storageR.leading_dimension);
		BOOST_CHECK_EQUAL(storage.values,storageR.values + Orientation::element(2, 3, storageR.leading_dimension));
	}
}
BOOST_AUTO_TEST_CASE( Matrix_Subrange){
	matrix_subrange_test(row_major(), denseData, denseData_cpu);
	matrix_subrange_test(column_major(), denseData, denseData_cpu);
}

template<class Orientation>
void matrix_rows_test(Orientation, matrix<float, row_major, hip_tag>& denseData, matrix<float, row_major>& denseData_cpu){
	matrix<float, Orientation, hip_tag> data = denseData;
	typedef typename std::conditional<
		std::is_same<Orientation,column_major>::value,
		dense_tag,
		continuous_dense_tag
	>::type Tag;
	
	auto storageR = data.raw_storage();
	storageR.values += Orientation::element(2, 0, storageR.leading_dimension);
	{
		dense_matrix_adaptor<float, Orientation, Tag, hip_tag> closure = rows(data,2,5);
		BOOST_CHECK_EQUAL(closure.size1(),3);
		BOOST_CHECK_EQUAL(closure.size2(),data.size2());
		auto storage = closure.raw_storage();
		BOOST_CHECK_EQUAL(storage.leading_dimension,storageR.leading_dimension);
		BOOST_CHECK_EQUAL(storage.values,storageR.values);
	}
	{
		dense_matrix_adaptor<float const, Orientation, Tag, hip_tag> closure = rows((matrix<float, Orientation, hip_tag> const&)data,2,5);
		BOOST_CHECK_EQUAL(closure.size1(),3);
		BOOST_CHECK_EQUAL(closure.size2(),data.size2());
		auto storage = closure.raw_storage();
		BOOST_CHECK_EQUAL(storage.leading_dimension,storageR.leading_dimension);
		BOOST_CHECK_EQUAL(storage.values,storageR.values);
	}
	{
		dense_matrix_adaptor<float const, Orientation, Tag, hip_tag> closure = rows(data,2,5);
		BOOST_CHECK_EQUAL(closure.size1(),3);
		BOOST_CHECK_EQUAL(closure.size2(),data.size2());
		auto storage = closure.raw_storage();
		BOOST_CHECK_EQUAL(storage.leading_dimension,storageR.leading_dimension);
		BOOST_CHECK_EQUAL(storage.values,storageR.values);
	}
	
	//also check from dense_tag
	{
		dense_matrix_adaptor<float, Orientation, dense_tag, hip_tag> closure = rows(dense_matrix_adaptor<float, Orientation, dense_tag, hip_tag>(data),2,5);
		BOOST_CHECK_EQUAL(closure.size1(),3);
		BOOST_CHECK_EQUAL(closure.size2(),data.size2());
		auto storage = closure.raw_storage();
		BOOST_CHECK_EQUAL(storage.leading_dimension,storageR.leading_dimension);
		BOOST_CHECK_EQUAL(storage.values,storageR.values);
	}
	{
		dense_matrix_adaptor<float const, Orientation, dense_tag, hip_tag> closure = rows(dense_matrix_adaptor<float const, Orientation, dense_tag, hip_tag>(data),2,5);
		BOOST_CHECK_EQUAL(closure.size1(),3);
		BOOST_CHECK_EQUAL(closure.size2(),data.size2());
		auto storage = closure.raw_storage();
		BOOST_CHECK_EQUAL(storage.leading_dimension,storageR.leading_dimension);
		BOOST_CHECK_EQUAL(storage.values,storageR.values);
	}
}

BOOST_AUTO_TEST_CASE( Matrix_Rows){
	matrix_rows_test(row_major(), denseData, denseData_cpu);
	matrix_rows_test(column_major(), denseData, denseData_cpu);
}

template<class Orientation>
void matrix_diagonal_test(Orientation, matrix<float, row_major, hip_tag>& denseData, matrix<float, row_major>& denseData_cpu){
	matrix<float, Orientation, hip_tag> data = subrange(denseData,0,Dimensions2,0,Dimensions2);
	auto storageR = data.raw_storage();
	{
		dense_vector_adaptor<float, dense_tag, hip_tag> closure = diag(data);
		BOOST_CHECK_EQUAL(closure.size(),data.size2());
		auto storage = closure.raw_storage();
		BOOST_CHECK_EQUAL(storage.stride,storageR.leading_dimension+1);
		BOOST_CHECK_EQUAL(storage.values,storageR.values);
	}
	{
		dense_vector_adaptor<float const, dense_tag, hip_tag> closure = diag((matrix<float, Orientation, hip_tag> const&)data);
		BOOST_CHECK_EQUAL(closure.size(),data.size2());
		auto storage = closure.raw_storage();
		BOOST_CHECK_EQUAL(storage.stride,storageR.leading_dimension+1);
		BOOST_CHECK_EQUAL(storage.values,storageR.values);
	}
	{
		dense_vector_adaptor<float const, dense_tag, hip_tag> closure = diag(data);
		BOOST_CHECK_EQUAL(closure.size(),data.size2());
		auto storage = closure.raw_storage();
		BOOST_CHECK_EQUAL(storage.stride,storageR.leading_dimension+1);
		BOOST_CHECK_EQUAL(storage.values,storageR.values);
	}
}

BOOST_AUTO_TEST_CASE( Matrix_Diagonal){
	matrix_diagonal_test(row_major(), denseData, denseData_cpu);
	matrix_diagonal_test(column_major(), denseData, denseData_cpu);
}

template<class Orientation>
void matrix_to_vector_test(Orientation, matrix<float, row_major, hip_tag>& denseData, matrix<float, row_major>& denseData_cpu){
	matrix<float, Orientation, hip_tag> data = denseData;
	auto storageR = data.raw_storage();
	{
		dense_vector_adaptor<float, continuous_dense_tag, hip_tag> closure = to_vector(data);
		BOOST_CHECK_EQUAL(closure.size(), data.size1() * data.size2());
		auto storage = closure.raw_storage();
		BOOST_CHECK_EQUAL(storage.stride,1);
		BOOST_CHECK_EQUAL(storage.values,storageR.values);
	}
	{
		dense_vector_adaptor<float const, continuous_dense_tag, hip_tag> closure = to_vector((matrix<float, Orientation, hip_tag> const&)data);
		BOOST_CHECK_EQUAL(closure.size(), data.size1() * data.size2());
		auto storage = closure.raw_storage();
		BOOST_CHECK_EQUAL(storage.stride,1);
		BOOST_CHECK_EQUAL(storage.values,storageR.values);
	}
	{
		dense_vector_adaptor<float const, continuous_dense_tag, hip_tag> closure = to_vector(data);
		BOOST_CHECK_EQUAL(closure.size(), data.size1() * data.size2());
		auto storage = closure.raw_storage();
		BOOST_CHECK_EQUAL(storage.stride,1);
		BOOST_CHECK_EQUAL(storage.values,storageR.values);
	}
}

BOOST_AUTO_TEST_CASE( Matrix_To_Vector){
	matrix_to_vector_test(row_major(), denseData, denseData_cpu);
	matrix_to_vector_test(column_major(), denseData, denseData_cpu);
}


BOOST_AUTO_TEST_CASE( Vector_To_Matrix){
	auto storageR = denseDataVec.raw_storage();

	{
		dense_matrix_adaptor<float, row_major, continuous_dense_tag, hip_tag> closure = to_matrix(denseDataVec,4,5);
		BOOST_CHECK_EQUAL(closure.size1(),4);
		BOOST_CHECK_EQUAL(closure.size2(),5);
		auto storage = closure.raw_storage();
		BOOST_CHECK_EQUAL(storage.leading_dimension,5);
		BOOST_CHECK_EQUAL(storage.values,storageR.values);
	}
	{
		dense_matrix_adaptor<float const, row_major, continuous_dense_tag, hip_tag> closure = to_matrix((remora::vector<float,hip_tag> const&)denseDataVec,4,5);
		BOOST_CHECK_EQUAL(closure.size1(),4);
		BOOST_CHECK_EQUAL(closure.size2(),5);
		auto storage = closure.raw_storage();
		BOOST_CHECK_EQUAL(storage.leading_dimension,5);
		BOOST_CHECK_EQUAL(storage.values,storageR.values);
	}
	{
		dense_matrix_adaptor<float const, row_major, continuous_dense_tag, hip_tag> closure = to_matrix(denseDataVec,4,5);
		BOOST_CHECK_EQUAL(closure.size1(),4);
		BOOST_CHECK_EQUAL(closure.size2(),5);
		auto storage = closure.raw_storage();
		BOOST_CHECK_EQUAL(storage.leading_dimension,5);
		BOOST_CHECK_EQUAL(storage.values,storageR.values);
	}
	//also check rvalue version
	{
		dense_matrix_adaptor<float, row_major, continuous_dense_tag, hip_tag> closure = to_matrix(dense_vector_adaptor<float, continuous_dense_tag, hip_tag>(denseDataVec),2,10);
		BOOST_CHECK_EQUAL(closure.size1(),2);
		BOOST_CHECK_EQUAL(closure.size2(),10);
		auto storage = closure.raw_storage();
		BOOST_CHECK_EQUAL(storage.leading_dimension,10);
		BOOST_CHECK_EQUAL(storage.values,storageR.values);
	}
	{
		dense_matrix_adaptor<float const, row_major, continuous_dense_tag, hip_tag> closure = to_matrix(dense_vector_adaptor<float const, continuous_dense_tag, hip_tag>(denseDataVec),2,10);
		BOOST_CHECK_EQUAL(closure.size1(),2);
		BOOST_CHECK_EQUAL(closure.size2(),10);
		auto storage = closure.raw_storage();
		BOOST_CHECK_EQUAL(storage.leading_dimension,10);
		BOOST_CHECK_EQUAL(storage.values,storageR.values);
	}
}


BOOST_AUTO_TEST_SUITE_END();