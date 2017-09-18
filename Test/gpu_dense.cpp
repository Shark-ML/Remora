#define BOOST_TEST_MODULE Remora_GPU_MatrixProxy
#include <boost/test/unit_test.hpp>
#include <boost/test/floating_point_comparison.hpp>

#include <remora/dense.hpp>
#include <remora/device_copy.hpp>
#include <boost/mpl/list.hpp>

using namespace remora;

std::size_t Dimensions1 = 20;
std::size_t Dimensions2 = 10;
struct MatrixProxyFixture
{
	matrix<float> denseData_cpu;
	vector<float> denseData_cpu_vec;
	matrix<float, row_major, gpu_tag> denseData;
	
	matrix<float, column_major, gpu_tag> denseDataColMajor;
	vector<float, gpu_tag> denseDataVec;
	MatrixProxyFixture():denseData_cpu(Dimensions1,Dimensions2),denseData_cpu_vec(Dimensions1){
		for(std::size_t row=0;row!= Dimensions1;++row){
			for(std::size_t col=0;col!=Dimensions2;++col){
				denseData_cpu(row,col) = row*Dimensions2+col+5.0;
			}
		}
		denseData = copy_to_gpu(denseData_cpu);
		denseDataColMajor = copy_to_gpu(denseData_cpu);
		denseDataVec = copy_to_gpu(denseData_cpu_vec);
	}
};

BOOST_FIXTURE_TEST_SUITE (Remora_matrix_proxy, MatrixProxyFixture);

BOOST_AUTO_TEST_CASE( Vector_Proxy){
	vector<float, gpu_tag> data = denseDataVec;
	auto storageVec = data.raw_storage();
	storageVec.offset = 3;
	storageVec.stride = 2;
	
	dense_vector_adaptor<float, continuous_dense_tag, gpu_tag> proxy(storageVec,denseDataVec.queue(), 6);
	BOOST_CHECK_EQUAL(proxy.size(),6);
	BOOST_CHECK_EQUAL(proxy.raw_storage().stride,storageVec.stride);
	BOOST_CHECK_EQUAL(proxy.raw_storage().offset,storageVec.offset);
	BOOST_CHECK_EQUAL(proxy.raw_storage().buffer.get(),storageVec.buffer.get());
	BOOST_CHECK_EQUAL(proxy.queue().get(),data.queue().get());
	
	//test whether the referenced values are correct
	vector<float> op = copy_to_cpu(proxy);
	BOOST_REQUIRE_EQUAL(op.size(), 6);
	for(std::size_t i = 0; i != op.size(); ++i){
		BOOST_CHECK_EQUAL(denseData_cpu_vec(storageVec.offset+storageVec.stride*i), op(i));
	}
	
	//test whether clearing works
	proxy.clear();
	vector<float> result = denseData_cpu_vec;
	for(std::size_t i = 0; i != proxy.size(); ++i){
		result(storageVec.offset+storageVec.stride*i) = 0;
	}
	op = copy_to_cpu(data);
	for(std::size_t i = 0; i != data.size(); ++i){
		BOOST_CHECK_EQUAL(result(i), op(i));
	}
	
	//Test assignment
	{
		vector<float, gpu_tag> dataConst(proxy.size(),1.0);
		for(std::size_t i = 0; i != proxy.size(); ++i){
			result(storageVec.offset+storageVec.stride*i) = 1;
		}
		proxy = dataConst;
		op = copy_to_cpu(data);
		for(std::size_t i = 0; i != data.size(); ++i){
			BOOST_CHECK_EQUAL(result(i), op(i));
		}
	}
	{
		vector<float, gpu_tag> dataConst(2 * proxy.size(),3.0,data.queue());
		for(std::size_t i = 0; i != proxy.size(); ++i){
			result(storageVec.offset+storageVec.stride*i) = 3.0;
		}
		auto storage = dataConst.raw_storage();
		storage.offset = 5;
		dense_vector_adaptor<float const, continuous_dense_tag, gpu_tag> proxyTest(storage,data.queue(), proxy.size());
		proxy = proxyTest;
		op = copy_to_cpu(data);
		for(std::size_t i = 0; i != data.size(); ++i){
			BOOST_CHECK_EQUAL(result(i), op(i));
		}
	}
}

typedef boost::mpl::list<row_major,column_major> result_orientations;
BOOST_AUTO_TEST_CASE_TEMPLATE( Matrix_Proxy, Orientation, result_orientations){
	matrix<float, Orientation, gpu_tag> data = denseData;
	auto storage = data.raw_storage();
	storage.offset = Orientation::stride1(storage.leading_dimension) * 3+ Orientation::stride2(storage.leading_dimension) * 2;
	
	dense_matrix_adaptor<float,Orientation, continuous_dense_tag, gpu_tag> proxy(storage,data.queue(), Dimensions1 - 6, Dimensions2 - 4);
	BOOST_CHECK_EQUAL(proxy.size1(),Dimensions1 - 6);
	BOOST_CHECK_EQUAL(proxy.size2(),Dimensions2 - 4);
	BOOST_CHECK_EQUAL(proxy.raw_storage().leading_dimension,storage.leading_dimension);
	BOOST_CHECK_EQUAL(proxy.raw_storage().offset,storage.offset);
	BOOST_CHECK_EQUAL(proxy.raw_storage().buffer.get(),storage.buffer.get());
	BOOST_CHECK_EQUAL(proxy.queue().get(),data.queue().get());
	
	//test whether the referenced values are correct
	matrix<float> op = copy_to_cpu(proxy);
	for(std::size_t i = 0; i != proxy.size1(); ++i){
		for(std::size_t j = 0; j != proxy.size2(); ++j){
			BOOST_CHECK_EQUAL(op(i,j),denseData_cpu(i+3,j+2));
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


//check that vectors are correctly transformed into their closures
BOOST_AUTO_TEST_CASE( Vector_Closure){
	auto storageVec = denseDataVec.raw_storage();
	BOOST_CHECK_EQUAL(storageVec.offset,0);
	BOOST_CHECK_EQUAL(storageVec.stride,1);
	{
		vector<float, gpu_tag>::closure_type closure = denseDataVec;
		BOOST_CHECK_EQUAL(closure.size(),denseDataVec.size());
		auto storage = closure.raw_storage();
		BOOST_CHECK_EQUAL(storage.offset,0);
		BOOST_CHECK_EQUAL(storage.stride,1);
		BOOST_CHECK_EQUAL(storage.buffer.get(),storageVec.buffer.get());
	}
	{
		vector<float, gpu_tag>::const_closure_type closure = (vector<float, gpu_tag> const&) denseDataVec;
		BOOST_CHECK_EQUAL(closure.size(),denseDataVec.size());
		auto storage = closure.raw_storage();
		BOOST_CHECK_EQUAL(storage.offset,0);
		BOOST_CHECK_EQUAL(storage.stride,1);
		BOOST_CHECK_EQUAL(storage.buffer.get(),storageVec.buffer.get());
	}
	{
		vector<float, gpu_tag>::const_closure_type closure = denseDataVec;
		BOOST_CHECK_EQUAL(closure.size(),denseDataVec.size());
		auto storage = closure.raw_storage();
		BOOST_CHECK_EQUAL(storage.offset,0);
		BOOST_CHECK_EQUAL(storage.stride,1);
		BOOST_CHECK_EQUAL(storage.buffer.get(),storageVec.buffer.get());
	}
}
//check that matrices are correctly transformed into their closures
BOOST_AUTO_TEST_CASE( Matrix_Closure_Row_Major){
	auto storageR = denseData.raw_storage();
	BOOST_CHECK_EQUAL(storageR.offset,0);
	BOOST_CHECK_EQUAL(storageR.leading_dimension,Dimensions2);
	
	{
		matrix<float, row_major, gpu_tag>::closure_type closure = denseData;
		BOOST_CHECK_EQUAL(closure.size1(),denseData.size1());
		BOOST_CHECK_EQUAL(closure.size2(),denseData.size2());
		auto storage = closure.raw_storage();
		BOOST_CHECK_EQUAL(storage.offset,0);
		BOOST_CHECK_EQUAL(storage.leading_dimension,Dimensions2);
		BOOST_CHECK_EQUAL(storage.buffer.get(),storageR.buffer.get());
	}
	{
		matrix<float, row_major, gpu_tag>::const_closure_type closure = (matrix<float, row_major, gpu_tag> const&)denseData;
		BOOST_CHECK_EQUAL(closure.size1(),denseData.size1());
		BOOST_CHECK_EQUAL(closure.size2(),denseData.size2());
		auto storage = closure.raw_storage();
		BOOST_CHECK_EQUAL(storage.offset,0);
		BOOST_CHECK_EQUAL(storage.leading_dimension,Dimensions2);
		BOOST_CHECK_EQUAL(storage.buffer.get(),storageR.buffer.get());
	}
	{
		matrix<float, row_major, gpu_tag>::const_closure_type closure = denseData;
		BOOST_CHECK_EQUAL(closure.size1(),denseData.size1());
		BOOST_CHECK_EQUAL(closure.size2(),denseData.size2());
		auto storage = closure.raw_storage();
		BOOST_CHECK_EQUAL(storage.offset,0);
		BOOST_CHECK_EQUAL(storage.leading_dimension,Dimensions2);
		BOOST_CHECK_EQUAL(storage.buffer.get(),storageR.buffer.get());
	}
}

BOOST_AUTO_TEST_CASE( Matrix_Closure_Column_Major){
	auto storageC = denseDataColMajor.raw_storage();
	BOOST_CHECK_EQUAL(storageC.offset,0);
	BOOST_CHECK_EQUAL(storageC.leading_dimension,Dimensions1);
	
	{
		matrix<float, column_major, gpu_tag>::closure_type closure = denseDataColMajor;
		BOOST_CHECK_EQUAL(closure.size1(),denseDataColMajor.size1());
		BOOST_CHECK_EQUAL(closure.size2(),denseDataColMajor.size2());
		auto storage = closure.raw_storage();
		BOOST_CHECK_EQUAL(storage.offset,0);
		BOOST_CHECK_EQUAL(storage.leading_dimension,Dimensions1);
		BOOST_CHECK_EQUAL(storage.buffer.get(),storageC.buffer.get());
	}
	{
		matrix<float, column_major, gpu_tag>::const_closure_type closure = (matrix<float, column_major, gpu_tag> const&)denseDataColMajor;
		BOOST_CHECK_EQUAL(closure.size1(),denseDataColMajor.size1());
		BOOST_CHECK_EQUAL(closure.size2(),denseDataColMajor.size2());
		auto storage = closure.raw_storage();
		BOOST_CHECK_EQUAL(storage.offset,0);
		BOOST_CHECK_EQUAL(storage.leading_dimension,Dimensions1);
		BOOST_CHECK_EQUAL(storage.buffer.get(),storageC.buffer.get());
	}
	{
		matrix<float, column_major, gpu_tag>::const_closure_type closure = denseDataColMajor;
		BOOST_CHECK_EQUAL(closure.size1(),denseDataColMajor.size1());
		BOOST_CHECK_EQUAL(closure.size2(),denseDataColMajor.size2());
		auto storage = closure.raw_storage();
		BOOST_CHECK_EQUAL(storage.offset,0);
		BOOST_CHECK_EQUAL(storage.leading_dimension,Dimensions1);
		BOOST_CHECK_EQUAL(storage.buffer.get(),storageC.buffer.get());
	}
}


// vector subrange
BOOST_AUTO_TEST_CASE( Vector_Subrange){
	auto storageVec = denseDataVec.raw_storage();
	{
		vector<float, gpu_tag>::closure_type closure = subrange(denseDataVec,3,7);
		BOOST_CHECK_EQUAL(closure.size(),4);
		auto storage = closure.raw_storage();
		BOOST_CHECK_EQUAL(storage.offset,3);
		BOOST_CHECK_EQUAL(storage.stride,1);
		BOOST_CHECK_EQUAL(storage.buffer.get(),storageVec.buffer.get());
	}
	{
		vector<float, gpu_tag>::const_closure_type closure = subrange((vector<float, gpu_tag> const&) denseDataVec, 3, 7);
		BOOST_CHECK_EQUAL(closure.size(),4);
		auto storage = closure.raw_storage();
		BOOST_CHECK_EQUAL(storage.offset,3);
		BOOST_CHECK_EQUAL(storage.stride,1);
		BOOST_CHECK_EQUAL(storage.buffer.get(),storageVec.buffer.get());
	}
	{
		vector<float, gpu_tag>::const_closure_type closure = subrange(denseDataVec,3,7);
		BOOST_CHECK_EQUAL(closure.size(),4);
		auto storage = closure.raw_storage();
		BOOST_CHECK_EQUAL(storage.offset,3);
		BOOST_CHECK_EQUAL(storage.stride,1);
		BOOST_CHECK_EQUAL(storage.buffer.get(),storageVec.buffer.get());
	}
	//now also check from non-trivial proxy
	storageVec.offset = 2;
	storageVec.stride = 2;
	vector<float, gpu_tag>::closure_type proxy(storageVec, denseDataVec.queue(), (denseDataVec.size()-2)/2);
	vector<float, gpu_tag>::const_closure_type const_proxy = proxy;
	{
		vector<float, gpu_tag>::closure_type closure = subrange(proxy,3,7);
		BOOST_CHECK_EQUAL(closure.size(),4);
		auto storage = closure.raw_storage();
		BOOST_CHECK_EQUAL(storage.offset,8);
		BOOST_CHECK_EQUAL(storage.stride,2);
		BOOST_CHECK_EQUAL(storage.buffer.get(),storageVec.buffer.get());
	}
	{
		vector<float, gpu_tag>::const_closure_type closure = subrange(const_proxy,3,7);
		BOOST_CHECK_EQUAL(closure.size(),4);
		auto storage = closure.raw_storage();
		BOOST_CHECK_EQUAL(storage.offset,8);
		BOOST_CHECK_EQUAL(storage.stride,2);
		BOOST_CHECK_EQUAL(storage.buffer.get(),storageVec.buffer.get());
	}
	
	//also check rvalue version
	{
		vector<float, gpu_tag>::closure_type closure = subrange(vector<float, gpu_tag>::closure_type(proxy),3,7);
		BOOST_CHECK_EQUAL(closure.size(),4);
		auto storage = closure.raw_storage();
		BOOST_CHECK_EQUAL(storage.offset,8);
		BOOST_CHECK_EQUAL(storage.stride,2);
		BOOST_CHECK_EQUAL(storage.buffer.get(),storageVec.buffer.get());
	}
	{
		vector<float, gpu_tag>::const_closure_type closure = subrange(vector<float, gpu_tag>::const_closure_type(proxy),3,7);
		BOOST_CHECK_EQUAL(closure.size(),4);
		auto storage = closure.raw_storage();
		BOOST_CHECK_EQUAL(storage.offset,8);
		BOOST_CHECK_EQUAL(storage.stride,2);
		BOOST_CHECK_EQUAL(storage.buffer.get(),storageVec.buffer.get());
	}
}
BOOST_AUTO_TEST_CASE_TEMPLATE( Matrix_Transpose, Orientation, result_orientations){
	matrix<float, Orientation, gpu_tag> data = denseData;
	
	auto storageR = data.raw_storage();
	if(std::is_same<Orientation,column_major>::value){
		BOOST_REQUIRE_EQUAL(storageR.leading_dimension,Dimensions1);
	}else{
		BOOST_REQUIRE_EQUAL(storageR.leading_dimension,Dimensions2);
	}
	typedef typename Orientation::transposed_orientation Transposed;
	{
		typename matrix<float, Transposed, gpu_tag>::closure_type closure = trans(data);
		BOOST_CHECK_EQUAL(closure.size1(),data.size2());
		BOOST_CHECK_EQUAL(closure.size2(),data.size1());
		auto storage = closure.raw_storage();
		BOOST_CHECK_EQUAL(storage.offset,0);
		BOOST_CHECK_EQUAL(storage.leading_dimension,storageR.leading_dimension);
		BOOST_CHECK_EQUAL(storage.buffer.get(),storageR.buffer.get());
	}
	{
		typename matrix<float, Transposed, gpu_tag>::const_closure_type closure = trans((matrix<float, Orientation, gpu_tag> const&)data);
		BOOST_CHECK_EQUAL(closure.size1(),data.size2());
		BOOST_CHECK_EQUAL(closure.size2(),data.size1());
		auto storage = closure.raw_storage();
		BOOST_CHECK_EQUAL(storage.offset,0);
		BOOST_CHECK_EQUAL(storage.leading_dimension,storageR.leading_dimension);
		BOOST_CHECK_EQUAL(storage.buffer.get(),storageR.buffer.get());
	}
	{
		typename matrix<float, Transposed, gpu_tag>::const_closure_type closure = trans(data);
		BOOST_CHECK_EQUAL(closure.size1(),data.size2());
		BOOST_CHECK_EQUAL(closure.size2(),data.size1());
		auto storage = closure.raw_storage();
		BOOST_CHECK_EQUAL(storage.offset,0);
		BOOST_CHECK_EQUAL(storage.leading_dimension,storageR.leading_dimension);
		BOOST_CHECK_EQUAL(storage.buffer.get(),storageR.buffer.get());
	}
	
	storageR.offset = 2* storageR.leading_dimension;
	typename matrix<float, Orientation, gpu_tag>::closure_type proxy(storageR, data.queue(), Dimensions1-2,Dimensions2-2);
	typename matrix<float, Orientation, gpu_tag>::const_closure_type const_proxy = proxy;
	{
		typename matrix<float, Transposed, gpu_tag>::closure_type closure = trans(proxy);
		BOOST_CHECK_EQUAL(closure.size1(),proxy.size2());
		BOOST_CHECK_EQUAL(closure.size2(),proxy.size1());
		auto storage = closure.raw_storage();
		BOOST_CHECK_EQUAL(storage.offset,storageR.offset);
		BOOST_CHECK_EQUAL(storage.leading_dimension,storageR.leading_dimension);
		BOOST_CHECK_EQUAL(storage.buffer.get(),storageR.buffer.get());
	}
	{
		typename matrix<float, Transposed, gpu_tag>::const_closure_type closure = trans(const_proxy);
		BOOST_CHECK_EQUAL(closure.size1(),proxy.size2());
		BOOST_CHECK_EQUAL(closure.size2(),proxy.size1());
		auto storage = closure.raw_storage();
		BOOST_CHECK_EQUAL(storage.offset,storageR.offset);
		BOOST_CHECK_EQUAL(storage.leading_dimension,storageR.leading_dimension);
		BOOST_CHECK_EQUAL(storage.buffer.get(),storageR.buffer.get());
	}
	
	//also check rvalue version
	{
		typename matrix<float, Transposed, gpu_tag>::closure_type closure = trans(typename matrix<float, Orientation, gpu_tag>::closure_type(proxy));
		BOOST_CHECK_EQUAL(closure.size1(),proxy.size2());
		BOOST_CHECK_EQUAL(closure.size2(),proxy.size1());
		auto storage = closure.raw_storage();
		BOOST_CHECK_EQUAL(storage.offset,storageR.offset);
		BOOST_CHECK_EQUAL(storage.leading_dimension,storageR.leading_dimension);
		BOOST_CHECK_EQUAL(storage.buffer.get(),storageR.buffer.get());
	}
	{
		typename matrix<float, Transposed, gpu_tag>::const_closure_type closure = trans(typename matrix<float, Orientation, gpu_tag>::const_closure_type(proxy));
		BOOST_CHECK_EQUAL(closure.size1(),proxy.size2());
		BOOST_CHECK_EQUAL(closure.size2(),proxy.size1());
		auto storage = closure.raw_storage();
		BOOST_CHECK_EQUAL(storage.offset,storageR.offset);
		BOOST_CHECK_EQUAL(storage.leading_dimension,storageR.leading_dimension);
		BOOST_CHECK_EQUAL(storage.buffer.get(),storageR.buffer.get());
	}
}

BOOST_AUTO_TEST_CASE_TEMPLATE( Matrix_Row, Orientation, result_orientations){
	matrix<float, Orientation, gpu_tag> data = denseData;
	typedef typename std::conditional<
		std::is_same<Orientation,column_major>::value,
		dense_tag,
		continuous_dense_tag
	>::type Tag;
	auto storageR = data.raw_storage();
	std::size_t stride1 = Dimensions2;
	std::size_t stride2 = 1;
	if(std::is_same<Orientation,column_major>::value){
		stride1 = 1;
		stride2 = Dimensions1;
	}
	
	{
		dense_vector_adaptor<float, Tag, gpu_tag> closure = row(data,3);
		BOOST_CHECK_EQUAL(closure.size(),data.size2());
		auto storage = closure.raw_storage();
		BOOST_CHECK_EQUAL(storage.offset,3*stride1);
		BOOST_CHECK_EQUAL(storage.stride,stride2);
		BOOST_CHECK_EQUAL(storage.buffer.get(),storageR.buffer.get());
	}
	{
		dense_vector_adaptor<float const, Tag, gpu_tag> closure = row((matrix<float, Orientation, gpu_tag> const&)data,3);
		BOOST_CHECK_EQUAL(closure.size(),data.size2());
		auto storage = closure.raw_storage();
		BOOST_CHECK_EQUAL(storage.offset,3*stride1);
		BOOST_CHECK_EQUAL(storage.stride,stride2);
		BOOST_CHECK_EQUAL(storage.buffer.get(),storageR.buffer.get());
	}
	{
		dense_vector_adaptor<float const, Tag, gpu_tag> closure = row(data,3);
		BOOST_CHECK_EQUAL(closure.size(),data.size2());
		auto storage = closure.raw_storage();
		BOOST_CHECK_EQUAL(storage.offset,3*stride1);
		BOOST_CHECK_EQUAL(storage.stride,stride2);
		BOOST_CHECK_EQUAL(storage.buffer.get(),storageR.buffer.get());
	}
	
	
	storageR.offset = 2*stride1;
	typename matrix<float, Orientation, gpu_tag>::closure_type proxy(storageR, data.queue(), Dimensions1-2,Dimensions2);
	typename matrix<float, Orientation, gpu_tag>::const_closure_type const_proxy = proxy;
	{
		dense_vector_adaptor<float, Tag, gpu_tag> closure = row(proxy,3);
		BOOST_CHECK_EQUAL(closure.size(),data.size2());
		auto storage = closure.raw_storage();
		BOOST_CHECK_EQUAL(storage.offset,5*stride1);
		BOOST_CHECK_EQUAL(storage.stride,stride2);
		BOOST_CHECK_EQUAL(storage.buffer.get(),storageR.buffer.get());
	}
	{
		dense_vector_adaptor<float const, Tag, gpu_tag> closure = row(const_proxy,3);
		BOOST_CHECK_EQUAL(closure.size(),data.size2());
		auto storage = closure.raw_storage();
		BOOST_CHECK_EQUAL(storage.offset,5*stride1);
		BOOST_CHECK_EQUAL(storage.stride,stride2);
		BOOST_CHECK_EQUAL(storage.buffer.get(),storageR.buffer.get());
	}
	
	//also check rvalue version
	{
		dense_vector_adaptor<float, Tag, gpu_tag> closure = row(typename matrix<float, Orientation, gpu_tag>::closure_type(proxy),3);
		BOOST_CHECK_EQUAL(closure.size(),data.size2());
		auto storage = closure.raw_storage();
		BOOST_CHECK_EQUAL(storage.offset,5*stride1);
		BOOST_CHECK_EQUAL(storage.stride,stride2);
		BOOST_CHECK_EQUAL(storage.buffer.get(),storageR.buffer.get());
	}
	{
		dense_vector_adaptor<float const, Tag, gpu_tag> closure = row(typename matrix<float, Orientation, gpu_tag>::const_closure_type(proxy),3);
		BOOST_CHECK_EQUAL(closure.size(),data.size2());
		auto storage = closure.raw_storage();
		BOOST_CHECK_EQUAL(storage.offset,5*stride1);
		BOOST_CHECK_EQUAL(storage.stride,stride2);
		BOOST_CHECK_EQUAL(storage.buffer.get(),storageR.buffer.get());
	}
}


BOOST_AUTO_TEST_CASE_TEMPLATE( Matrix_Subrange, Orientation, result_orientations){
	matrix<float, Orientation, gpu_tag> data = denseData;
	auto storageR = data.raw_storage();
	std::size_t stride1 = Dimensions2;
	std::size_t stride2 = 1;
	if(std::is_same<Orientation,column_major>::value){
		stride1 = 1;
		stride2 = Dimensions1;
	}
	
	{
		dense_matrix_adaptor<float, Orientation, dense_tag, gpu_tag> closure = subrange(data,2,5,3,8);
		BOOST_CHECK_EQUAL(closure.size1(),3);
		BOOST_CHECK_EQUAL(closure.size2(),5);
		auto storage = closure.raw_storage();
		BOOST_CHECK_EQUAL(storage.offset,2*stride1+3*stride2);
		BOOST_CHECK_EQUAL(storage.leading_dimension,storageR.leading_dimension);
		BOOST_CHECK_EQUAL(storage.buffer.get(),storageR.buffer.get());
	}
	{
		dense_matrix_adaptor<float const, Orientation, dense_tag, gpu_tag> closure = subrange((matrix<float, Orientation, gpu_tag> const&)data,2,5,3,8);
		BOOST_CHECK_EQUAL(closure.size1(),3);
		BOOST_CHECK_EQUAL(closure.size2(),5);
		auto storage = closure.raw_storage();
		BOOST_CHECK_EQUAL(storage.offset,2*stride1+3*stride2);
		BOOST_CHECK_EQUAL(storage.leading_dimension,storageR.leading_dimension);
		BOOST_CHECK_EQUAL(storage.buffer.get(),storageR.buffer.get());
	}
	{
		dense_matrix_adaptor<float const, Orientation, dense_tag, gpu_tag> closure = subrange(data,2,5,3,8);
		BOOST_CHECK_EQUAL(closure.size1(),3);
		BOOST_CHECK_EQUAL(closure.size2(),5);
		auto storage = closure.raw_storage();
		BOOST_CHECK_EQUAL(storage.offset,2*stride1+3*stride2);
		BOOST_CHECK_EQUAL(storage.leading_dimension,storageR.leading_dimension);
		BOOST_CHECK_EQUAL(storage.buffer.get(),storageR.buffer.get());
	}
	
	storageR.offset = 2 * stride1;
	dense_matrix_adaptor<float, Orientation, dense_tag, gpu_tag> proxy(storageR, data.queue(), Dimensions1-2,Dimensions2-2);
	dense_matrix_adaptor<float const, Orientation, dense_tag, gpu_tag> const_proxy = proxy;
	{
		dense_matrix_adaptor<float, Orientation, dense_tag, gpu_tag> closure = subrange(proxy,2,5,3,8);
		BOOST_CHECK_EQUAL(closure.size1(),3);
		BOOST_CHECK_EQUAL(closure.size2(),5);
		auto storage = closure.raw_storage();
		BOOST_CHECK_EQUAL(storage.offset,4*stride1+3*stride2);
		BOOST_CHECK_EQUAL(storage.leading_dimension,storageR.leading_dimension);
		BOOST_CHECK_EQUAL(storage.buffer.get(),storageR.buffer.get());
	}
	{
		dense_matrix_adaptor<float const, Orientation, dense_tag, gpu_tag> closure = subrange(const_proxy,2,5,3,8);
		BOOST_CHECK_EQUAL(closure.size1(),3);
		BOOST_CHECK_EQUAL(closure.size2(),5);
		auto storage = closure.raw_storage();
		BOOST_CHECK_EQUAL(storage.offset,4*stride1+3*stride2);
		BOOST_CHECK_EQUAL(storage.leading_dimension,storageR.leading_dimension);
		BOOST_CHECK_EQUAL(storage.buffer.get(),storageR.buffer.get());
	}
	
	//also check rvalue version
	{
		dense_matrix_adaptor<float, Orientation, dense_tag, gpu_tag> closure = subrange(dense_matrix_adaptor<float, Orientation, dense_tag, gpu_tag>(proxy),2,5,3,8);
		BOOST_CHECK_EQUAL(closure.size1(),3);
		BOOST_CHECK_EQUAL(closure.size2(),5);
		auto storage = closure.raw_storage();
		BOOST_CHECK_EQUAL(storage.offset,4*stride1+3*stride2);
		BOOST_CHECK_EQUAL(storage.leading_dimension,storageR.leading_dimension);
		BOOST_CHECK_EQUAL(storage.buffer.get(),storageR.buffer.get());
	}
	{
		dense_matrix_adaptor<float const, Orientation, dense_tag, gpu_tag> closure = subrange(dense_matrix_adaptor<float const, Orientation, dense_tag, gpu_tag>(proxy),2,5,3,8);
		BOOST_CHECK_EQUAL(closure.size1(),3);
		BOOST_CHECK_EQUAL(closure.size2(),5);
		auto storage = closure.raw_storage();
		BOOST_CHECK_EQUAL(storage.offset,4*stride1+3*stride2);
		BOOST_CHECK_EQUAL(storage.leading_dimension,storageR.leading_dimension);
		BOOST_CHECK_EQUAL(storage.buffer.get(),storageR.buffer.get());
	}
}

BOOST_AUTO_TEST_CASE_TEMPLATE( Matrix_Rows, Orientation, result_orientations){
	matrix<float, Orientation, gpu_tag> data = denseData;
	auto storageR = data.raw_storage();
	std::size_t stride1 = Dimensions2;
	if(std::is_same<Orientation,column_major>::value){
		stride1 = 1;
	}
	typedef typename std::conditional<
		std::is_same<Orientation,column_major>::value,
		dense_tag,
		continuous_dense_tag
	>::type Tag;
	
	{
		dense_matrix_adaptor<float, Orientation, Tag, gpu_tag> closure = rows(data,2,5);
		BOOST_CHECK_EQUAL(closure.size1(),3);
		BOOST_CHECK_EQUAL(closure.size2(),data.size2());
		auto storage = closure.raw_storage();
		BOOST_CHECK_EQUAL(storage.offset,2*stride1);
		BOOST_CHECK_EQUAL(storage.leading_dimension,storageR.leading_dimension);
		BOOST_CHECK_EQUAL(storage.buffer.get(),storageR.buffer.get());
	}
	{
		dense_matrix_adaptor<float const, Orientation, Tag, gpu_tag> closure = rows((matrix<float, Orientation, gpu_tag> const&)data,2,5);
		BOOST_CHECK_EQUAL(closure.size1(),3);
		BOOST_CHECK_EQUAL(closure.size2(),data.size2());
		auto storage = closure.raw_storage();
		BOOST_CHECK_EQUAL(storage.offset,2*stride1);
		BOOST_CHECK_EQUAL(storage.leading_dimension,storageR.leading_dimension);
		BOOST_CHECK_EQUAL(storage.buffer.get(),storageR.buffer.get());
	}
	{
		dense_matrix_adaptor<float const, Orientation, Tag, gpu_tag> closure = rows(data,2,5);
		BOOST_CHECK_EQUAL(closure.size1(),3);
		BOOST_CHECK_EQUAL(closure.size2(),data.size2());
		auto storage = closure.raw_storage();
		BOOST_CHECK_EQUAL(storage.offset,2*stride1);
		BOOST_CHECK_EQUAL(storage.leading_dimension,storageR.leading_dimension);
		BOOST_CHECK_EQUAL(storage.buffer.get(),storageR.buffer.get());
	}
	
	storageR.offset = 2 * stride1;
	dense_matrix_adaptor<float, Orientation, continuous_dense_tag, gpu_tag> proxy(storageR, data.queue(), Dimensions1-2,Dimensions2-2);
	dense_matrix_adaptor<float const, Orientation, continuous_dense_tag, gpu_tag> const_proxy = proxy;
	{
		dense_matrix_adaptor<float, Orientation, Tag, gpu_tag> closure = rows(proxy,2,5);
		BOOST_CHECK_EQUAL(closure.size1(),3);
		BOOST_CHECK_EQUAL(closure.size2(),proxy.size2());
		auto storage = closure.raw_storage();
		BOOST_CHECK_EQUAL(storage.offset,4*stride1);
		BOOST_CHECK_EQUAL(storage.leading_dimension,storageR.leading_dimension);
		BOOST_CHECK_EQUAL(storage.buffer.get(),storageR.buffer.get());
	}
	{
		dense_matrix_adaptor<float const, Orientation, Tag, gpu_tag> closure = rows(const_proxy,2,5);
		BOOST_CHECK_EQUAL(closure.size1(),3);
		BOOST_CHECK_EQUAL(closure.size2(),proxy.size2());
		auto storage = closure.raw_storage();
		BOOST_CHECK_EQUAL(storage.offset,4*stride1);
		BOOST_CHECK_EQUAL(storage.leading_dimension,storageR.leading_dimension);
		BOOST_CHECK_EQUAL(storage.buffer.get(),storageR.buffer.get());
	}
	
	//also check rvalue version
	{
		dense_matrix_adaptor<float, Orientation, Tag, gpu_tag> closure = rows(dense_matrix_adaptor<float, Orientation, continuous_dense_tag, gpu_tag>(proxy),2,5);
		BOOST_CHECK_EQUAL(closure.size1(),3);
		BOOST_CHECK_EQUAL(closure.size2(),proxy.size2());
		auto storage = closure.raw_storage();
		BOOST_CHECK_EQUAL(storage.offset,4*stride1);
		BOOST_CHECK_EQUAL(storage.leading_dimension,storageR.leading_dimension);
		BOOST_CHECK_EQUAL(storage.buffer.get(),storageR.buffer.get());
	}
	{
		dense_matrix_adaptor<float const, Orientation, Tag, gpu_tag> closure = rows(dense_matrix_adaptor<float const, Orientation, continuous_dense_tag, gpu_tag>(proxy),2,5);
		BOOST_CHECK_EQUAL(closure.size1(),3);
		BOOST_CHECK_EQUAL(closure.size2(),proxy.size2());
		auto storage = closure.raw_storage();
		BOOST_CHECK_EQUAL(storage.offset,4*stride1);
		BOOST_CHECK_EQUAL(storage.leading_dimension,storageR.leading_dimension);
		BOOST_CHECK_EQUAL(storage.buffer.get(),storageR.buffer.get());
	}
	
	//also check from dense_tag
	{
		dense_matrix_adaptor<float, Orientation, dense_tag, gpu_tag> closure = rows(dense_matrix_adaptor<float, Orientation, dense_tag, gpu_tag>(proxy),2,5);
		BOOST_CHECK_EQUAL(closure.size1(),3);
		BOOST_CHECK_EQUAL(closure.size2(),proxy.size2());
		auto storage = closure.raw_storage();
		BOOST_CHECK_EQUAL(storage.offset,4*stride1);
		BOOST_CHECK_EQUAL(storage.leading_dimension,storageR.leading_dimension);
		BOOST_CHECK_EQUAL(storage.buffer.get(),storageR.buffer.get());
	}
	{
		dense_matrix_adaptor<float const, Orientation, dense_tag, gpu_tag> closure = rows(dense_matrix_adaptor<float const, Orientation, dense_tag, gpu_tag>(proxy),2,5);
		BOOST_CHECK_EQUAL(closure.size1(),3);
		BOOST_CHECK_EQUAL(closure.size2(),proxy.size2());
		auto storage = closure.raw_storage();
		BOOST_CHECK_EQUAL(storage.offset,4*stride1);
		BOOST_CHECK_EQUAL(storage.leading_dimension,storageR.leading_dimension);
		BOOST_CHECK_EQUAL(storage.buffer.get(),storageR.buffer.get());
	}
}

BOOST_AUTO_TEST_CASE_TEMPLATE( Matrix_Diagonal, Orientation, result_orientations){
	matrix<float, Orientation, gpu_tag> data = subrange(denseData,0,Dimensions2,0,Dimensions2);
	auto storageR = data.raw_storage();

	{
		dense_vector_adaptor<float, dense_tag, gpu_tag> closure = diag(data);
		BOOST_CHECK_EQUAL(closure.size(),data.size2());
		auto storage = closure.raw_storage();
		BOOST_CHECK_EQUAL(storage.offset,0);
		BOOST_CHECK_EQUAL(storage.stride,storageR.leading_dimension+1);
		BOOST_CHECK_EQUAL(storage.buffer.get(),storageR.buffer.get());
	}
	{
		dense_vector_adaptor<float const, dense_tag, gpu_tag> closure = diag((matrix<float, Orientation, gpu_tag> const&)data);
		BOOST_CHECK_EQUAL(closure.size(),data.size2());
		auto storage = closure.raw_storage();
		BOOST_CHECK_EQUAL(storage.offset,0);
		BOOST_CHECK_EQUAL(storage.stride,storageR.leading_dimension+1);
		BOOST_CHECK_EQUAL(storage.buffer.get(),storageR.buffer.get());
	}
	{
		dense_vector_adaptor<float const, dense_tag, gpu_tag> closure = diag(data);
		BOOST_CHECK_EQUAL(closure.size(),data.size2());
		auto storage = closure.raw_storage();
		BOOST_CHECK_EQUAL(storage.offset,0);
		BOOST_CHECK_EQUAL(storage.stride,storageR.leading_dimension+1);
		BOOST_CHECK_EQUAL(storage.buffer.get(),storageR.buffer.get());
	}
	
	
	storageR.offset = 4;
	typename matrix<float, Orientation, gpu_tag>::closure_type proxy(storageR, data.queue(), Dimensions2-1,Dimensions2-1);
	typename matrix<float, Orientation, gpu_tag>::const_closure_type const_proxy = proxy;
	{
		dense_vector_adaptor<float, dense_tag, gpu_tag> closure = diag(proxy);
		BOOST_CHECK_EQUAL(closure.size(),data.size2()-1);
		auto storage = closure.raw_storage();
		BOOST_CHECK_EQUAL(storage.offset,4);
		BOOST_CHECK_EQUAL(storage.stride,storageR.leading_dimension+1);
		BOOST_CHECK_EQUAL(storage.buffer.get(),storageR.buffer.get());
	}
	{
		dense_vector_adaptor<float const, dense_tag, gpu_tag> closure = diag(const_proxy);
		BOOST_CHECK_EQUAL(closure.size(),data.size2()-1);
		auto storage = closure.raw_storage();
		BOOST_CHECK_EQUAL(storage.offset,4);
		BOOST_CHECK_EQUAL(storage.stride,storageR.leading_dimension+1);
		BOOST_CHECK_EQUAL(storage.buffer.get(),storageR.buffer.get());
	}
	
	//also check rvalue version
	{
		dense_vector_adaptor<float, dense_tag, gpu_tag> closure = diag(typename matrix<float, Orientation, gpu_tag>::closure_type(proxy));
		BOOST_CHECK_EQUAL(closure.size(),data.size2()-1);
		auto storage = closure.raw_storage();
		BOOST_CHECK_EQUAL(storage.offset,4);
		BOOST_CHECK_EQUAL(storage.stride,storageR.leading_dimension+1);
		BOOST_CHECK_EQUAL(storage.buffer.get(),storageR.buffer.get());
	}
	{
		dense_vector_adaptor<float const, dense_tag, gpu_tag> closure = diag(typename matrix<float, Orientation, gpu_tag>::const_closure_type(proxy));
		BOOST_CHECK_EQUAL(closure.size(),data.size2()-1);
		auto storage = closure.raw_storage();
		BOOST_CHECK_EQUAL(storage.offset,4);
		BOOST_CHECK_EQUAL(storage.stride,storageR.leading_dimension+1);
		BOOST_CHECK_EQUAL(storage.buffer.get(),storageR.buffer.get());
	}
}

BOOST_AUTO_TEST_CASE_TEMPLATE( Matrix_Linearizer, Orientation, result_orientations){
	matrix<float, Orientation, gpu_tag> data = denseData;
	auto storageR = data.raw_storage();

	{
		dense_vector_adaptor<float, continuous_dense_tag, gpu_tag> closure = to_vector(data);
		BOOST_CHECK_EQUAL(closure.size(), data.size1() * data.size2());
		auto storage = closure.raw_storage();
		BOOST_CHECK_EQUAL(storage.offset,0);
		BOOST_CHECK_EQUAL(storage.stride,1);
		BOOST_CHECK_EQUAL(storage.buffer.get(),storageR.buffer.get());
	}
	{
		dense_vector_adaptor<float const, continuous_dense_tag, gpu_tag> closure = to_vector((matrix<float, Orientation, gpu_tag> const&)data);
		BOOST_CHECK_EQUAL(closure.size(), data.size1() * data.size2());
		auto storage = closure.raw_storage();
		BOOST_CHECK_EQUAL(storage.offset,0);
		BOOST_CHECK_EQUAL(storage.stride,1);
		BOOST_CHECK_EQUAL(storage.buffer.get(),storageR.buffer.get());
	}
	{
		dense_vector_adaptor<float const, continuous_dense_tag, gpu_tag> closure = to_vector(data);
		BOOST_CHECK_EQUAL(closure.size(), data.size1() * data.size2());
		auto storage = closure.raw_storage();
		BOOST_CHECK_EQUAL(storage.offset,0);
		BOOST_CHECK_EQUAL(storage.stride,1);
		BOOST_CHECK_EQUAL(storage.buffer.get(),storageR.buffer.get());
	}
	
	
	typename matrix<float, Orientation, gpu_tag>::closure_type proxy(storageR, data.queue(), Dimensions1,Dimensions2);
	typename matrix<float, Orientation, gpu_tag>::const_closure_type const_proxy = proxy;
	//also check rvalue version
	{
		dense_vector_adaptor<float, continuous_dense_tag, gpu_tag> closure = to_vector(typename matrix<float, Orientation, gpu_tag>::closure_type(proxy));
		BOOST_CHECK_EQUAL(closure.size(), data.size1() * data.size2());
		auto storage = closure.raw_storage();
		BOOST_CHECK_EQUAL(storage.offset,0);
		BOOST_CHECK_EQUAL(storage.stride,1);
		BOOST_CHECK_EQUAL(storage.buffer.get(),storageR.buffer.get());
	}
	{
		dense_vector_adaptor<float const, continuous_dense_tag, gpu_tag> closure = to_vector(typename matrix<float, Orientation, gpu_tag>::const_closure_type(proxy));
		BOOST_CHECK_EQUAL(closure.size(), data.size1() * data.size2());
		auto storage = closure.raw_storage();
		BOOST_CHECK_EQUAL(storage.offset,0);
		BOOST_CHECK_EQUAL(storage.stride,1);
		BOOST_CHECK_EQUAL(storage.buffer.get(),storageR.buffer.get());
	}
}


BOOST_AUTO_TEST_CASE( Vector_To_Matrix){
	auto storageR = denseDataVec.raw_storage();

	{
		dense_matrix_adaptor<float, row_major, continuous_dense_tag, gpu_tag> closure = to_matrix(denseDataVec,4,5);
		BOOST_CHECK_EQUAL(closure.size1(),4);
		BOOST_CHECK_EQUAL(closure.size2(),5);
		auto storage = closure.raw_storage();
		BOOST_CHECK_EQUAL(storage.offset,0);
		BOOST_CHECK_EQUAL(storage.leading_dimension,5);
		BOOST_CHECK_EQUAL(storage.buffer.get(),storageR.buffer.get());
	}
	{
		dense_matrix_adaptor<float const, row_major, continuous_dense_tag, gpu_tag> closure = to_matrix((remora::vector<float,gpu_tag> const&)denseDataVec,4,5);
		BOOST_CHECK_EQUAL(closure.size1(),4);
		BOOST_CHECK_EQUAL(closure.size2(),5);
		auto storage = closure.raw_storage();
		BOOST_CHECK_EQUAL(storage.offset,0);
		BOOST_CHECK_EQUAL(storage.leading_dimension,5);
		BOOST_CHECK_EQUAL(storage.buffer.get(),storageR.buffer.get());
	}
	{
		dense_matrix_adaptor<float const, row_major, continuous_dense_tag, gpu_tag> closure = to_matrix(denseDataVec,4,5);
		BOOST_CHECK_EQUAL(closure.size1(),4);
		BOOST_CHECK_EQUAL(closure.size2(),5);
		auto storage = closure.raw_storage();
		BOOST_CHECK_EQUAL(storage.offset,0);
		BOOST_CHECK_EQUAL(storage.leading_dimension,5);
		BOOST_CHECK_EQUAL(storage.buffer.get(),storageR.buffer.get());
	}
	
	
	storageR.offset = 4;
	typename vector<float, gpu_tag>::closure_type proxy(storageR, denseDataVec.queue(), 16);
	typename vector<float, gpu_tag>::const_closure_type const_proxy = proxy;
	{
		dense_matrix_adaptor<float, row_major, continuous_dense_tag, gpu_tag> closure = to_matrix(proxy,2,8);
		BOOST_CHECK_EQUAL(closure.size1(),2);
		BOOST_CHECK_EQUAL(closure.size2(),8);
		auto storage = closure.raw_storage();
		BOOST_CHECK_EQUAL(storage.offset,4);
		BOOST_CHECK_EQUAL(storage.leading_dimension,8);
		BOOST_CHECK_EQUAL(storage.buffer.get(),storageR.buffer.get());
	}
	{
		dense_matrix_adaptor<float const, row_major, continuous_dense_tag, gpu_tag> closure = to_matrix(const_proxy,2,8);
		BOOST_CHECK_EQUAL(closure.size1(),2);
		BOOST_CHECK_EQUAL(closure.size2(),8);
		auto storage = closure.raw_storage();
		BOOST_CHECK_EQUAL(storage.offset,4);
		BOOST_CHECK_EQUAL(storage.leading_dimension,8);
		BOOST_CHECK_EQUAL(storage.buffer.get(),storageR.buffer.get());
	}
	
	//also check rvalue version
	{
		dense_matrix_adaptor<float, row_major, continuous_dense_tag, gpu_tag> closure = to_matrix(dense_vector_adaptor<float, continuous_dense_tag, gpu_tag>(proxy),2,8);
		BOOST_CHECK_EQUAL(closure.size1(),2);
		BOOST_CHECK_EQUAL(closure.size2(),8);
		auto storage = closure.raw_storage();
		BOOST_CHECK_EQUAL(storage.offset,4);
		BOOST_CHECK_EQUAL(storage.leading_dimension,8);
		BOOST_CHECK_EQUAL(storage.buffer.get(),storageR.buffer.get());
	}
	{
		dense_matrix_adaptor<float const, row_major, continuous_dense_tag, gpu_tag> closure = to_matrix(dense_vector_adaptor<float const, continuous_dense_tag, gpu_tag>(const_proxy),2,8);
		BOOST_CHECK_EQUAL(closure.size1(),2);
		BOOST_CHECK_EQUAL(closure.size2(),8);
		auto storage = closure.raw_storage();
		BOOST_CHECK_EQUAL(storage.offset,4);
		BOOST_CHECK_EQUAL(storage.leading_dimension,8);
		BOOST_CHECK_EQUAL(storage.buffer.get(),storageR.buffer.get());
	}
}


BOOST_AUTO_TEST_SUITE_END();