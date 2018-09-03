#define BOOST_TEST_MODULE Remora_hip_triangular_prod
#include <boost/test/unit_test.hpp>
#include <boost/test/floating_point_comparison.hpp>
#include <boost/mpl/list.hpp>

#include <remora/dense.hpp>
#include <remora/device_copy.hpp>
#include <remora/matrix_expression.hpp>

#include <iostream>
using namespace remora;

template<class M, class V, class Result>
void checkMatrixVectorMultiply(M const& arg1, V const& arg2_hip, Result const& result_hip,float init, float alpha){
	BOOST_REQUIRE_EQUAL(arg1.size1(), result_hip.size());
	BOOST_REQUIRE_EQUAL(arg2_hip.size(), arg1.size2());
	
	vector<float> arg2 = copy_to_cpu(arg2_hip);
	vector<float> result = copy_to_cpu(result_hip);
	
	for(std::size_t i = 0; i != arg1.size1(); ++i) {
		float test_result = alpha*inner_prod(row(arg1,i),arg2)+init;
		BOOST_CHECK_CLOSE(result(i), test_result, 1.e-3);
	}
}

template<class M1, class M2, class Result>
void checkMatrixMatrixMultiply(M1 const& arg1, M2 const& arg2_hip, Result const& result_hip,float init, float alpha) {
	BOOST_REQUIRE_EQUAL(arg1.size1(), arg1.size2());
	BOOST_REQUIRE_EQUAL(arg1.size2(), arg2_hip.size1());
	BOOST_REQUIRE_EQUAL(arg1.size2(), result_hip.size1());
	BOOST_REQUIRE_EQUAL(arg2_hip.size2(), result_hip.size2());
	
	matrix<float> arg2 = copy_to_cpu(arg2_hip);
	matrix<float> result = copy_to_cpu(result_hip);
	
	for(std::size_t i = 0; i != arg2.size1(); ++i) {
		for(std::size_t j = 0; j != arg2.size2(); ++j) {
			float test_result = alpha*inner_prod(row(arg1,i),column(arg2,j))+init;
			BOOST_CHECK_CLOSE(result(i,j), test_result, 1.e-3);
		}
	}
}

BOOST_AUTO_TEST_SUITE(Remora_hip_triangular_prod)

BOOST_AUTO_TEST_CASE(Remora_hip_triangular_prod_matrix_vector) {
	std::size_t dims = 231;//chosen as not to be a multiple of the block size
	//initialize the arguments in both row and column major, lower and upper, unit and non-unit diagonal
	//we add one on the remaining elements to ensure, that triangular_prod does not tuch these elements
	matrix<float,row_major> arg1lowerrm_cpu(dims,dims,1.0);
	matrix<float,column_major> arg1lowercm_cpu(dims,dims,1.0);
	matrix<float,row_major> arg1upperrm_cpu(dims,dims,1.0);
	matrix<float,column_major> arg1uppercm_cpu(dims,dims,1.0);
	
	//inputs to compare to with the standard prod
	matrix<float,row_major> arg1lowertest(dims,dims,0.0);
	matrix<float,row_major> arg1uppertest(dims,dims,0.0);
	for(std::size_t i = 0; i != dims; ++i){
		for(std::size_t j = 0; j <=i; ++j){
			arg1lowerrm_cpu(i,j) = arg1lowercm_cpu(i,j) = i*dims+0.2*j+1;
			arg1lowertest(i,j) = i*dims+0.2*j+1;
			arg1upperrm_cpu(j,i) = arg1uppercm_cpu(j,i) = i*dims+0.2*j+1;
			arg1uppertest(j,i) = i*dims+0.2*j+1;
		}
	}
	
	
	vector<float> arg2_cpu(dims);
	for(std::size_t j = 0; j != dims; ++j){
		arg2_cpu(j)  = 1.5*j+2;
	}
	
	matrix<float, row_major, hip_tag> arg1lowerrm = copy_to_device(arg1lowerrm_cpu, hip_tag());
	matrix<float, column_major, hip_tag> arg1lowercm = copy_to_device(arg1lowercm_cpu, hip_tag());
	matrix<float, row_major, hip_tag> arg1upperrm = copy_to_device(arg1upperrm_cpu, hip_tag());
	matrix<float, row_major, hip_tag> arg1uppercm = copy_to_device(arg1uppercm_cpu, hip_tag());
	vector<float, hip_tag> arg2 = copy_to_device(arg2_cpu, hip_tag());

	std::cout<<"\nchecking matrix-vector prod v=Ax non-unit"<<std::endl;
	{
		std::cout<<"row major lower Ax"<<std::endl;
		vector<float, hip_tag> result(dims,3.0);
		noalias(result) = triangular_prod<lower>(arg1lowerrm,arg2);
		checkMatrixVectorMultiply(arg1lowertest,arg2,result,0.0,1);
	}
	{
		std::cout<<"column major lower Ax"<<std::endl;
		vector<float, hip_tag> result(dims,3.0);
		noalias(result) = triangular_prod<lower>(arg1lowercm,arg2);
		checkMatrixVectorMultiply(arg1lowertest,arg2,result,0.0,1);
	}
	{
		std::cout<<"row major upper Ax"<<std::endl;
		vector<float, hip_tag> result(dims,3.0);
		noalias(result) = triangular_prod<upper>(arg1upperrm,arg2);
		checkMatrixVectorMultiply(arg1uppertest,arg2,result,0.0,1);
	}
	{
		std::cout<<"column major upper Ax"<<std::endl;
		vector<float, hip_tag> result(dims,3.0);
		noalias(result) = triangular_prod<upper>(arg1uppercm,arg2);
		checkMatrixVectorMultiply(arg1uppertest,arg2,result,0.0,1);
	}
	//with prefactor
	{
		std::cout<<"row major lower Ax"<<std::endl;
		vector<float, hip_tag> result(dims,3.0);
		noalias(result) = -2 * triangular_prod<lower>(arg1lowerrm,arg2);
		checkMatrixVectorMultiply(arg1lowertest,arg2,result,0.0,-2);
	}
	{
		std::cout<<"column major lower Ax"<<std::endl;
		vector<float, hip_tag> result(dims,3.0);
		noalias(result) = -2 * triangular_prod<lower>(arg1lowercm,arg2);
		checkMatrixVectorMultiply(arg1lowertest,arg2,result,0.0,-2);
	}
	{
		std::cout<<"row major upper Ax"<<std::endl;
		vector<float, hip_tag> result(dims,3.0);
		noalias(result) = -2 * triangular_prod<upper>(arg1upperrm,arg2);
		checkMatrixVectorMultiply(arg1uppertest,arg2,result,0.0,-2);
	}
	{
		std::cout<<"column major upper Ax"<<std::endl;
		vector<float, hip_tag> result(dims,3.0);
		noalias(result) = -2 * triangular_prod<upper>(arg1uppercm,arg2);
		checkMatrixVectorMultiply(arg1uppertest,arg2,result,0.0,-2);
	}
	std::cout<<"\nchecking matrix-vector prod v+=Ax non-unit"<<std::endl;
	{
		std::cout<<"row major lower Ax"<<std::endl;
		vector<float, hip_tag> result(dims,3.0);
		noalias(result) += -2 * triangular_prod<lower>(arg1lowerrm,arg2);
		checkMatrixVectorMultiply(arg1lowertest,arg2,result,3.0,-2);
	}
	{
		std::cout<<"column major lower Ax"<<std::endl;
		vector<float, hip_tag> result(dims,3.0);
		noalias(result) += -2 * triangular_prod<lower>(arg1lowercm,arg2);
		checkMatrixVectorMultiply(arg1lowertest,arg2,result,3.0,-2);
	}
	{
		std::cout<<"row major upper Ax"<<std::endl;
		vector<float, hip_tag> result(dims,3.0);
		noalias(result) += -2 * triangular_prod<upper>(arg1upperrm,arg2);
		checkMatrixVectorMultiply(arg1uppertest,arg2,result,3.0,-2);
	}
	{
		std::cout<<"column major upper Ax"<<std::endl;
		vector<float, hip_tag> result(dims,3.0);
		noalias(result) += -2 * triangular_prod<upper>(arg1uppercm,arg2);
		checkMatrixVectorMultiply(arg1uppertest,arg2,result,3.0,-2);
	}
	std::cout<<"\nchecking matrix-vector prod v-=Ax non-unit"<<std::endl;
	{
		std::cout<<"row major lower Ax"<<std::endl;
		vector<float, hip_tag> result(dims,3.0);
		noalias(result) -= 2 * triangular_prod<lower>(arg1lowerrm,arg2);
		checkMatrixVectorMultiply(arg1lowertest,arg2,result,3.0,-2);
	}
	{
		std::cout<<"column major lower Ax"<<std::endl;
		vector<float, hip_tag> result(dims,3.0);
		noalias(result) -= 2 * triangular_prod<lower>(arg1lowercm,arg2);
		checkMatrixVectorMultiply(arg1lowertest,arg2,result,3.0,-2);
	}
	{
		std::cout<<"row major upper Ax"<<std::endl;
		vector<float, hip_tag> result(dims,3.0);
		noalias(result) -= 2 * triangular_prod<upper>(arg1upperrm,arg2);
		checkMatrixVectorMultiply(arg1uppertest,arg2,result,3.0,-2);
	}
	{
		std::cout<<"column major upper Ax"<<std::endl;
		vector<float, hip_tag> result(dims,3.0);
		noalias(result) -= 2 * triangular_prod<upper>(arg1uppercm,arg2);
		checkMatrixVectorMultiply(arg1uppertest,arg2,result,3.0,-2);
	}
	
	
	diag(arg1lowertest) = repeat(1.0,dims);
	diag(arg1uppertest) = repeat(1.0,dims);
	std::cout<<"\nchecking matrix-vector prod v=Ax unit"<<std::endl;
	{
		std::cout<<"row major lower Ax"<<std::endl;
		vector<float, hip_tag> result(dims,3.0);
		noalias(result) = triangular_prod<unit_lower>(arg1lowerrm,arg2);
		checkMatrixVectorMultiply(arg1lowertest,arg2,result,0.0,1);
	}
	{
		std::cout<<"column major lower Ax"<<std::endl;
		vector<float, hip_tag> result(dims,3.0);
		noalias(result) = triangular_prod<unit_lower>(arg1lowercm,arg2);
		checkMatrixVectorMultiply(arg1lowertest,arg2,result,0.0,1);
	}
	{
		std::cout<<"row major upper Ax"<<std::endl;
		vector<float, hip_tag> result(dims,3.0);
		noalias(result) = triangular_prod<unit_upper>(arg1upperrm,arg2);
		checkMatrixVectorMultiply(arg1uppertest,arg2,result,0.0,1);
	}
	{
		std::cout<<"column major upper Ax"<<std::endl;
		vector<float, hip_tag> result(dims,3.0);
		noalias(result) = triangular_prod<unit_upper>(arg1uppercm,arg2);
		checkMatrixVectorMultiply(arg1uppertest,arg2,result,0.0,1);
	}
	//with prefactor
	{
		std::cout<<"row major lower Ax"<<std::endl;
		vector<float, hip_tag> result(dims,3.0);
		noalias(result) = -2 * triangular_prod<unit_lower>(arg1lowerrm,arg2);
		checkMatrixVectorMultiply(arg1lowertest,arg2,result,0.0,-2);
	}
	{
		std::cout<<"column major lower Ax"<<std::endl;
		vector<float, hip_tag> result(dims,3.0);
		noalias(result) = -2 * triangular_prod<unit_lower>(arg1lowercm,arg2);
		checkMatrixVectorMultiply(arg1lowertest,arg2,result,0.0,-2);
	}
	{
		std::cout<<"row major upper Ax"<<std::endl;
		vector<float, hip_tag> result(dims,3.0);
		noalias(result) = -2 * triangular_prod<unit_upper>(arg1upperrm,arg2);
		checkMatrixVectorMultiply(arg1uppertest,arg2,result,0.0,-2);
	}
	{
		std::cout<<"column major upper Ax"<<std::endl;
		vector<float, hip_tag> result(dims,3.0);
		noalias(result) = -2 * triangular_prod<unit_upper>(arg1uppercm,arg2);
		checkMatrixVectorMultiply(arg1uppertest,arg2,result,0.0,-2);
	}
	std::cout<<"\nchecking matrix-vector prod v+=Ax unit"<<std::endl;
	{
		std::cout<<"row major lower Ax"<<std::endl;
		vector<float, hip_tag> result(dims,3.0);
		noalias(result) += -2 * triangular_prod<unit_lower>(arg1lowerrm,arg2);
		checkMatrixVectorMultiply(arg1lowertest,arg2,result,3.0,-2);
	}
	{
		std::cout<<"column major lower Ax"<<std::endl;
		vector<float, hip_tag> result(dims,3.0);
		noalias(result) += -2 * triangular_prod<unit_lower>(arg1lowercm,arg2);
		checkMatrixVectorMultiply(arg1lowertest,arg2,result,3.0,-2);
	}
	{
		std::cout<<"row major upper Ax"<<std::endl;
		vector<float, hip_tag> result(dims,3.0);
		noalias(result) += -2 * triangular_prod<unit_upper>(arg1upperrm,arg2);
		checkMatrixVectorMultiply(arg1uppertest,arg2,result,3.0,-2);
	}
	{
		std::cout<<"column major upper Ax"<<std::endl;
		vector<float, hip_tag> result(dims,3.0);
		noalias(result) += -2 * triangular_prod<unit_upper>(arg1uppercm,arg2);
		checkMatrixVectorMultiply(arg1uppertest,arg2,result,3.0,-2);
	}
	std::cout<<"\nchecking matrix-vector prod v-=Ax unit"<<std::endl;
	{
		std::cout<<"row major lower Ax"<<std::endl;
		vector<float, hip_tag> result(dims,3.0);
		noalias(result) -= 2 * triangular_prod<unit_lower>(arg1lowerrm,arg2);
		checkMatrixVectorMultiply(arg1lowertest,arg2,result,3.0,-2);
	}
	{
		std::cout<<"column major lower Ax"<<std::endl;
		vector<float, hip_tag> result(dims,3.0);
		noalias(result) -= 2 * triangular_prod<unit_lower>(arg1lowercm,arg2);
		checkMatrixVectorMultiply(arg1lowertest,arg2,result,3.0,-2);
	}
	{
		std::cout<<"row major upper Ax"<<std::endl;
		vector<float, hip_tag> result(dims,3.0);
		noalias(result) -= 2 * triangular_prod<unit_upper>(arg1upperrm,arg2);
		checkMatrixVectorMultiply(arg1uppertest,arg2,result,3.0,-2);
	}
	{
		std::cout<<"column major upper Ax"<<std::endl;
		vector<float, hip_tag> result(dims,3.0);
		noalias(result) -= 2 * triangular_prod<unit_upper>(arg1uppercm,arg2);
		checkMatrixVectorMultiply(arg1uppertest,arg2,result,3.0,-2);
	}
}

template<class Orientation>
void triangular_prod_matrix_matrix_test(Orientation){
	std::size_t dims = 231;//chosen as not to be a multiple of the block size
	std::size_t N = 255;
	//initialize the arguments in both row and column major, lower and upper, unit and non-unit diagonal
	//we add one on the remaining elements to ensure, that triangular_prod does not tuch these elements
	matrix<float,row_major> arg1lowerrm_cpu(dims,dims,1.0);
	matrix<float,column_major> arg1lowercm_cpu(dims,dims,1.0);
	matrix<float,row_major> arg1upperrm_cpu(dims,dims,1.0);
	matrix<float,column_major> arg1uppercm_cpu(dims,dims,1.0);
	
	//inputs to compare to with the standard prod
	matrix<float,row_major> arg1lowertest(dims,dims,0.0);
	matrix<float,row_major> arg1uppertest(dims,dims,0.0);
	for(std::size_t i = 0; i != dims; ++i){
		for(std::size_t j = 0; j <=i; ++j){
			arg1lowerrm_cpu(i,j) = arg1lowercm_cpu(i,j) = 0.1*i*dims+0.2*j+1;
			arg1lowertest(i,j) = 0.1*i*dims+0.2*j+1;
			arg1upperrm_cpu(j,i) = arg1uppercm_cpu(j,i) = 0.1*i*dims+0.2*j+1;
			arg1uppertest(j,i) = 0.1*i*dims+0.2*j+1;
		}
	}
	matrix<float> arg2_cpu(dims,N);
	for(std::size_t i = 0; i != dims; ++i) {
		for(std::size_t j = 0; j != N; ++j) {
			arg2_cpu(i,j)  = (1.5/N) * j + 2+i;
		}
	}
	
	matrix<float, row_major, hip_tag> arg1lowerrm = copy_to_device(arg1lowerrm_cpu, hip_tag());
	matrix<float, column_major, hip_tag> arg1lowercm = copy_to_device(arg1lowercm_cpu, hip_tag());
	matrix<float, row_major, hip_tag> arg1upperrm = copy_to_device(arg1upperrm_cpu, hip_tag());
	matrix<float, row_major, hip_tag> arg1uppercm = copy_to_device(arg1uppercm_cpu, hip_tag());
	matrix<float, column_major, hip_tag> arg2 = copy_to_device(arg2_cpu, hip_tag());

	std::cout << "\nchecking matrix-matrix prod V=AX non-unit" << std::endl;
	{
		std::cout<<"row major lower AX"<<std::endl;
		matrix<float, Orientation, hip_tag> result(dims, N, 3.0);
		noalias(result) = triangular_prod<lower>(arg1lowerrm,arg2);
		checkMatrixMatrixMultiply(arg1lowertest,arg2,result,0.0,1);
	}
	{
		std::cout<<"column major lower AX"<<std::endl;
		matrix<float, Orientation, hip_tag> result(dims, N, 3.0);
		noalias(result) = triangular_prod<lower>(arg1lowercm,arg2);
		checkMatrixMatrixMultiply(arg1lowertest,arg2,result,0.0,1);
	}
	{
		std::cout<<"row major upper AX"<<std::endl;
		matrix<float, Orientation, hip_tag> result(dims, N, 3.0);
		noalias(result) = triangular_prod<upper>(arg1upperrm,arg2);
		checkMatrixMatrixMultiply(arg1uppertest,arg2,result,0.0,1);
	}
	{
		std::cout<<"column major upper AX"<<std::endl;
		matrix<float, Orientation, hip_tag> result(dims, N, 3.0);
		noalias(result) = triangular_prod<upper>(arg1uppercm,arg2);
		checkMatrixMatrixMultiply(arg1uppertest,arg2,result,0.0,1);
	}
	//with prefactor
	{
		std::cout<<"row major lower AX"<<std::endl;
		matrix<float, Orientation, hip_tag> result(dims, N, 3.0);
		noalias(result) = -2 * triangular_prod<lower>(arg1lowerrm,arg2);
		checkMatrixMatrixMultiply(arg1lowertest,arg2,result,0.0,-2);
	}
	{
		std::cout<<"column major lower AX"<<std::endl;
		matrix<float, Orientation, hip_tag> result(dims, N, 3.0);
		noalias(result) = -2 * triangular_prod<lower>(arg1lowercm,arg2);
		checkMatrixMatrixMultiply(arg1lowertest,arg2,result,0.0,-2);
	}
	{
		std::cout<<"row major upper AX"<<std::endl;
		matrix<float, Orientation, hip_tag> result(dims, N, 3.0);
		noalias(result) = -2 * triangular_prod<upper>(arg1upperrm,arg2);
		checkMatrixMatrixMultiply(arg1uppertest,arg2,result,0.0,-2);
	}
	{
		std::cout<<"column major upper AX"<<std::endl;
		matrix<float, Orientation, hip_tag> result(dims, N, 3.0);
		noalias(result) = -2 * triangular_prod<upper>(arg1uppercm,arg2);
		checkMatrixMatrixMultiply(arg1uppertest,arg2,result,0.0,-2);
	}
	std::cout << "\nchecking matrix-matrix prod V+=AX non-unit" << std::endl;
	{
		std::cout<<"row major lower AX"<<std::endl;
		matrix<float, Orientation, hip_tag> result(dims, N, 3.0);
		noalias(result) += -2 * triangular_prod<lower>(arg1lowerrm,arg2);
		checkMatrixMatrixMultiply(arg1lowertest,arg2,result,3.0,-2);
	}
	{
		std::cout<<"column major lower AX"<<std::endl;
		matrix<float, Orientation, hip_tag> result(dims, N, 3.0);
		noalias(result) += -2 * triangular_prod<lower>(arg1lowercm,arg2);
		checkMatrixMatrixMultiply(arg1lowertest,arg2,result,3.0,-2);
	}
	{
		std::cout<<"row major upper AX"<<std::endl;
		matrix<float, Orientation, hip_tag> result(dims, N, 3.0);
		noalias(result) += -2 * triangular_prod<upper>(arg1upperrm,arg2);
		checkMatrixMatrixMultiply(arg1uppertest,arg2,result,3.0,-2);
	}
	{
		std::cout<<"column major upper AX"<<std::endl;
		matrix<float, Orientation, hip_tag> result(dims, N, 3.0);
		noalias(result) += -2 * triangular_prod<upper>(arg1uppercm,arg2);
		checkMatrixMatrixMultiply(arg1uppertest,arg2,result,3.0,-2);
	}
	std::cout << "\nchecking matrix-matrix prod V-=AX non-unit" << std::endl;
	{
		std::cout<<"row major lower AX"<<std::endl;
		matrix<float, Orientation, hip_tag> result(dims, N, 3.0);
		noalias(result) -= 2 * triangular_prod<lower>(arg1lowerrm,arg2);
		checkMatrixMatrixMultiply(arg1lowertest,arg2,result,3.0,-2);
	}
	{
		std::cout<<"column major lower AX"<<std::endl;
		matrix<float, Orientation, hip_tag> result(dims, N, 3.0);
		noalias(result) -= 2 * triangular_prod<lower>(arg1lowercm,arg2);
		checkMatrixMatrixMultiply(arg1lowertest,arg2,result,3.0,-2);
	}
	{
		std::cout<<"row major upper AX"<<std::endl;
		matrix<float, Orientation, hip_tag> result(dims, N, 3.0);
		noalias(result) -= 2 * triangular_prod<upper>(arg1upperrm,arg2);
		checkMatrixMatrixMultiply(arg1uppertest,arg2,result,3.0,-2);
	}
	{
		std::cout<<"column major upper AX"<<std::endl;
		matrix<float, Orientation, hip_tag> result(dims, N, 3.0);
		noalias(result) -= 2 * triangular_prod<upper>(arg1uppercm,arg2);
		checkMatrixMatrixMultiply(arg1uppertest,arg2,result,3.0,-2);
	}
	

	diag(arg1lowertest) = repeat(1.0, dims);
	diag(arg1uppertest) = repeat(1.0, dims);
	std::cout << "\nchecking matrix-matrix prod V=AX unit" << std::endl;
	{
		std::cout<<"row major lower AX"<<std::endl;
		matrix<float, Orientation, hip_tag> result(dims, N, 3.0);
		noalias(result) = triangular_prod<unit_lower>(arg1lowerrm,arg2);
		checkMatrixMatrixMultiply(arg1lowertest,arg2,result,0.0,1);
	}
	{
		std::cout<<"column major lower AX"<<std::endl;
		matrix<float, Orientation, hip_tag> result(dims, N, 3.0);
		noalias(result) = triangular_prod<unit_lower>(arg1lowercm,arg2);
		checkMatrixMatrixMultiply(arg1lowertest,arg2,result,0.0,1);
	}
	{
		std::cout<<"row major upper AX"<<std::endl;
		matrix<float, Orientation, hip_tag> result(dims, N, 3.0);
		noalias(result) = triangular_prod<unit_upper>(arg1upperrm,arg2);
		checkMatrixMatrixMultiply(arg1uppertest,arg2,result,0.0,1);
	}
	{
		std::cout<<"column major upper AX"<<std::endl;
		matrix<float, Orientation, hip_tag> result(dims, N, 3.0);
		noalias(result) = triangular_prod<unit_upper>(arg1uppercm,arg2);
		checkMatrixMatrixMultiply(arg1uppertest,arg2,result,0.0,1);
	}
	//with prefactor
	{
		std::cout<<"row major lower AX"<<std::endl;
		matrix<float, Orientation, hip_tag> result(dims, N, 3.0);
		noalias(result) = -2 * triangular_prod<unit_lower>(arg1lowerrm,arg2);
		checkMatrixMatrixMultiply(arg1lowertest,arg2,result,0.0,-2);
	}
	{
		std::cout<<"column major lower AX"<<std::endl;
		matrix<float, Orientation, hip_tag> result(dims, N, 3.0);
		noalias(result) = -2 * triangular_prod<unit_lower>(arg1lowercm,arg2);
		checkMatrixMatrixMultiply(arg1lowertest,arg2,result,0.0,-2);
	}
	{
		std::cout<<"row major upper AX"<<std::endl;
		matrix<float, Orientation, hip_tag> result(dims, N, 3.0);
		noalias(result) = -2 * triangular_prod<unit_upper>(arg1upperrm,arg2);
		checkMatrixMatrixMultiply(arg1uppertest,arg2,result,0.0,-2);
	}
	{
		std::cout<<"column major upper AX"<<std::endl;
		matrix<float, Orientation, hip_tag> result(dims, N, 3.0);
		noalias(result) = -2 * triangular_prod<unit_upper>(arg1uppercm,arg2);
		checkMatrixMatrixMultiply(arg1uppertest,arg2,result,0.0,-2);
	}
	std::cout << "\nchecking matrix-matrix prod V+=AX unit" << std::endl;
	{
		std::cout<<"row major lower AX"<<std::endl;
		matrix<float, Orientation, hip_tag> result(dims, N, 3.0);
		noalias(result) += -2 * triangular_prod<unit_lower>(arg1lowerrm,arg2);
		checkMatrixMatrixMultiply(arg1lowertest,arg2,result,3.0,-2);
	}
	{
		std::cout<<"column major lower AX"<<std::endl;
		matrix<float, Orientation, hip_tag> result(dims, N, 3.0);
		noalias(result) += -2 * triangular_prod<unit_lower>(arg1lowercm,arg2);
		checkMatrixMatrixMultiply(arg1lowertest,arg2,result,3.0,-2);
	}
	{
		std::cout<<"row major upper AX"<<std::endl;
		matrix<float, Orientation, hip_tag> result(dims, N, 3.0);
		noalias(result) += -2 * triangular_prod<unit_upper>(arg1upperrm,arg2);
		checkMatrixMatrixMultiply(arg1uppertest,arg2,result,3.0,-2);
	}
	{
		std::cout<<"column major upper AX"<<std::endl;
		matrix<float, Orientation, hip_tag> result(dims, N, 3.0);
		noalias(result) += -2 * triangular_prod<unit_upper>(arg1uppercm,arg2);
		checkMatrixMatrixMultiply(arg1uppertest,arg2,result,3.0,-2);
	}
	std::cout << "\nchecking matrix-matrix prod V-=AX unit" << std::endl;
	{
		std::cout<<"row major lower AX"<<std::endl;
		matrix<float, Orientation, hip_tag> result(dims, N, 3.0);
		noalias(result) -= 2 * triangular_prod<unit_lower>(arg1lowerrm,arg2);
		checkMatrixMatrixMultiply(arg1lowertest,arg2,result,3.0,-2);
	}
	{
		std::cout<<"column major lower AX"<<std::endl;
		matrix<float, Orientation, hip_tag> result(dims, N, 3.0);
		noalias(result) -= 2 * triangular_prod<unit_lower>(arg1lowercm,arg2);
		checkMatrixMatrixMultiply(arg1lowertest,arg2,result,3.0,-2);
	}
	{
		std::cout<<"row major upper AX"<<std::endl;
		matrix<float, Orientation, hip_tag> result(dims, N, 3.0);
		noalias(result) -= 2 * triangular_prod<unit_upper>(arg1upperrm,arg2);
		checkMatrixMatrixMultiply(arg1uppertest,arg2,result,3.0,-2);
	}
	{
		std::cout<<"column major upper AX"<<std::endl;
		matrix<float, Orientation, hip_tag> result(dims, N, 3.0);
		noalias(result) -= 2 * triangular_prod<unit_upper>(arg1uppercm,arg2);
		checkMatrixMatrixMultiply(arg1uppertest,arg2,result,3.0,-2);
	}
}
BOOST_AUTO_TEST_CASE(Remora_triangular_prod_matrix_matrix) {
	triangular_prod_matrix_matrix_test(row_major());
	triangular_prod_matrix_matrix_test(column_major());
}

BOOST_AUTO_TEST_SUITE_END()
