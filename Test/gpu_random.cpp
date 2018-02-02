#define BOOST_COMPUTE_DEBUG_KERNEL_COMPILATION
#define BOOST_TEST_MODULE Remora_GPU_Random
#include <boost/test/unit_test.hpp>
#include <boost/test/floating_point_comparison.hpp>

#include <remora/dense.hpp>
#include <remora/device_copy.hpp>
#include <remora/matrix_expression.hpp>
#include <remora/random.hpp>

#include <boost/math/distributions/normal.hpp>
#include <boost/math/distributions/uniform.hpp>
#include <boost/math/distributions/chi_squared.hpp>

#include <iostream>
using namespace remora;

void chi_squared_gof(std::vector<std::size_t> const& bins, std::vector<float> const& pBin, std::size_t n){
	float stat = 0;
	for(std::size_t i = 0; i != bins.size(); ++i){
		float E = pBin[i] * n;
		stat += (bins[i]-E)*(bins[i]-E)/E;
	}
	
	boost::math::chi_squared dist(bins.size() - 1);
	float val = quantile(complement(dist, 0.001));
	std::cout<<stat<<" "<<val<<std::endl;
	BOOST_CHECK( stat < val);
}

template<class V, class Dist>
void chi_squared_gof(vector_expression<V, cpu_tag> const& v, Dist const& dist, typename V::value_type min, typename V::value_type max, std::size_t numBins){
	std::vector<std::size_t> bins(numBins,0);
	std::size_t numValid = 0;
	for(auto val:v()){
		if(val < min || val > max) continue;
		std::size_t bin = std::size_t( numBins * (val - min)/(max-min));
		if(bin == numBins)
			bin -= 1;
		++bins[bin];
		++numValid;
	}
	BOOST_REQUIRE_GT(numValid, std::size_t(0.9 * v().size()));
	
	std::vector<float> p(numBins,0);
	double sumpi = 0.0;
	for(std::size_t i = 0;  i != p.size(); ++i){
		p[i] = cdf(dist,min+(i+1)*(max-min)/numBins) - cdf(dist,min+i*(max-min)/numBins);
		sumpi += p[i];
	}
	for(std::size_t i = 0;  i != p.size(); ++i){
		p[i] /= sumpi;
	}
	
	chi_squared_gof(bins,p,numValid);
}

template<class V, class Dist>
void chi_squared_gof(matrix_expression<V, gpu_tag> const& m_gpu, Dist const& dist, typename V::value_type min, typename V::value_type max, std::size_t numBins){
	matrix<typename V::value_type> m = copy_to_cpu(m_gpu);
	vector<typename V::value_type> v(m.size1() * m.size2());
	for(std::size_t i = 0; i != m.size1(); ++i){
		for(std::size_t j = 0; j != m.size2(); ++j){
			v(i*m.size2()+j) = m(i,j);
		}
	}
	chi_squared_gof(v,dist,min,max,numBins);
}

template<class V, class Dist>
void chi_squared_gof(vector_expression<V, gpu_tag> const& v_gpu, Dist const& dist, typename V::value_type min, typename V::value_type max, std::size_t numBins){
	vector<typename V::value_type> v = copy_to_cpu(v_gpu);
	chi_squared_gof(v,dist,min,max,numBins);
}

BOOST_AUTO_TEST_SUITE (Remora_GPU_Random)

BOOST_AUTO_TEST_CASE(Remora_Random_Normal) {
	std::size_t Dimensions = 30000;
	std::mt19937 gen(42);
	boost::math::normal_distribution<> dist(-2,3);
	for(std::size_t i = 0; i != 10; ++i){
		//test kernels
		{
			vector<float, gpu_tag> v(Dimensions);
			kernels::generate_normal(v, gen, -2, 9);
			chi_squared_gof(v,dist, -10, 10,20);
		}
		{
			matrix<float, row_major, gpu_tag> m(77,Dimensions/77);
			kernels::generate_normal(m, gen, -2, 9);
			chi_squared_gof(m,dist, -10, 10,20);
		}
		{
			matrix<float, column_major, gpu_tag> m(77,Dimensions/77);
			kernels::generate_normal(m, gen, -2, 9);
			chi_squared_gof(m,dist, -10,10,20);
		}
		//test expressions
		{
			vector<float, gpu_tag> v = 3.0*normal(gen, Dimensions, -2.0/3.0, 1.0, gpu_tag());
			chi_squared_gof(v,dist, -10, 10,20);
		}
		{
			vector<float, gpu_tag> v(Dimensions,1);
			noalias(v) += 3.0*normal(gen, Dimensions, -1.0, 1.0, gpu_tag());
			chi_squared_gof(v,dist, -10, 10,20);
		}
		{
			matrix<float, row_major, gpu_tag> m = 3.0*normal(gen, 77, Dimensions/77, -2.0/3.0, 1.0, gpu_tag());
			chi_squared_gof(m,dist, -10, 10,20);
		}
		{
			matrix<float, row_major, gpu_tag> m(77,Dimensions/77,1);
			noalias(m) += 3.0*normal(gen, 77, Dimensions/77, -1.0, 1.0, gpu_tag());
			chi_squared_gof(m,dist, -10, 10,20);
		}
	}
}

BOOST_AUTO_TEST_CASE(Remora_Random_uniform) {
	std::size_t Dimensions = 10000;
	std::mt19937 gen(42);
	boost::math::uniform_distribution<> dist(-5,7);
	for(std::size_t i = 0; i != 10; ++i){
		{
			vector<float, gpu_tag> v(Dimensions);
			kernels::generate_uniform(v, gen, -5,7);
			chi_squared_gof(v,dist,-5,7,20);
		}
		{
			matrix<float, row_major, gpu_tag> m(77,Dimensions/77);
			kernels::generate_uniform(m, gen, -5, 7);
			chi_squared_gof(m,dist,-5,7,20);
		}
		{
			matrix<float, column_major, gpu_tag> m(77,Dimensions/77);
			kernels::generate_uniform(m, gen, -5, 7);
			chi_squared_gof(m,dist,-5,7,20);
		}
		//test expressions
		{
			vector<float, gpu_tag> v = 3.0*uniform(gen, Dimensions, -5.0/3.0, 7.0/3.0, gpu_tag());
			chi_squared_gof(v,dist,-5,7,20);
		}
		{
			vector<float, gpu_tag> v(Dimensions,1);
			noalias(v) += 3.0*uniform(gen, Dimensions, -6/3.0, 6/3.0, gpu_tag());
			chi_squared_gof(v,dist,-5,7,20);
		}
		{
			matrix<float, row_major, gpu_tag> m = 3.0*uniform(gen, 77, Dimensions/77, -5.0/3.0, 7.0/3.0, gpu_tag());
			chi_squared_gof(m,dist,-5,7,20);
		}
		{
			matrix<float, row_major, gpu_tag> m(77,Dimensions/77,1);
			noalias(m) += 3.0*uniform(gen, 77, Dimensions/77, -6/3.0, 6/3.0, gpu_tag());
			chi_squared_gof(m,dist,-5,7,20);
		}
	}
}


BOOST_AUTO_TEST_CASE(Remora_Random_discrete) {
	std::size_t Dimensions = 50000;
	std::mt19937 gen(42);
	boost::math::uniform_distribution<> dist(-5,7);
	for(std::size_t i = 0; i != 10; ++i){
		vector<float,gpu_tag> v(Dimensions);
		kernels::generate_discrete(v, gen, -5, 7);
		chi_squared_gof(v,dist,-5,7,13);
		
		{
			matrix<float, row_major, gpu_tag> m(77,Dimensions/77);
			kernels::generate_discrete(m, gen, -5, 7);
			chi_squared_gof(m,dist,-5,7,13);
		}
		{
			matrix<float, column_major, gpu_tag> m(77,Dimensions/77);
			kernels::generate_discrete(m, gen, -5, 7);
			chi_squared_gof(m,dist,-5,7,13);
		}
	}
}

BOOST_AUTO_TEST_SUITE_END()
