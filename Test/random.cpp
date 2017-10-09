#define BOOST_TEST_MODULE Remora_Random
#include <boost/test/unit_test.hpp>
#include <boost/test/floating_point_comparison.hpp>

#include <remora/dense.hpp>
#include <remora/matrix_expression.hpp>
#include <remora/kernels/random.hpp>

#include <boost/math/distributions/normal.hpp>
#include <boost/math/distributions/uniform.hpp>
#include <boost/math/distributions/chi_squared.hpp>

#include <iostream>
using namespace remora;

bool chi_squared_gof(std::vector<std::size_t> const& bins, std::vector<double> const& pBin, std::size_t n){
	double stat = 0;
	for(std::size_t i = 0; i != bins.size(); ++i){
		double E = pBin[i] * n;
		stat += (bins[i]-E)*(bins[i]-E)/E;
	}
	
	boost::math::chi_squared dist(bins.size() - 1);
	double val = quantile(complement(dist, 0.02));
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
	
	std::vector<double> p(numBins,0);
	for(std::size_t i = 0;  i != p.size(); ++i){
		p[i] = cdf(dist,min+(i+1)*(max-min)/numBins) - cdf(dist,min+i*(max-min)/numBins);
	}
	
	chi_squared_gof(bins,p,numValid);
}

template<class V, class Dist>
void chi_squared_gof(matrix_expression<V, cpu_tag> const& m, Dist const& dist, typename V::value_type min, typename V::value_type max, std::size_t numBins){
	vector<typename V::value_type> v(m().size1() * m().size2());
	for(std::size_t i = 0; i != m().size1(); ++i){
		for(std::size_t j = 0; j != m().size2(); ++j){
			v(i*m().size2()+j) = m()(i,j);
		}
	}
	chi_squared_gof(v,dist,min,max,numBins);
}

BOOST_AUTO_TEST_SUITE (Remora_Random)

BOOST_AUTO_TEST_CASE(Remora_Random_Normal) {
	std::size_t Dimensions = 50000;
	std::mt19937 gen(42);
	boost::math::normal_distribution<> dist(-2,3);
	for(std::size_t i = 0; i != 10; ++i){
		vector<double> v(Dimensions);
		kernels::generate_normal(v, gen, -2, 9);
		chi_squared_gof(v,dist,-10,10,20);
		
		{
			matrix<double, row_major> m(100,Dimensions/100);
			kernels::generate_normal(m, gen, -2, 9);
			chi_squared_gof(m,dist,-10,10,20);
		}
		{
			matrix<double, column_major> m(100,Dimensions/100);
			kernels::generate_normal(m, gen, -2, 9);
			chi_squared_gof(m,dist,-10,10,20);
		}
	}
}

BOOST_AUTO_TEST_CASE(Remora_Random_uniform) {
	std::size_t Dimensions = 50000;
	std::mt19937 gen(42);
	boost::math::uniform_distribution<> dist(-5,7);
	for(std::size_t i = 0; i != 10; ++i){
		vector<double> v(Dimensions);
		kernels::generate_uniform(v, gen, -5, 7);
		chi_squared_gof(v,dist,-5,7,20);
		
		{
			matrix<double, row_major> m(100,Dimensions/100);
			kernels::generate_uniform(m, gen, -5, 7);
			chi_squared_gof(m,dist,-5,7,20);
		}
		{
			matrix<double, column_major> m(100,Dimensions/100);
			kernels::generate_uniform(m, gen, -5, 7);
			chi_squared_gof(m,dist,-5,7,20);
		}
	}
}


BOOST_AUTO_TEST_CASE(Remora_Random_discrete) {
	std::size_t Dimensions = 100000;
	std::mt19937 gen(42);
	boost::math::uniform_distribution<> dist(-5,7);
	for(std::size_t i = 0; i != 10; ++i){
		vector<double> v(Dimensions);
		kernels::generate_discrete(v, gen, -5, 7);
		chi_squared_gof(v,dist,-5,7,13);
		
		{
			matrix<double, row_major> m(100,Dimensions/100);
			kernels::generate_discrete(m, gen, -5, 7);
			chi_squared_gof(m,dist,-5,7,13);
		}
		{
			matrix<double, column_major> m(100,Dimensions/100);
			kernels::generate_discrete(m, gen, -5, 7);
			chi_squared_gof(m,dist,-5,7,13);
		}
	}
}

BOOST_AUTO_TEST_SUITE_END()
