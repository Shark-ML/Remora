#define BOOST_TEST_MODULE Remora_eigensymm
#include <boost/test/unit_test.hpp>
#include <boost/test/floating_point_comparison.hpp>

#include <iostream>

#include <remora/decompositions.hpp>
#include <remora/matrix.hpp>
#include <remora/vector.hpp>

using namespace remora;

BOOST_AUTO_TEST_SUITE (Remora_eigensymm)

matrix<double> createSymm(std::size_t dimensions, std::size_t rank = 0){
	if(rank == 0) rank = dimensions;
	matrix<double> R(dimensions,dimensions,0.0);
	
	for(std::size_t i = 0; i != dimensions; ++i){
		for(std::size_t j = 0; j <std::min(i,rank); ++j){
			R(i,j) = 0.2/std::abs((int)i -(int)j);
		}
		if(i < rank)
			R(i,i) = 0.5/dimensions*i+1;
	}
	matrix<double> A = prod(R,trans(R));
	if(rank != dimensions){
		for(std::size_t i = 0; i != rank/2; ++i){
			A.swap_rows(2*i,dimensions-i-1);
			A.swap_columns(2*i,dimensions-i-1);
		}
	}
	return A;
}

BOOST_AUTO_TEST_CASE( Remora_eigensymm_decomposition)
{
	std::size_t Dimensions = 123;
	matrix<double> A = createSymm(Dimensions);
		
	symm_eigenvalue_decomposition<matrix<double> > solver(A);
	
	matrix<double> Atest = solver.Q() % to_diagonal(solver.D()) % trans(solver.Q());
	BOOST_CHECK_SMALL(norm_inf(Atest-A),norm_inf(A) * 1.e-12);

}

BOOST_AUTO_TEST_CASE( Remora_eigensymm_solve )
{
	std::size_t Dimensions = 153;
	std::size_t K = 35;
	//first generate a suitable eigenvalue problem matrix A
	matrix<double> A = createSymm(Dimensions);
		
	symm_eigenvalue_decomposition<matrix<double> > solver(A);
	cholesky_decomposition<matrix<double> > solver_cholesky(A);
	
	matrix<double> B(Dimensions,K);
	for(std::size_t i = 0; i != K; ++i){
		for(std::size_t j = 0; j != Dimensions; ++j){
			B(j,i) = (1.0 + j+K)/Dimensions; 
		}
	}
	
	vector<double> b(Dimensions);
	for(std::size_t j = 0; j != Dimensions; ++j){
		b(j) = (1.0 + j)/Dimensions; 
	}
	
	{
		vector<double> sol=b;
		vector<double> sol2=b;
		solver.solve(sol,left());
		solver_cholesky.solve(sol2,left());
		BOOST_CHECK_SMALL(norm_2(sol - sol2), norm_2(sol2)*1.e-8);
	}
	{
		vector<double> sol=b;
		vector<double> sol2=b;
		solver.solve(sol,right());
		solver_cholesky.solve(sol2,right());
		BOOST_CHECK_SMALL(norm_2(sol - sol2), norm_2(sol2)*1.e-8);
	}
	
	{
		matrix<double> sol=B;
		matrix<double> sol2=B;
		solver.solve(sol,left());
		solver_cholesky.solve(sol2,left());
		BOOST_CHECK_SMALL(norm_inf(sol - sol2), norm_inf(sol2)*1.e-8);
	}
	{
		matrix<double> sol=trans(B);
		matrix<double> sol2=trans(B);
		solver.solve(sol,right());
		solver_cholesky.solve(sol2,right());
		BOOST_CHECK_SMALL(norm_frobenius(sol - sol2), norm_frobenius(sol2)*1.e-8);
	}
}
BOOST_AUTO_TEST_SUITE_END()
