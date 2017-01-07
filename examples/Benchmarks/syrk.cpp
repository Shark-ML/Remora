#include <remora/remora.hpp>
#include "Timer.hpp"
#include <iostream>
using namespace remora;

template<class Triangular, class AMat, class CMat>
double benchmark(
	matrix_expression<AMat, cpu_tag> const& A,
	matrix_expression<CMat, cpu_tag> & C
){
	double minTime = std::numeric_limits<double>::max();
	for(std::size_t i = 0; i != 10; ++i){
		Timer time;
		kernels::syrk<Triangular::is_upper>(A,C,2.0);
		minTime = std::min(minTime,time.stop());
	}
	return (0.5*A().size1()*A().size2()*A().size1())/minTime/1024/1024;
}

int main(int argc, char **argv) {
	std::size_t size = 100;
	std::cout<<"Mega Flops"<<std::endl;
	for(std::size_t iter = 0; iter != 5; ++iter){
		matrix<double,row_major> Arow(size,size);
		for(std::size_t i = 0; i != size; ++i){
			for(std::size_t k = 0; k != size; ++k){
				Arow(i,k)  = 0.1/size*i+0.1/size*k;
			}
		}
		matrix<double,column_major> Acol = Arow;

		matrix<double,row_major> Crow(size,size,0.0);
		matrix<double,column_major> Ccol(size,size,0.0);
		std::cout<<size<<"\trow major result - lower\t"<<benchmark<lower>(Arow,Crow)<<"\t"<< benchmark<lower>(Acol,Crow)<<std::endl;
		std::cout<<size<<"\trow major result - upper\t"<<benchmark<upper>(Arow,Crow)<<"\t"<< benchmark<upper>(Acol,Crow)<<std::endl;
		std::cout<<size<<"\tcolumn major result - lower\t"<<benchmark<lower>(Arow,Ccol)<<"\t"<< benchmark<lower>(Acol,Ccol)<<std::endl;
		std::cout<<size<<"\tcolumn major result - upper\t"<<benchmark<upper>(Arow,Ccol)<<"\t"<< benchmark<upper>(Acol,Ccol)<<std::endl;


		std::cout<<std::endl;
		size *=2;
	}
}
