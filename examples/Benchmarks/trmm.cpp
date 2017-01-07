#include <remora/remora.hpp>
#include "Timer.hpp"
#include <iostream>
using namespace remora;

template<class Triangular, class AMat, class BMat, class CMat>
double benchmark(
	matrix_expression<AMat, cpu_tag> const& A,
	matrix_expression<BMat, cpu_tag> const& B,
	matrix_expression<CMat, cpu_tag> & C
){
	double minTime = std::numeric_limits<double>::max();
	for(std::size_t i = 0; i != 10; ++i){
		Timer time;
		noalias(C) += triangular_prod<Triangular>(A,B);
		minTime = std::min(minTime,time.stop());
	}
	return (0.5*A().size1()*A().size2()*B().size2())/minTime/1024/1024;
}

int main(int argc, char **argv) {
	std::size_t size = 100;
	std::cout<<"Mega Flops"<<std::endl;
	for(std::size_t iter = 0; iter != 10; ++iter){
		matrix<double,row_major> Arow(size,size);
		for(std::size_t i = 0; i != size; ++i){
			for(std::size_t k = 0; k != size; ++k){
				Arow(i,k)  = 0.1/size*i+0.1/size*k;
			}
		}

		matrix<double,row_major> Brow(size,size);
		for(std::size_t k = 0; k != size; ++k){
			for(std::size_t j = 0; j != size; ++j){
				Brow(k,j) = 0.1/size*j+0.1/size*k;
			}
		}
		matrix<double,column_major> Acol = Arow;
		matrix<double,column_major> Bcol = Brow;

		matrix<double,row_major> Crow(size,size,0.0);
		matrix<double,column_major> Ccol(size,size,0.0);
		std::cout<<size<<"\t row major result - lower\t"<<benchmark<lower>(Arow,Brow,Crow)<<"\t"<< benchmark<lower>(Acol,Brow,Crow)
		<<"\t"<< benchmark<lower>(Arow,Bcol,Crow) <<"\t" <<benchmark<lower>(Acol,Bcol,Crow) <<std::endl;
		std::cout<<size<<"\t row major result - upper\t"<<benchmark<upper>(Arow,Brow,Crow)<<"\t"<< benchmark<upper>(Acol,Brow,Crow)
		<<"\t"<< benchmark<upper>(Arow,Bcol,Crow) <<"\t" <<benchmark<upper>(Acol,Bcol,Crow) <<std::endl;
		std::cout<<size<<"\t column major result - lower\t"<<benchmark<lower>(Arow,Brow,Ccol)<<"\t"<< benchmark<lower>(Acol,Brow,Ccol)
		<<"\t"<< benchmark<lower>(Arow,Bcol,Ccol) <<"\t" <<benchmark<lower>(Acol,Bcol,Ccol) <<std::endl;
		std::cout<<size<<"\t column major result - upper\t"<<benchmark<upper>(Arow,Brow,Ccol)<<"\t"<< benchmark<upper>(Acol,Brow,Ccol)
		<<"\t"<< benchmark<upper>(Arow,Bcol,Ccol) <<"\t" <<benchmark<upper>(Acol,Bcol,Ccol) <<std::endl;
		std::cout<<std::endl;
		size *=2;
	}
}
