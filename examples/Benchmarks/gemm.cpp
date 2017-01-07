#include <remora/remora.hpp>
#include "Timer.hpp"
#include <iostream>
using namespace remora;

template<class AMat, class BMat, class CMat>
double benchmark(
	matrix_expression<AMat, cpu_tag> const& A,
	matrix_expression<BMat, cpu_tag> const& B,
	matrix_expression<CMat, cpu_tag> & C
){
	double minTime = std::numeric_limits<double>::max();
	for(std::size_t i = 0; i != 10; ++i){
		Timer time;
		noalias(C) += prod(A,B);
		minTime = std::min(minTime,time.stop());
	}
	return (A().size1()*A().size2()*B().size2())/minTime/1024/1024;
}

int main(int argc, char **argv) {
	std::size_t size = 100;
	std::cout<<"Flops"<<std::endl;
	for(std::size_t iter = 0; iter != 5; ++iter){
		std::size_t middle = size;
		matrix<double,row_major> Arow(size,middle);
		for(std::size_t i = 0; i != size; ++i){
			for(std::size_t k = 0; k != middle; ++k){
				Arow(i,k)  = 0.1/size*i+0.1/size*k;
			}
		}

		matrix<double,row_major> Brow(middle,size);
		for(std::size_t k = 0; k != middle; ++k){
			for(std::size_t j = 0; j != size; ++j){
				Brow(k,j) = 0.1/size*j+0.1/size*k;
			}
		}
		matrix<double,column_major> Acol = Arow;
		matrix<double,column_major> Bcol = Brow;

		matrix<double,row_major> Crow(size,size,0.0);
		matrix<double,column_major> Ccol(size,size,0.0);
		std::cout<<size<<"\t row major results\t"<<benchmark(Arow,Brow,Crow)<<"\t"<< benchmark(Acol,Brow,Crow)
		<<"\t"<< benchmark(Arow,Bcol,Crow) <<"\t" <<benchmark(Acol,Bcol,Crow) <<std::endl;
		std::cout<<size<<"\t column major results\t"<<benchmark(Arow,Brow,Ccol)<<"\t"<< benchmark(Acol,Brow,Ccol)
		<<"\t"<< benchmark(Arow,Bcol,Ccol) <<"\t" <<benchmark(Acol,Bcol,Ccol) <<std::endl;
		size *=2;
	}
}
