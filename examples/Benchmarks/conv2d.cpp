#include <remora/kernels/conv2d.hpp>
#include <remora/remora.hpp>
#include "Timer.hpp"
#include <iostream>
using namespace remora;
using namespace std;

template<class E1, class E2>
void benchmark(
	matrix_expression<E1, cpu_tag> const& image,
	matrix_expression<E2, cpu_tag> const& filter,
	std::size_t num_channels,
	std::size_t num_filters
){
	std::size_t filter_size = filter().size2();
	std::size_t image_size1 = image().size1()/num_channels;
	std::size_t image_size2 = image().size2();
	std::size_t output_size1 = image_size1 - filter_size +1;
	std::size_t output_size2 = image_size2 - filter_size +1;
	typedef typename E1::value_type value_type;

	matrix<value_type> out(output_size1 * num_filters, output_size2 ,0.0);
	double minOptTime = std::numeric_limits<double>::max();
	for(std::size_t i = 0; i != 20; ++i){
		Timer time;
		kernels::conv2d(image,filter,out, num_channels, num_filters);
		minOptTime = min(minOptTime,time.stop());
	}

	double mults = output_size1 * output_size2 * filter_size * filter_size * num_filters * num_channels;
	double flops = mults /1024/1024/minOptTime;

	std::cout<<output_size1<<"\t"<<filter_size<<"\t"<<num_channels<<"\t"<< num_filters<<"\t";
	std::cout<<"\t"<<flops<< std::endl;
}


int main(int argc, char **argv) {
	std::cout<<"Flops"<<std::endl;
	std::size_t num_channels = 8;
	std::size_t num_outputs = 16;
	std::cout<<"performance float"<<std::endl;
	for(std::size_t filterSize = 4; filterSize != 32; filterSize *= 2){
		for(std::size_t iter = 0; iter != 6; ++iter){
			std::size_t sizeOut1 = 3+16 * (2<<iter);
			std::size_t sizeOut2 = 3+16 * (2<<iter);
			std::size_t sizeIm1 = sizeOut1 + filterSize-1;
			std::size_t sizeIm2 = sizeOut2 + filterSize-1;

			matrix<float> image(num_channels * sizeIm1 , sizeIm2);
			matrix<float> filter(num_channels * num_outputs *  filterSize, filterSize);

			for(std::size_t i = 0; i != num_channels * sizeIm1; ++i){
				for(std::size_t j = 0; j != sizeIm2; ++j){
					image(i,j)  = 1.0/(num_channels * sizeOut1)*i + 0.1 - (0.1/sizeOut2)*j;
				}
			}
			for(std::size_t i = 0; i != num_channels * num_outputs * filterSize; ++i){
				for(std::size_t j = 0; j != filterSize; ++j){
					filter(i,j)  = 1.0/(num_channels * filterSize)*i + 0.1 - (0.1/filterSize)*j;
				}
			}

			benchmark(image,filter,num_channels,num_outputs);
		}
	}
	num_outputs = 8;
	std::cout<<"performance double"<<std::endl;
	for(std::size_t filterSize = 4; filterSize != 32; filterSize *= 2){
		for(std::size_t iter = 0; iter != 6; ++iter){
			std::size_t sizeOut1 = (3+16 * 2<<iter);
			std::size_t sizeOut2 = (3+16 * 2<<iter);
			std::size_t sizeIm1 = sizeOut1 + filterSize-1;
			std::size_t sizeIm2 = sizeOut2 + filterSize-1;

			matrix<double> image(num_channels * sizeIm1 , sizeIm2);
			matrix<double> filter(num_channels * num_outputs *  filterSize, filterSize);

			for(std::size_t i = 0; i != num_channels * sizeIm1; ++i){
				for(std::size_t j = 0; j != sizeIm2; ++j){
					image(i,j)  = 1.0/(num_channels * sizeOut1)*i + 0.1 - (0.1/sizeOut2)*j;
				}
			}
			for(std::size_t i = 0; i != num_channels * num_outputs * filterSize; ++i){
				for(std::size_t j = 0; j != filterSize; ++j){
					filter(i,j)  = 1.0/(num_channels * filterSize)*i + 0.1 - (0.1/filterSize)*j;
				}
			}

			benchmark(image,filter,num_channels,num_outputs);
		}
	}
}
