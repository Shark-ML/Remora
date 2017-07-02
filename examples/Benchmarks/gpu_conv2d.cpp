#include <remora/kernels/conv2d.hpp>
#include <remora/remora.hpp>
#include "Timer.hpp"
#include <iostream>
using namespace remora;

template<class E1, class E2>
void benchmark(
	matrix_expression<E1, cpu_tag> const& image,
	vector_expression<E2, cpu_tag> const& filter,
	std::size_t num_channels,
	std::size_t num_filters,
	std::size_t image_size1,
	std::size_t image_size2,
	std::size_t filter_size
){
	std::size_t output_size1 = image_size1 - filter_size +1;
	std::size_t output_size2 = image_size2 - filter_size +1;
	typedef typename E1::value_type value_type;

	remora::matrix<value_type, row_major, gpu_tag> image_gpu = copy_to_gpu(image);
	remora::vector<value_type, gpu_tag> filter_gpu = copy_to_gpu(filter);
	remora::matrix<value_type, row_major, gpu_tag> out_gpu(image().size1(), output_size1 * num_filters * output_size2, 0.0);
	kernels::conv2d(image_gpu,filter_gpu,out_gpu, num_channels, num_filters, image_size1, image_size2, filter_size, filter_size,0,0);
	out_gpu.queue().finish();
	double minOptTime = std::numeric_limits<double>::max();
	for(std::size_t i = 0; i != 10; ++i){
		Timer time;
		kernels::conv2d(image_gpu,filter_gpu,out_gpu, num_channels, num_filters, image_size1, image_size2, filter_size, filter_size,0,0);
		out_gpu.queue().finish();
		minOptTime = std::min(minOptTime,time.stop());
	}

	double mults = output_size1 * output_size2 * filter_size * filter_size * num_filters * num_channels;
	double flops = image().size1() * mults /1024/1024/minOptTime;

	std::cout<<output_size1<<"\t"<<filter_size<<"\t"<<num_channels<<"\t"<< num_filters<<"\t";
	std::cout<<"\t"<<flops<< std::endl;
}

template<class T>
void benchmark(std::size_t num_channels, std::size_t num_outputs, std::size_t num_images){
	std::cout<<"Flops"<<std::endl;
	for(std::size_t filterSize = 4; filterSize != 32; filterSize *= 2){
		for(std::size_t iter = 0; iter != 5; ++iter){
			std::size_t sizeOut1 = 3 + 16 * (2<<iter);
			std::size_t sizeOut2 = 3 + 16 * (2<<iter);
			std::size_t sizeIm1 = sizeOut1 + filterSize-1;
			std::size_t sizeIm2 = sizeOut2 + filterSize-1;

			remora::matrix<T> image(num_images, num_channels * sizeIm1 * sizeIm2);
			remora::vector<T> filter(num_channels * num_outputs *  filterSize * filterSize);
			
			for(std::size_t im = 0; im != num_images; ++im){
				for(std::size_t i = 0; i != num_channels * sizeIm1; ++i){
					for(std::size_t j = 0; j != sizeIm2; ++j){
						image(im, i * sizeIm2 + j)  = 1.0/(num_channels * sizeOut1)*i + 0.1 - (0.1/sizeOut2)*j;
					}
				}
			}
			for(std::size_t i = 0; i != num_channels * num_outputs * filterSize; ++i){
				for(std::size_t j = 0; j != filterSize; ++j){
					filter(i * filterSize + j)  = 1.0/(num_channels * filterSize)*i + 0.1 - (0.1/filterSize)*j;
				}
			}

			benchmark(image,filter,num_channels,num_outputs, sizeIm1, sizeIm2, filterSize);
		}
	}
}


int main(int argc, char **argv) {
	std::cout<<"performance float"<<std::endl;
	benchmark<float>(8,32,16);
}
