#define BOOST_TEST_MODULE Remora_GPU_Conv2d
#include <boost/test/unit_test.hpp>
#include <boost/test/floating_point_comparison.hpp>
#include <boost/mpl/list.hpp>

#include <remora/io.hpp>
#include <iostream>

#include <remora/kernels/conv2d.hpp>
#include <remora/matrix_proxy.hpp>//fixme: required by assign :(
#include <remora/matrix.hpp>
#include <remora/vector_expression.hpp>

using namespace remora;

void test(
	std::size_t image_size1, std::size_t image_size2,
	std::size_t filter_size1, std::size_t filter_size2,
	std::size_t num_channels,
	std::size_t num_filters,
	std::size_t num_images,
	std::size_t padding_height = 0,
	std::size_t padding_width = 0
){
	//create filter on CPU
	vector<float> filter(num_channels * num_filters *  filter_size1 * filter_size2);
	{
		std::size_t lin_elem = 0;
		for(std::size_t f = 0; f != num_filters; ++f){
			for(std::size_t i = 0; i !=  filter_size1; ++i){
				for(std::size_t j = 0; j != filter_size2; ++j){
					for(std::size_t c = 0; c != num_channels; ++c, ++lin_elem){
						double val = 1.0/(num_channels * filter_size1)*i + 0.1 - (0.1/filter_size2)*j+0.01*f-0.01*c;
						filter(lin_elem) = val;
					}
				}
			}
		}
	}
	
	//Create images on CPU
	matrix<float> image(num_images, num_channels * image_size1 *  image_size2);
	//create images and ground truth
	for(std::size_t im = 0; im != num_images; ++im){
		std::size_t lin_elem = 0;
		for(std::size_t i = 0; i !=  image_size1; ++i){
			for(std::size_t j = 0; j != image_size2; ++j){
				for(std::size_t c = 0; c != num_channels; ++c, ++lin_elem){
					image(im,lin_elem) = 1.0/(num_channels * image_size1)*i + 0.1 - (0.1/image_size2)*j;
				}
			}
		}
	}
	
	//copy to Device
	vector<float,gpu_tag> filter_gpu = copy_to_gpu(filter);
	matrix<float,row_major, gpu_tag> image_gpu = copy_to_gpu(image);
	
	//Reserve enough space for output
	
	std::size_t output_size1 = image_size1 - filter_size1 + 1 + padding_height;
	std::size_t output_size2 = image_size2 - filter_size2 + 1 + padding_width;
	matrix<float> out(num_images, output_size1 * output_size2 * num_filters, 0.0);
	matrix<float,row_major,gpu_tag> out_gpu(num_images, output_size1 * output_size2 * num_filters, 0.0);
	
	
	//compute baseline and gpu result
	kernels::conv2d(
		image,filter,out,num_channels, num_filters, 
		image_size1, image_size2, filter_size1, filter_size2,
		padding_height, padding_width
	);
	
	kernels::conv2d(
		image_gpu,filter_gpu,out_gpu,num_channels, num_filters, 
		image_size1, image_size2, filter_size1, filter_size2,
		padding_height, padding_width
	);
	
	//copy result back and test
	matrix<float,row_major> out_cpu = copy_to_cpu(out_gpu);
	
	for(std::size_t im = 0; im != num_images; ++im){
		for(std::size_t k = 0; k != out.size2(); ++k){
			BOOST_CHECK_CLOSE(out(im,k),out_cpu(im,k),1.e-2);
		}
	}	
}



BOOST_AUTO_TEST_SUITE(Remora_Conv2d)

BOOST_AUTO_TEST_CASE(conv2d_test) {
	//~ test(32,16,4,8,5,1,1);
	test(16,12,4,8,4,4,3);
	//~ test(57,33,7,3,22,15,3);
}

BOOST_AUTO_TEST_SUITE_END()
