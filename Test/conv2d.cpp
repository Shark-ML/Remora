#define BOOST_TEST_MODULE Remora_Conv2d
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

template<class E1, class E2, class M>
void conv2dTest(
	matrix_expression<E1, cpu_tag> const& image,
	matrix_expression<E2, cpu_tag> const& filter,
	matrix_expression<M, cpu_tag>& output,
	std::size_t num_channels,
	std::size_t num_filters
){
	std::size_t filter_size1 = filter().size1()/(num_channels * num_filters);
	std::size_t filter_size2 = filter().size2();
	std::size_t output_size1 = output().size1() / num_filters;
	std::size_t output_size2 = output().size2();
	std::size_t image_size1 = image().size1()/num_channels;
	
	output().clear();
	
	for(std::size_t f = 0; f != num_filters; ++f){
		std::size_t start_out1 = f * output_size1;
		for(std::size_t c = 0; c != num_channels; ++c){
			std::size_t start_filter1 = (f * num_channels + c) * filter_size1;
			std::size_t start_image1 = c * image_size1;			
			for(std::size_t i = 0; i != output_size1; ++i){
				for(std::size_t j = 0; j != output_size2; ++j){
					for(std::size_t i0 = 0; i0 != filter_size1; ++i0){
						for(std::size_t j0 = 0; j0 != filter_size2; ++j0){
							output()(start_out1 + i,j) += image()(start_image1 + i + i0,j + j0) * filter()(start_filter1 + i0, j0); 
						}
					}
				}
			}
		}
	}
}


template<class T>
void test(
	std::size_t image_size1, std::size_t image_size2,
	std::size_t filter_size1, std::size_t filter_size2,
	std::size_t num_channels,
	std::size_t num_filters
){
	matrix<T> image(
		num_channels * image_size1,image_size2,0.0
	);
	vector<T> image_lin(num_channels * image_size1 *  image_size2);
	matrix<T> filter(num_channels * num_filters *  filter_size1, filter_size2);
	vector<T> filter_lin(num_channels * num_filters *  filter_size1 * filter_size2);
	
	std::size_t lin_elem = 0;
	for(std::size_t c = 0; c != num_channels; ++c){
		for(std::size_t i = 0; i !=  image_size1; ++i){
			for(std::size_t j = 0; j != image_size2; ++j, ++lin_elem){
				int di = c*image_size1;
				image(di + i,j)  = 1.0/(num_channels * image_size1)*i + 0.1 - (0.1/image_size2)*j;
				image_lin(lin_elem) = image(di + i,j);
			}
		}
	}
	lin_elem = 0;
	for(std::size_t i = 0; i != num_channels * num_filters * filter_size1; ++i){
		for(std::size_t j = 0; j != filter_size2; ++j, ++lin_elem){
			filter(i,j)  = 1.0/(num_channels * filter_size1)*i + 0.1 - (0.1/filter_size2)*j;
			filter_lin(lin_elem) = filter(i,j);
		}
	}
	std::size_t output_size1 = image_size1 - filter_size1 + 1;
	std::size_t output_size2 = image_size2 - filter_size2 + 1;

	vector<T> out_lin(output_size1 * output_size2 * num_filters, 0.0);
	matrix<T> outTest(output_size1 * num_filters, output_size2, 0.0);
	
	kernels::conv2d(
		image_lin,filter_lin,out_lin,num_channels, num_filters, 
		image_size1, image_size2, filter_size1, filter_size2,false
	);
	conv2dTest(image,filter,outTest,num_channels, num_filters);

	for(std::size_t k = 0; k != num_filters; ++k){
		for(std::size_t i = 0; i != output_size1; ++i){
			for(std::size_t j = 0; j != output_size2; ++j){
				double val = out_lin(k * output_size1 * output_size2 + i * output_size2 + j);
				double valTest = outTest(k * output_size1 + i,j);
				BOOST_CHECK_CLOSE(val,valTest,1.e-3);
			}
		}
	}
}


template<class T>
void test_im_to_mat_pad(
	std::size_t image_size1, std::size_t image_size2,
	std::size_t filter_size1, std::size_t filter_size2,
	std::size_t num_channels
){
	std::size_t image_size1pad = image_size1 + filter_size1-1;
	std::size_t image_size2pad = image_size2 + filter_size2-1;
	vector<T> image_lin(num_channels * image_size1 *  image_size2,0);
	vector<T> image_lin_pad(num_channels * image_size1pad *  image_size2pad,0);
	for(std::size_t c = 0; c != num_channels; ++c){
		std::size_t lin_elem = c*image_size1*image_size2;
		std::size_t lin_elem_pad = c*image_size1pad*image_size2pad + ((filter_size1-1)/2)*image_size2pad+(filter_size2-1)/2;
		for(std::size_t i = 0; i !=  image_size1; ++i){
			for(std::size_t j = 0; j != image_size2; ++j, ++lin_elem_pad, ++lin_elem){
				double v = 1.0/(num_channels * image_size1)*i + 0.1 - (0.1/image_size2)*j;
				image_lin(lin_elem) = v;
				image_lin_pad(lin_elem_pad) = v;
			}
			lin_elem_pad += filter_size2-1;
		}
	}

	std::size_t output_rows_per_filter = image_size1 * image_size2;
	std::size_t filter_size = num_channels * filter_size1 * filter_size2;
	matrix<T> output(output_rows_per_filter, filter_size,-1.0);
	matrix<T> output_pad(output_rows_per_filter, filter_size);
	bindings::im2mat(image_lin_pad,output_pad,num_channels, image_size1pad, image_size2pad, filter_size1, filter_size2);
	bindings::im2mat_pad(image_lin,output,num_channels, image_size1, image_size2, filter_size1, filter_size2, filter_size1-1, filter_size2-1);
	for(std::size_t i = 0; i != output_rows_per_filter; ++i){
		BOOST_CHECK_SMALL(norm_inf(row(output,i)-row(output_pad,i)),T(1.e-7));
	}
}


template<class T>
void test_pad(
	std::size_t image_size1, std::size_t image_size2,
	std::size_t filter_size1, std::size_t filter_size2,
	std::size_t num_channels,
	std::size_t num_filters
){
	std::size_t image_size1pad = image_size1 + filter_size1-1;
	std::size_t image_size2pad = image_size2 + filter_size2-1;
	vector<T> image_lin(num_channels * image_size1 *  image_size2,0);
	vector<T> image_lin_pad(num_channels * image_size1pad *  image_size2pad,0);
	for(std::size_t c = 0; c != num_channels; ++c){
		std::size_t lin_elem = c*image_size1*image_size2;
		std::size_t lin_elem_pad = c*image_size1pad*image_size2pad + ((filter_size1-1)/2)*image_size2pad+(filter_size2-1)/2;
		for(std::size_t i = 0; i !=  image_size1; ++i){
			for(std::size_t j = 0; j != image_size2; ++j, ++lin_elem_pad, ++lin_elem){
				double v = 1.0/(num_channels * image_size1)*i + 0.1 - (0.1/image_size2)*j;
				image_lin(lin_elem) = v;
				image_lin_pad(lin_elem_pad) = v;
			}
			lin_elem_pad += filter_size2-1;
		}
	}
	
	vector<T> filter_lin(num_channels * num_filters *  filter_size1 * filter_size2);
	std::size_t lin_elem = 0;
	for(std::size_t i = 0; i != num_channels * num_filters * filter_size1; ++i){
		for(std::size_t j = 0; j != filter_size2; ++j, ++lin_elem){
			filter_lin(lin_elem) = 1.0/(num_channels * filter_size1)*i + 0.1 - (0.1/filter_size2)*j;
		}
	}
	
	vector<T> out_lin(image_size1 * image_size2 * num_filters, 0.0);
	vector<T> out_lin_pad(image_size1 * image_size2 * num_filters, 0.0);
	kernels::conv2d(
		image_lin,filter_lin,out_lin,num_channels, num_filters, 
		image_size1, image_size2, filter_size1, filter_size2,filter_size1 - 1, filter_size2 - 1
	);
	kernels::conv2d(
		image_lin_pad,filter_lin,out_lin_pad,num_channels, num_filters, 
		image_size1pad, image_size2pad, filter_size1, filter_size2,0,0
	);
	
	for(std::size_t k = 0; k != num_filters; ++k){
		for(std::size_t i = 0; i != image_size1; ++i){
			for(std::size_t j = 0; j != image_size2; ++j){
				double val = out_lin(k * image_size1 * image_size2 + i * image_size2 + j);
				double valTest = out_lin_pad(k * image_size1 * image_size2 + i * image_size2 + j);
				BOOST_CHECK_CLOSE(val,valTest,1.e-3);
			}
		}
	}
}

BOOST_AUTO_TEST_SUITE(Remora_Conv2d)

typedef boost::mpl::list<float,double> data_types;
BOOST_AUTO_TEST_CASE_TEMPLATE(conv2d_im_to_mat_pad_test, value_type,data_types) {
	test_im_to_mat_pad<value_type>(7,10,4,4,1);
	test_im_to_mat_pad<value_type>(7,10,4,4,5);
	test_im_to_mat_pad<value_type>(32,16,4,8,5);
	test_im_to_mat_pad<value_type>(57,33,7,3,22);
}

BOOST_AUTO_TEST_CASE_TEMPLATE(conv2d_test, value_type,data_types) {
	test<value_type>(32,16,4,8,5,7);
	test<value_type>(16,12,4,8,1,1);
	test<value_type>(57,33,7,3,22,15);
	test<value_type>(192,333,7,3,22,15);
}

BOOST_AUTO_TEST_CASE_TEMPLATE(conv2d_test_pad, value_type,data_types) {
	test_pad<value_type>(32,16,4,8,5,7);
	test_pad<value_type>(16,12,4,8,1,1);
	test_pad<value_type>(57,33,7,3,22,15);
	test_pad<value_type>(192,333,7,3,22,15);
}

BOOST_AUTO_TEST_SUITE_END()
