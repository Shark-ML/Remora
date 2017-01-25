#include <random>
#include <iostream>

//###begin<include>
#include <remora/remora.hpp>
using namespace remora;
//###end<include>


int main(){
	//Step 0: Theory
	// The goal of linear regression is to find a linear function f(x) = w^Tx + b
	// that fits best a given set of point-label pairs (x1,y1),(x2,y2),...,(xN,yN).
	// This is measured by the squared error:
	// E(w) = 1/(2N) sum_i (f(x_i)-y_i)^2
	// It turns out that the optimal solution can be written in simple matrix algebra,
	// when X is the data matrix where points are stored row-wise and y is the vector
	// of labels:
	// w=((X|1)^T (X|1))^{-1} (X|1)^Ty
	
	//Step 1: Generate some random data
	//###begin<data_declaration>
	std::size_t num_data_points = 100;
	std::size_t num_dims = 50;
	matrix<double> X(num_data_points,num_dims);
	vector<double> y(num_data_points);
	//###end<data_declaration>
	//###begin<generate_X>
	std::random_device rd;
	std::mt19937 gen(rd());
	std::normal_distribution<> normal(0,2);
	for(std::size_t i = 0; i != num_data_points; ++i){
		for(std::size_t j = 0; j != num_dims; ++j){
			X(i,j) = normal(gen); //set element (i,j) of X to a rnadomly generated number
		}
	}
	//###end<generate_X>
	//###begin<generate_Y>
	std::normal_distribution<> normal_noise(0,0.1);
	for(std::size_t i = 0; i != num_data_points; ++i){
		//label is chosen to be just the sum of entries plus some noise
		y(i) = sum(row(X,i)) + normal_noise(gen) + 1;
	}
	//###end<generate_Y>
	// Step 2: compute the linear regression
	// formula is w=((X|1)^T (X|1))^{-1} (X|1)^Ty
	// we need to tell the algebra how to solve the system of equations,
	// in this case we tell it that the matrix is symmetric positive definite.
	// but we have to be aware that our matrix is not always full rank,
	// e..g when we have more variables than data or when some variable
	// is constant 0.
	//###begin<solve_w>
	vector<double> w = inv(trans(X|1) % (X|1), symm_semi_pos_def()) % trans(X|1) % y; 
	//###end<solve_w>
	// Step 3: evaluate solution
	// we compute: E(w) = 1/(2N) sum_i (f(x_i)-y_i)^2
	//###begin<compute_error>
	double error = 0.5 * sum(sqr((X|1) % w - y))/num_data_points;
	//###end<compute_error>
	// Step 4: For ensuring correctness, we will check that
	// the derivative of E(w) at the solution is small (on the order of 1.e-14)
	//###begin<verify_derivative>
	vector<double> derE= trans(X|1) % ((X|1) % w - y) / num_data_points;
	double error_derivative = norm_inf(derE);
	//###end<verify_derivative>
	std::cout<<"final error of fit: "<< error<<std::endl;
	std::cout<<"norm_1 of derivative: "<< error_derivative <<std::endl;
}