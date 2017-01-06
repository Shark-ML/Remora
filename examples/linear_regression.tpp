#include <remora/remora.hpp>
#include <random>
#include <iostream>
using namespace remora;

int main(){
	//Step 0: Theory
	// The goal of linear regression is to find a linear function f(x) = w^Tx
	// that fits best a given set of point-label pairs (x1,y1),(x2,y2),...,(xN,yN).
	// This is measured by the squared error:
	// E(w) = 1/(2N) sum_i (f(x_i)-y_i)^2
	// It turns out that the optimal solution can be written in simple matrix algebra,
	// when X is the data matrix where points are stored row-wise and y is the vector
	// of labels:
	// w=(X^TX)^{-1}X^Ty
	
	//Step 1: Generate some random data
	std::size_t num_data_points = 100;
	std::size_t num_dims = 50;
	matrix<double> X(num_data_points,num_dims);
	vector<double> y(num_data_points,num_dims);
	
	std::random_device rd;
	std::mt19937 gen(rd());
	std::normal_distribution<> normal(0,2);
	for(std::size_t i = 0; i != num_data_points; ++i){
		for(std::size_t j = 0; j != num_dims; ++j){
			X(i,j) = normal(gen); //set element (i,j) of X to a rnadomly generated number
		}
	}
	std::normal_distribution<> normal_noise(0,0.1);
	for(std::size_t i = 0; i != num_data_points; ++i){
		//label is chosen to be just the sum of entries plus some noise
		y(i) = sum(row(X,i)) + normal_noise(gen);
	}
	// Step 2: compute the linear regression
	// formula is w=(X^TX)^{-1}X^Ty
	// we need to tell the algebra how to solve the system of equations,
	// in this case we tell it that the matrix is symmetric positive definite.
	// but we have to be aware that our matrix is not always full rank,
	// e..g when we have more variables than data or when some variable
	// is constant 0.
	vector<double> w = inv(trans(X) % X, symm_semi_pos_def()) % trans(X) % y; 
	
	// Step 3: evaluate solution
	// we compute: E(w) = sum_i (f(x_i)-y_i)^2
	double error = sum(sqr(X % w - y))/num_data_points;
	
	// Step 4: For ensuring correctness, we will check that
	// the derivative of E(w) at the solution is small (on the order of 1.e-14)
	vector<double> derE= trans(X) % (X % w - y) / num_data_points;
	double error_derivative = norm_inf(derE);
	
	std::cout<<"final error of fit: "<< error<<std::endl;
	std::cout<<"norm_1 of derivative: "<< error_derivative <<std::endl;
}