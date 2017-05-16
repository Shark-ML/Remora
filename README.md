Remora BLAS Library
================================

Remora is a general purpose linear algebra library written in C++. 
It features:

  * Dense and Sparse Matrix and Vector operations
  * A basic set of optimized routines for matrix products, solving linear systems of equations etc
  * Bindings to highly optimized routines of BLAS packages
  * A powerful expression template syntax which features algebraic optimizations of operations
  * (experimental and very early) GPU support via OpenCL
  
Remora is used by the shark machine learning library.
  
Installation
---------------------------------------
  
Remora is header-only. just download and copy the contents of the include/ folder into its 
target location. Remora depends on the boost c++ libraries.
When using Remora, the following defines can be supplied at compile time
  
* REMORA_USE_SIMD if defined, Remora uses the compilers' auto vectorizing capabilities 
  to speed up its computational routines. Requires G++ or clang
* REMORA_USE_CBLAS if defined, Remora binds to a cblas library. 
  On MacOsX, this flag is interpreted as using the accelerate framework.
  Make sure to add the appropriate compile and linker flags for the library.
* REMORA_USE_GPU if defined, Remora is enabling gpu support via the boost/compute
  library. Highly experimental.
* REMORA_USE_CLBLAST if defined, Remora is using CLBlast as GPU/opencl backend. 
  This should be more stable and give better performance on most devices.
  
Contributing
----------------------------------------------------------
Contributing is easy via [Pull Requests][1]. We are open
for any types of contribution, but we favour them in the following order

  1. Bug fixes, test cases, documentation, benchmarks, examples...
  2. Optimizing existing computational routines found in include/Remora/kernels/
  3. Adding new computation routines in include/Remora/kernels/
  4. extending the expression template system
  
The reason for this order is that extending the expression template system often leads
to a large increase of possible operation combinations to cover, some of which might not
be implemented in the kernel routines and thus leading to compile errors or inefficient code.
We would like to prevent this by favouring the quality and number of underlying computational
routines, which are very often easier to implement (and it is okay if those routines are very 
specialized).


[1]: https://github.com/Shark-ML/Remora/pulls
