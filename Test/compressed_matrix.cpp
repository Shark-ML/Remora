#define BOOST_TEST_MODULE Remora_MatrixSparse
#include <boost/test/unit_test.hpp>
#include <boost/test/floating_point_comparison.hpp>

#include <remora/sparse.hpp>
#include <random>

using namespace remora;

struct Element{
	std::size_t major_index;
	std::size_t minor_index;
	int value;
	Element(){}
	Element(std::size_t major_index, std::size_t minor_index, int v)
		:major_index(major_index), minor_index(minor_index),value(v){}
		
	bool operator<(Element const& other)const{
		if(major_index == other.major_index)
			return minor_index < other.minor_index;
		else
			return major_index < other.major_index;
	}
};

//checks the internal memory structure of the matrix and ensures that it stores the same elements as 
//are given in the vector.
template<class Matrix>
void checkCompressedMatrixStructure(std::vector<Element> const& elements, Matrix const& matrix){
	//check storage invariants
	BOOST_REQUIRE_EQUAL(matrix.raw_storage().major_indices_begin[0], 0);
	
	std::size_t elem = 0;
	for(std::size_t i = 0; i != major_size(matrix); ++i){
		//find row end in the array
		std::size_t elem_end = elem;
		while(elem_end != elements.size() && elements[elem_end].major_index == i) ++elem_end;
		std::size_t line_size = elem_end-elem;
		
		//check storage invariants
		BOOST_REQUIRE_EQUAL(matrix.raw_storage().major_indices_end[i] - matrix.raw_storage().major_indices_begin[i], line_size);//end-start = line_size
		BOOST_REQUIRE(matrix.raw_storage().major_indices_end[i] <= matrix.raw_storage().major_indices_begin[i+1] );//end of current is smaller or equal start of next
		
		//check query functions
		BOOST_REQUIRE_EQUAL(matrix.raw_storage().major_indices_begin[i+1] - matrix.raw_storage().major_indices_begin[i], matrix.major_capacity(i));//capacity is correctly implemented
		BOOST_REQUIRE_EQUAL(matrix.raw_storage().major_indices_end[i] - matrix.raw_storage().major_indices_begin[i], matrix.major_nnz(i));//major_nnz is correctly implemented
		
		
		//check iterator invariants
		BOOST_REQUIRE_EQUAL(matrix.major_end(i) - matrix.major_begin(i), line_size);
		BOOST_REQUIRE_EQUAL(matrix.major_begin(i).major_index(), i);
		BOOST_REQUIRE_EQUAL(matrix.major_end(i).major_index(), i);
		
		
		//check line elements
		std::size_t line_index = matrix.raw_storage().major_indices_begin[i];
		for(auto pos = matrix.major_begin(i); pos != matrix.major_end(i); 
			++pos,++elem,++line_index
		){
			//check array
			BOOST_CHECK_EQUAL(matrix.raw_storage().indices[line_index],elements[elem].minor_index);
			BOOST_CHECK_EQUAL(matrix.raw_storage().values[line_index],elements[elem].value);
			//check iterator
			BOOST_CHECK_EQUAL(pos.index(),elements[elem].minor_index);
			BOOST_CHECK_EQUAL(*pos,elements[elem].value);
		}
	}
	
	
}

void check_line_sizes(std::vector<std::size_t> const& line_sizes, compressed_matrix<int> const& matrix){
	std::size_t line_start = 0;
	for(std::size_t i = 0; i != line_sizes.size(); ++i){
		BOOST_CHECK_EQUAL(matrix.raw_storage().major_indices_begin[i], line_start);
		BOOST_CHECK_GE(matrix.major_capacity(i), line_sizes[i]);
		BOOST_CHECK_GE(matrix.raw_storage().major_indices_end[i], line_start);
		line_start += line_sizes[i];
	}
	BOOST_CHECK_GE(matrix.nnz_capacity(), line_start);
}



BOOST_AUTO_TEST_SUITE (Remora_compressed_matrix)
//tests whether reserve calls are correct
BOOST_AUTO_TEST_CASE( Remora_sparse_matrix_reserve_row){
	std::size_t rows = 11;//Should be prime :)
	std::size_t columns = 30;
	std::size_t base = 8;
	compressed_matrix<int> matrix(rows,columns);
	std::vector<std::size_t> line_sizes(rows,0);
	
	std::size_t i = 1;
	for(std::size_t j = 0; j != columns; ++j){
		i = (i*base)% rows;
		line_sizes[i]=j;
		matrix.major_reserve(i,j);
		check_line_sizes(line_sizes,matrix);
	}
}

//this test tests push_back behavior of set_element and operator()
BOOST_AUTO_TEST_CASE( Remora_sparse_matrix_insert_element_end){
	std::size_t rows = 10;
	std::size_t columns = 20;
	
	compressed_matrix<int> matrix_set(rows,columns);
	std::vector<Element> elements;
	for(std::size_t i = 0; i != rows; ++i){
		compressed_matrix<int>::major_iterator major_iter = matrix_set.major_begin(i);
		for(std::size_t j = 0; j < columns; j+=3){
			BOOST_REQUIRE_EQUAL(major_iter.major_index(),i);
			int val = (int)(i + j);
			major_iter = matrix_set.set_element(major_iter,j,val);
			elements.push_back(Element(i,j,val));
			BOOST_REQUIRE( matrix_set.nnz_reserved() >= elements.size());
			BOOST_REQUIRE( matrix_set.nnz_capacity() >= elements.size());
			checkCompressedMatrixStructure(elements,matrix_set);
		}
	}
}

//we still insert row by row, but now with different gaps between indices.
BOOST_AUTO_TEST_CASE( Remora_sparse_matrix_insert_random ){
	std::size_t rows = 10;
	std::size_t columns = 23;
	
	//generate random elements to insert, this will also cause duplicates
	std::mt19937 gen(42);
	std::vector<Element> insertions;
	for(std::size_t i = 0; i != 1000; ++i){
		std::uniform_int_distribution<> row_dist(0, rows-1);
		std::uniform_int_distribution<> col_dist(0, 5);//5 different elements per row
		std::size_t major_index = row_dist(gen);
		std::size_t minor_index = (major_index + col_dist(gen) * 7) % columns;
		int value = i;
		insertions.push_back({major_index,minor_index, value});
	}
	
	compressed_matrix<int> matrix_set(rows,columns);
	compressed_matrix<int> matrix_ref_set(rows,columns);
	detail::compressed_matrix_proxy<compressed_matrix<int>, row_major> matrix_ref(matrix_ref_set);
	
	std::vector<Element> elements;
	for(auto elem: insertions){
		//update elements in the matrix
		std::size_t pos = 0;
		for(pos = 0; pos != elements.size(); ++pos){
			if(elements[pos].major_index == elem.major_index 
			&& elements[pos].minor_index == elem.minor_index) break;				
		}
		if(pos != elements.size()){
			elements[pos] = elem;
		}else{
			elements.push_back(elem);
			std::sort(elements.begin(),elements.end());
		}
		
		//insert into matrix. first create iterator to position
		{
			auto major_iter = matrix_set.major_begin(elem.major_index);
			while(major_iter != matrix_set.major_end(elem.major_index) && major_iter.index()< elem.minor_index)
				++major_iter;
			//now insert and check
			matrix_set.set_element(major_iter,elem.minor_index,elem.value);
			BOOST_REQUIRE( matrix_set.nnz_reserved() >= elements.size());
			BOOST_REQUIRE( matrix_set.nnz_capacity() >= elements.size());
			checkCompressedMatrixStructure(elements,matrix_set);
		}
		{
			auto major_iter = matrix_ref.major_begin(elem.major_index);
			while(major_iter != matrix_ref.major_end(elem.major_index) && major_iter.index()< elem.minor_index)
				++major_iter;
			//now insert and check
			matrix_ref.set_element(major_iter,elem.minor_index,elem.value);
			checkCompressedMatrixStructure(elements,matrix_ref);
			checkCompressedMatrixStructure(elements,matrix_ref_set);
		}
	}
}



BOOST_AUTO_TEST_SUITE_END()
