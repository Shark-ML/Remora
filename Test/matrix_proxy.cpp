#define BOOST_TEST_MODULE Remora_MatrixProxy
#include <boost/test/unit_test.hpp>
#include <boost/test/floating_point_comparison.hpp>

#include <remora/proxy_expressions.hpp>
#include <remora/dense.hpp>


using namespace remora;


template<class M1>
double get(M1 const& m, std::size_t i, std::size_t j, row_major){
	return m(i,j);
}
template<class M1>
double get(M1 const& m, std::size_t i, std::size_t j, column_major){
	return m(j,i);
}
template<class M1, class M2>
void checkDenseMatrixEqual(M1 const& m1, M2 const& m2){
	BOOST_REQUIRE_EQUAL(m1.size1(),m2.size1());
	BOOST_REQUIRE_EQUAL(m1.size2(),m2.size2());
	//indexed access
	for(std::size_t i = 0; i != m2.size1(); ++i){
		for(std::size_t j = 0; j != m2.size2(); ++j){
			BOOST_CHECK_EQUAL(m1(i,j),m2(i,j));
		}
	}
	//iterator access rows
	for(std::size_t i = 0; i != m2.size1(); ++i){
		typedef typename M1::const_major_iterator Iter;
		BOOST_REQUIRE_EQUAL(m1.major_end(i)-m1.major_begin(i), m1.size2());
		std::size_t k = 0;
		for(Iter it = m1.major_begin(i); it != m1.major_end(i); ++it,++k){
			BOOST_CHECK_EQUAL(k,it.index());
			BOOST_CHECK_EQUAL(*it,get(m2,i,k, typename M1::orientation()));
		}
		//test that the actual iterated length equals the number of elements
		BOOST_CHECK_EQUAL(k, m1.size2());
	}
}


template<class M1, class M2>
void checkDenseMatrixAssignment(M1& m1, M2 const& m2){
	BOOST_REQUIRE_EQUAL(m1.size1(),m2.size1());
	BOOST_REQUIRE_EQUAL(m1.size2(),m2.size2());
	//indexed access
	for(std::size_t i = 0; i != m2.size1(); ++i){
		for(std::size_t j = 0; j != m2.size2(); ++j){
			m1(i,j) = 0;
			BOOST_CHECK_EQUAL(m1(i,j),0);
			m1(i,j) = m2(i,j);
			BOOST_CHECK_EQUAL(m1(i,j),m2(i,j));
			m1(i,j) = 0;
			BOOST_CHECK_EQUAL(m1(i,j),0);
		}
	}
	//iterator access rows
	for(std::size_t i = 0; i != major_size(m1); ++i){
		typedef typename M1::major_iterator Iter;
		BOOST_REQUIRE_EQUAL(m1.major_end(i)-m1.major_begin(i), minor_size(m1));
		std::size_t k = 0;
		for(Iter it = m1.major_begin(i); it != m1.major_end(i); ++it,++k){
			BOOST_CHECK_EQUAL(k,it.index());
			*it=0;
			BOOST_CHECK_EQUAL(*it,0);
			BOOST_CHECK_EQUAL(get(m1,i,k, typename M1::orientation()),0);
			*it = get(m2,i,k, typename M1::orientation());
			BOOST_CHECK_EQUAL(*it,get(m2,i,k, typename M1::orientation()));
			BOOST_CHECK_EQUAL(get(m1,i,k, typename M1::orientation()),get(m2,i,k, typename M1::orientation()));
		}
		//test that the actual iterated length equals the number of elements
		BOOST_CHECK_EQUAL(k, minor_size(m1));
	}
}
template<class V1, class V2>
void checkDenseVectorEqual(V1 const& v1, V2 const& v2){
	BOOST_REQUIRE_EQUAL(v1.size(),v2.size());
	//indexed access
	for(std::size_t i = 0; i != v2.size(); ++i){
		BOOST_CHECK_EQUAL(v1(i),v2(i));
	}
	//iterator access rows
	typedef typename V1::const_iterator Iter;
	BOOST_REQUIRE_EQUAL(v1.end()-v1.begin(), v1.size());
	std::size_t k = 0;
	for(Iter it = v1.begin(); it != v1.end(); ++it,++k){
		BOOST_CHECK_EQUAL(k,it.index());
		BOOST_CHECK_EQUAL(*it,v2(k));
	}
	//test that the actual iterated length equals the number of elements
	BOOST_CHECK_EQUAL(k, v2.size());
}

template<class V1, class V2>
void checkDenseVectorAssignment(V1& v1, V2 const& v2){
	BOOST_REQUIRE_EQUAL(v1.size(),v2.size());
	//indexed access
	for(std::size_t i = 0; i != v2.size(); ++i){
		v1(i) = 0;
		BOOST_CHECK_EQUAL(v1(i),0);
		v1(i) = v2(i);
		BOOST_CHECK_EQUAL(v1(i),v2(i));
		v1(i) = 0;
		BOOST_CHECK_EQUAL(v1(i),0);
	}
	//iterator access rows
	typedef typename V1::iterator Iter;
	BOOST_REQUIRE_EQUAL(v1.end()-v1.begin(), v1.size());
	std::size_t k = 0;
	for(Iter it = v1.begin(); it != v1.end(); ++it,++k){
		BOOST_CHECK_EQUAL(k,it.index());
		*it = 0;
		BOOST_CHECK_EQUAL(v1(k),0);
		*it = v2(k);
		BOOST_CHECK_EQUAL(v1(k),v2(k));
		*it = 0;
		BOOST_CHECK_EQUAL(v1(k),0);
	}
	//test that the actual iterated length equals the number of elements
	BOOST_CHECK_EQUAL(k, v2.size());
}

std::size_t Dimensions1 = 4;
std::size_t Dimensions2 = 5;
struct MatrixProxyFixture
{
	matrix<double,row_major> denseData;
	matrix<double,column_major> denseDataColMajor;
	
	MatrixProxyFixture():denseData(Dimensions1,Dimensions2),denseDataColMajor(Dimensions1,Dimensions2){
		for(std::size_t row=0;row!= Dimensions1;++row){
			for(std::size_t col=0;col!=Dimensions2;++col){
				denseData(row,col) = row*Dimensions2+col+5.0;
				denseDataColMajor(row,col) = row*Dimensions2+col+5.0;
			}
		}
	}
};

BOOST_FIXTURE_TEST_SUITE (Remora_matrix_proxy, MatrixProxyFixture);

BOOST_AUTO_TEST_CASE( Remora_Dense_Subrange ){
	//all possible combinations of ranges on the data matrix
	for(std::size_t rowEnd=0;rowEnd!= Dimensions1;++rowEnd){
		for(std::size_t rowBegin =0;rowBegin <= rowEnd;++rowBegin){//<= for 0 range
			for(std::size_t colEnd=0;colEnd!=Dimensions2;++colEnd){
				for(std::size_t colBegin=0;colBegin != colEnd;++colBegin){
					std::size_t size1= rowEnd-rowBegin;
					std::size_t size2= colEnd-colBegin;
					matrix<double> mTest(size1,size2);
					for(std::size_t i = 0; i != size1; ++i){
						for(std::size_t j = 0; j != size2; ++j){
							mTest(i,j) = denseData(i+rowBegin,j+colBegin);
						}
					}
					checkDenseMatrixEqual(
						subrange(denseData,rowBegin,rowEnd,colBegin,colEnd),
						mTest
					);
					matrix<double> newData(Dimensions1,Dimensions2,0);
					matrix<double,column_major> newDataColMaj(Dimensions1,Dimensions2,0);
					auto rangeTest = subrange(newData,rowBegin,rowEnd,colBegin,colEnd);
					auto rangeTestColMaj = subrange(newDataColMaj,rowBegin,rowEnd,colBegin,colEnd);
					checkDenseMatrixAssignment(rangeTest,mTest);
					checkDenseMatrixAssignment(rangeTestColMaj,mTest);
					
					//check assignment
					{
						rangeTest=mTest;
						matrix<double> newData2(Dimensions1,Dimensions2,0);
						auto rangeTest2 = subrange(newData2,rowBegin,rowEnd,colBegin,colEnd);
						rangeTest2=rangeTest;
						for(std::size_t i = 0; i != size1; ++i){
							for(std::size_t j = 0; j != size2; ++j){
								BOOST_CHECK_EQUAL(newData(i+rowBegin,j+colBegin),mTest(i,j));
								BOOST_CHECK_EQUAL(newData2(i+rowBegin,j+colBegin),mTest(i,j));
							}
						}
					}
					
					//check clear
					for(std::size_t i = 0; i != size1; ++i){
						for(std::size_t j = 0; j != size2; ++j){
							rangeTest(i,j) = denseData(i+rowBegin,j+colBegin);
							rangeTestColMaj(i,j) = denseData(i+rowBegin,j+colBegin);
						}
					}
					rangeTest.clear();
					rangeTestColMaj.clear();
					for(std::size_t i = 0; i != size1; ++i){
						for(std::size_t j = 0; j != size2; ++j){
							BOOST_CHECK_EQUAL(rangeTest(i,j),0);
							BOOST_CHECK_EQUAL(rangeTestColMaj(i,j),0);
						}
					}
				}
			}
		}
	}
}

BOOST_AUTO_TEST_CASE( Remora_Dense_row){
	for(std::size_t r = 0;r != Dimensions1;++r){
		vector<double> vecTest(Dimensions2);
		for(std::size_t i = 0; i != Dimensions2; ++i)
			vecTest(i) = denseData(r,i);
		checkDenseVectorEqual(row(denseData,r),vecTest);
		checkDenseVectorEqual(row(denseDataColMajor,r),vecTest);
		matrix<double> newData(Dimensions1,Dimensions2,0);
		matrix<double,column_major> newDataColMajor(Dimensions1,Dimensions2,0);
		auto rowTest = row(newData,r);
		auto rowTestColMajor = row(newDataColMajor,r);
		checkDenseVectorAssignment(rowTest,vecTest);
		checkDenseVectorAssignment(rowTestColMajor,vecTest);
		
		//check assignment
		{
			rowTest=vecTest;
			matrix<double> newData2(Dimensions1,Dimensions2,0);
			auto rowTest2 = row(newData2,r);
			rowTest2=rowTest;
			for(std::size_t i = 0; i != Dimensions2; ++i){
				BOOST_CHECK_EQUAL(newData(r,i),vecTest(i));
				BOOST_CHECK_EQUAL(newData2(r,i),vecTest(i));
			}
		}
		//check clear
		for(std::size_t i = 0; i != Dimensions2; ++i){
			rowTest(i) = double(i);
			rowTestColMajor(i) = double(i);
		}
		rowTest.clear();
		rowTestColMajor.clear();
		for(std::size_t i = 0; i != Dimensions2; ++i){
			BOOST_CHECK_EQUAL(rowTest(i),0.0);
			BOOST_CHECK_EQUAL(rowTestColMajor(i),0.0);
		}
		
	}
}
BOOST_AUTO_TEST_CASE( Remora_Dense_column){
	for(std::size_t c = 0;c != Dimensions2;++c){
		vector<double> vecTest(Dimensions1);
		for(std::size_t i = 0; i != Dimensions1; ++i)
			vecTest(i) = denseData(i,c);
		checkDenseVectorEqual(column(denseData,c),vecTest);
		matrix<double> newData(Dimensions1,Dimensions2,0);
		matrix<double,column_major> newDataColMajor(Dimensions1,Dimensions2,0);
		auto columnTest = column(newData,c);
		auto columnTestColMajor = column(newDataColMajor,c);
		checkDenseVectorAssignment(columnTest,vecTest);
		checkDenseVectorAssignment(columnTestColMajor,vecTest);
		
		{
			columnTest=vecTest;
			matrix<double> newData2(Dimensions1,Dimensions2,0);
			auto columnTest2 = column(newData2,c);
			columnTest2=columnTest;
			for(std::size_t i = 0; i != Dimensions1; ++i){
				BOOST_CHECK_EQUAL(newData(i,c),vecTest(i));
				BOOST_CHECK_EQUAL(newData2(i,c),vecTest(i));
			}
		}
		//check clear
		for(std::size_t i = 0; i != Dimensions1; ++i){
			columnTest(i) = double(i);
			columnTestColMajor(i) = double(i);
		}
		columnTest.clear();
		columnTestColMajor.clear();
		for(std::size_t i = 0; i != Dimensions1; ++i){
			BOOST_CHECK_EQUAL(columnTest(i),0.0);
			BOOST_CHECK_EQUAL(columnTestColMajor(i),0.0);
		}
	}
}

BOOST_AUTO_TEST_CASE( Remora_To_Vector){	
	
	{
		vector<double> vecTest(Dimensions1 * Dimensions2);
		for(std::size_t i = 0; i != Dimensions1; ++i)
			for(std::size_t j = 0;j != Dimensions2;++j)
				vecTest(i * Dimensions2 + j) = denseData(i,j);
		checkDenseVectorEqual(to_vector(denseData),vecTest);
		
		matrix<double,row_major> m(Dimensions1,Dimensions2,0);
		auto test = to_vector(m);
		checkDenseVectorAssignment(test,vecTest);	
	}

	{
		vector<double> vecTest(Dimensions1 * Dimensions2);
		for(std::size_t i = 0; i != Dimensions1; ++i)
			for(std::size_t j = 0;j != Dimensions2;++j)
				vecTest(j *  Dimensions1 + i) = denseData(i,j);
		checkDenseVectorEqual(to_vector(denseDataColMajor),vecTest);
		matrix<double,column_major> m(Dimensions1,Dimensions2,0);
		auto test = to_vector(m);
		checkDenseVectorAssignment(test,vecTest);	
	}
	
}

BOOST_AUTO_TEST_CASE( Remora_To_Matrix){	
	vector<double> vecData(Dimensions1 * Dimensions2);
	for(std::size_t i = 0; i != Dimensions1; ++i)
		for(std::size_t j = 0;j != Dimensions2;++j)
			vecData(i * Dimensions2 + j) = denseData(i,j);
	checkDenseMatrixEqual(to_matrix(vecData,Dimensions1,Dimensions2),denseData);
	vector<double> newData(Dimensions1 * Dimensions2,1.0);
	auto test = to_matrix(newData,Dimensions1,Dimensions2);
	checkDenseMatrixAssignment(test,denseData);
	test = denseData;
	checkDenseVectorEqual(newData, vecData);
	
}


BOOST_AUTO_TEST_SUITE_END();