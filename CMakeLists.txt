#####################################################################
#       Shark Machine Learning Library
#       Top-Level CMake driver file
#       Optionally included sub-probjects:
#         * Test/CMakeLists.txt
#         * examples/CMakeLists.txt
#         * doc/CMakeLists.txt
#####################################################################
project( remora )
cmake_minimum_required( VERSION 3.1 FATAL_ERROR)

set_property(GLOBAL PROPERTY USE_FOLDERS ON)

cmake_policy(SET CMP0003 NEW)
if(POLICY CMP0042)
	cmake_policy(SET CMP0042 NEW)
endif()
if(POLICY CMP0053)
	cmake_policy(SET CMP0053 NEW)
endif()
#=========================================================
# Output directories.
if(NOT CMAKE_RUNTIME_OUTPUT_DIRECTORY)
	set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "${remora_BINARY_DIR}/bin")
endif()
if(NOT CMAKE_LIBRARY_OUTPUT_DIRECTORY)
	if(UNIX)
		set(CMAKE_LIBRARY_OUTPUT_DIRECTORY "${remora_BINARY_DIR}/lib")
	else()
		set(CMAKE_LIBRARY_OUTPUT_DIRECTORY "${remora_BINARY_DIR}/bin")
	endif()
endif()
if(NOT CMAKE_ARCHIVE_OUTPUT_DIRECTORY)
	set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY "${remora_BINARY_DIR}/lib")
endif()
mark_as_advanced(
	CMAKE_RUNTIME_OUTPUT_DIRECTORY
	CMAKE_LIBRARY_OUTPUT_DIRECTORY
	CMAKE_ARCHIVE_OUTPUT_DIRECTORY
)

#####################################################################
# Version information
#####################################################################
option(BUILD_OFFICIAL_RELEASE "Is this an official Shark release." OFF )
mark_as_advanced( BUILD_OFFICIAL_RELEASE )

set(REMORA_VERSION_MAJOR 0)
set(REMORA_VERSION_MINOR 1)
set(REMORA_VERSION_PATCH 0)
set(REMORA_VERSION ${REMORA_VERSION_MAJOR}.${REMORA_VERSION_MINOR}.${REMORA_VERSION_PATCH})
set(REMORA_SOVERSION 0)

#####################################################################
#    Adjust include, lib, example and doc paths for installation
#####################################################################
if( UNIX )
	include(GNUInstallDirs)
	set( REMORA_INSTALL_INCLUDE_DIR ${CMAKE_INSTALL_INCLUDEDIR} CACHE STRING "" )
	set( REMORA_INSTALL_DOC_DIR ${CMAKE_INSTALL_DATADIR}/remora/doc CACHE STRING "" )
else()
	set( REMORA_INSTALL_INCLUDE_DIR include/remora/ )
	set( REMORA_INSTALL_EXAMPLE_DIR examples/ )
	set( REMORA_INSTALL_DOC_DIR doc/ )
endif()


#####################################################################
#           Explicit macro setup for debug configuration
#####################################################################
# enable or disable debugging, default is Release
if(NOT CMAKE_BUILD_TYPE)
	set(CMAKE_BUILD_TYPE "Release")
endif()

if (UNIX)
    add_compile_options("$<$<CONFIG:DEBUG>:-Wall>")
endif()

list(APPEND COMPILE_DEFINITIONS_RELEASE NDEBUG)
message(STATUS "Will build: " ${CMAKE_BUILD_TYPE})

#####################################################################
#           Boost configuration
#####################################################################
set(Boost_USE_STATIC_LIBS OFF CACHE BOOL "use static libraries from Boost")
add_definitions(-DBOOST_TEST_DYN_LINK)

find_package(
	Boost 1.54.0 REQUIRED
)

if(NOT Boost_FOUND)
	message(FATAL_ERROR "Please make sure Boost 1.54.0 is installed on your system")
endif()
include_directories(SYSTEM ${Boost_INCLUDE_DIR} )
message( STATUS "Using boost" ${Boost_VERSION} " from " ${Boost_LIBRARY_DIR} )


#####################################################################
#           BLAS configuration
#####################################################################

option( ENABLE_SIMD "Use SIMD for computational routines" OFF )
if( ENABLE_SIMD )
	add_definitions(-DREMORA_USE_SIMD)
endif()
option( ENABLE_ACCELERATE "Use Installed Accelerate framework" OFF )
if( ENABLE_ACCELERATE )
	set(CBLAS_LIBRARIES "-framework Accelerate" )
	add_definitions(-DREMORA_USE_CBLAS)
endif()

option( ENABLE_OPENBLAS "Use Installed OpenBLAS" OFF )
if( ENABLE_OPENBLAS )
	find_package(
		OpenBLAS
	)
	include_directories(${OpenBLAS_INCLUDE_DIRS} )
	set(CBLAS_LIBRARIES ${OpenBLAS_LIBRARIES} )
	add_definitions(-DREMORA_USE_CBLAS)
endif()

option( ENABLE_GENERIC_CBLAS "Use Installed cblas" OFF )
if( ENABLE_GENERIC_CBLAS )
	find_library(CBLAS_LIBRARY cblas
		HINTS ${CBLAS_ROOT}/lib /opt/local/lib /usr/lib64/atlas/
	)
	get_filename_component(CBLAS_LIB_PATH ${CBLAS_LIBRARY} PATH )
	find_file(CBLAS_INCLUDES cblas.h 
		PATHS ${CBLAS_CBLAS_LIB_PATH}/../include
	)
	get_filename_component(CBLAS_INCLUDES ${CBLAS_INCLUDES} PATH )
	include_directories(${CBLAS_INCLUDES} )
	set(CBLAS_LIBRARIES ${CBLAS_LIBRARY} )
	add_definitions(-DREMORA_USE_CBLAS)
endif()

option( ENABLE_CLBLAST "Use CLBlast as OpenCL backend" OFF )
if(ENABLE_CLBLAST)
	find_package(CLBlast REQUIRED)
	add_definitions(-DREMORA_USE_CLBLAST)	
endif()

#####################################################################
#           OpenCL configuration
#####################################################################

option( ENABLE_OPENCL "Use OpenCL and boost.compute" OFF )
if(ENABLE_OPENCL OR ENABLE_CLBLAST)
	find_package(OpenCL REQUIRED)
	include_directories(${OpenCL_INCLUDE_DIRS})
	list( APPEND CBLAS_LIBRARIES ${OpenCL_LIBRARY})
	add_definitions(-DREMORA_USE_OPENCL )
endif()

#####################################################################
#           HIP configuration
#####################################################################

option( ENABLE_HIP "Use HIP gpu backend" OFF )
if(ENABLE_HIP)
	find_package(HIP REQUIRED)
endif()



#####################################################################
#                       Include Headers
#####################################################################
include_directories( ${remora_SOURCE_DIR}/include )
add_subdirectory( include )

#####################################################################
#                       Include Examples
#####################################################################
option( BUILD_EXAMPLES "Build example programs." ON )
add_subdirectory( examples )

#####################################################################
#                       Include Unit Tests
#####################################################################
option(BUILD_TESTING "Build tests." OFF)
if(BUILD_TESTING)
	include(CTest)
	enable_testing()
	add_subdirectory( Test )
endif()

#####################################################################
#                       Include Documentation
#####################################################################
option(BUILD_DOCUMENTATION "Build documentation." OFF)
if(BUILD_DOCUMENTATION)
	add_subdirectory(doc)
endif()
