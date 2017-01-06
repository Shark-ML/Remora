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
	set( REMORA_USE_SIMD 1)
endif()

option( ENABLE_CBLAS "Use Installed Linear Algebra Library" ON )
if( ENABLE_CBLAS )
	set(CBLAS_VENDOR FALSE)
	if( APPLE )
		set(CBLAS_VENDOR "Accelerate")
		set(CBLAS_INCLUDES "")
		set(CBLAS_LIBRARIES "-framework Accelerate" )
	else()
		#todo: do a propper vendor check
		find_library(OPENBLAS_LIBRARY openblas
			HINTS ${CBLAS_ROOT}/lib /opt/local/lib /usr/lib/openblas-base
		)
		find_library(CBLAS_LIBRARY cblas
			HINTS ${ATLAS_ROOT}/lib ${CBLAS_ROOT}/lib /opt/local/lib /usr/lib64/atlas/ /usr/lib/libblas
		)
		find_library(CLAPACK_LIBRARY lapack
			HINTS ${ATLAS_ROOT}/lib ${CBLAS_ROOT}/lib /opt/local/lib /usr/lib64/atlas/ /usr/lib/lapack
		)
		find_library(ATLAS_LIBRARY atlas
			HINTS ${ATLAS_ROOT}/lib ${CBLAS_ROOT}/lib /opt/local/lib /usr/lib64/atlas/ /usr/lib/atlas-base
		)
		mark_as_advanced(
			OPENBLAS_LIBRARY
			CBLAS_LIBRARY
			CLAPACK_LIBRARY
			ATLAS_LIBRARY
		)
		#find the cblas.h include path
		if(CBLAS_LIBRARY )
			get_filename_component(CBLAS_LIB_PATH ${CBLAS_LIBRARY} PATH )
		elseif( OPENBLAS_LIBRARY)	
			get_filename_component(CBLAS_LIB_PATH ${OPENBLAS_LIBRARY} PATH )
		endif()
		if(CBLAS_LIB_PATH)
			find_file(CBLAS_INCLUDES cblas.h 
				PATHS ${CBLAS_LIB_PATH} ${CBLAS_LIB_PATH}/../include
			)
			get_filename_component(CBLAS_INCLUDES ${CBLAS_INCLUDES} PATH )
		endif()
		if(ATLAS_LIBRARY)
			get_filename_component(ATLAS_LIBRARY_PATH ${ATLAS_LIBRARY} PATH )
			find_file(CLAPACK_INCLUDES clapack.h 
				PATHS
				${ATLAS_LIBRARY_PATH} 
				${ATLAS_LIBRARY_PATH}/../include
				${ATLAS_LIBRARY_PATH}/../include/atlas
				${ATLAS_LIBRARY_PATH}/../../include/atlas
			)
			get_filename_component(CLAPACK_INCLUDES ${CLAPACK_INCLUDES} PATH )
			set(CBLAS_INCLUDES ${CBLAS_INCLUDES} ${CLAPACK_INCLUDES})
		endif()

		if( OPENBLAS_LIBRARY AND CBLAS_INCLUDES)
			set(CBLAS_VENDOR "OpenBLAS")
			set(CBLAS_LIBRARIES ${OPENBLAS_LIBRARY})
		elseif( CBLAS_LIBRARY AND CLAPACK_LIBRARY AND ATLAS_LIBRARY AND CBLAS_INCLUDES)
			set(CBLAS_VENDOR "ATLAS")
			set(CBLAS_LIBRARIES ${CLAPACK_LIBRARY} ${CBLAS_LIBRARY} ${ATLAS_LIBRARY})
		elseif( CBLAS_LIBRARY AND CBLAS_INCLUDES)
			#check that we can compile a basic program with the libraries we have found
			#vendor versions might come with additional libraries which would be bad.
			try_compile(CBLAS_COMPILE 
				"${PROJECT_BINARY_DIR}/cBlasCheck"
				"${CMAKE_SOURCE_DIR}/cBlasCheck.cpp"
				CMAKE_FLAGS "-DINCLUDE_DIRECTORIES=${CBLAS_INCLUDES}"
				LINK_LIBRARIES ${CBLAS_LIBRARY}
			)
			if(CBLAS_COMPILE)
				set(CBLAS_VENDOR "GENERIC")
				set(CBLAS_LIBRARIES ${CBLAS_LIBRARY})
			else()
				message(WARNING "Unknown CBLAS. Can not use it")
			endif()
		endif()
	endif()
	
	if(CBLAS_VENDOR)
		message(STATUS "CBLAS FOUND: " ${CBLAS_VENDOR} " with include directory " ${CBLAS_INCLUDES} )
		set(REMORA_USE_CBLAS 1)
		list(APPEND EXTRA_INCLUDE_DIRECTORIES ${CBLAS_INCLUDES} )
		list(APPEND LINK_LIBRARIES ${CBLAS_LIBRARIES})
		include_directories ( ${CBLAS_INCLUDES} )
	else()
		message(STATUS "No usable CBLAS Library found. No fast linear Algebra used.")
	endif()
endif()


#####################################################################
#                       General Path settings
#####################################################################
include_directories( ${remora_SOURCE_DIR}/include )
add_subdirectory( include )

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