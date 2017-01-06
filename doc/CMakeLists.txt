CMAKE_MINIMUM_REQUIRED( VERSION 2.8 )

find_package( Doxygen REQUIRED )
find_package( PythonInterp REQUIRED )

ADD_SUBDIRECTORY(tutToRst)

CONFIGURE_FILE (
  "${CMAKE_CURRENT_SOURCE_DIR}/sphinx_pages/conf.py.in"
  "${CMAKE_CURRENT_SOURCE_DIR}/sphinx_pages/conf.py"
)

set( SPHINX_EXECUTABLE sphinx-build )
set( SPHINX_PARAMETERS -b html )

add_custom_target(doc_creation)

#find all .tut files
file(GLOB_RECURSE TutFiles sphinx_pages *.tut)
message(STATUS ${TutFiles})
foreach(tut ${TutFiles})
	GET_FILENAME_COMPONENT(tutPath ${tut} PATH)
	GET_FILENAME_COMPONENT(tutName ${tut} NAME_WE)
	add_custom_command(TARGET doc_creation POST_BUILD COMMAND
		tut2rst ${tutPath}/${tutName} ${PROJECT_SOURCE_DIR}/examples
	)
endforeach()
add_dependencies(doc_creation tut2rst)

add_custom_target( doc ALL
	COMMAND ${SPHINX_EXECUTABLE} ${SPHINX_PARAMETERS} ${CMAKE_CURRENT_SOURCE_DIR}/sphinx_pages ${CMAKE_CURRENT_BINARY_DIR}/sphinx_pages/build/html
)


add_dependencies(doc doc_creation)