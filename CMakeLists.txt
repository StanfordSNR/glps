cmake_minimum_required ( VERSION 2.8.5 )
cmake_policy ( SET CMP0048 NEW )
project ( glps )

include ( etc/build_type.cmake )
include ( etc/cflags.cmake )
include ( etc/clang_format.cmake )
include ( etc/tests.cmake )

find_package ( PkgConfig )

find_package ( boost_headers REQUIRED Boost )
include_directories ( SYSTEM ${Boost_INCLUDE_DIRS} )
add_compile_options ( ${Boost_CFLAGS} )

pkg_check_modules ( FFTW REQUIRED fftw3 )
include_directories ( SYSTEM ${FFTW_INCLUDE_DIRS} )
add_compile_options ( ${FFTW_CFLAGS} )

include_directories ("${PROJECT_SOURCE_DIR}/src/util")
include_directories ("${PROJECT_SOURCE_DIR}/src/signal")

add_subdirectory ("${PROJECT_SOURCE_DIR}/src/util")
add_subdirectory ("${PROJECT_SOURCE_DIR}/src/signal")
add_subdirectory ("${PROJECT_SOURCE_DIR}/src/frontend")
add_subdirectory ("${PROJECT_SOURCE_DIR}/src/tests")
