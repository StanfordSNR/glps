add_executable (example1 "example1.cc")
target_link_libraries (example1 util signal)

target_link_libraries ("example1" ${Boost_LDFLAGS})
target_link_libraries ("example1" ${Boost_LDFLAGS_OTHER})
target_link_libraries ("example1" ${FFTW_LDFLAGS})
target_link_libraries ("example1" ${FFTW_LDFLAGS_OTHER})
target_link_libraries ("example1" "-lfftw3_threads")

add_executable (corrtest "corrtest.cc")
target_link_libraries (corrtest util signal)

target_link_libraries ("corrtest" ${Boost_LDFLAGS})
target_link_libraries ("corrtest" ${Boost_LDFLAGS_OTHER})
target_link_libraries ("corrtest" ${FFTW_LDFLAGS})
target_link_libraries ("corrtest" ${FFTW_LDFLAGS_OTHER})
target_link_libraries ("corrtest" "-lfftw3_threads")

add_executable (make-fft "make-fft.cc")
target_link_libraries (make-fft util signal)

target_link_libraries ("make-fft" ${Boost_LDFLAGS})
target_link_libraries ("make-fft" ${Boost_LDFLAGS_OTHER})
target_link_libraries ("make-fft" ${FFTW_LDFLAGS})
target_link_libraries ("make-fft" ${FFTW_LDFLAGS_OTHER})
target_link_libraries ("make-fft" "-lfftw3_threads")
