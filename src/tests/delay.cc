#include "fft.hh"
#include "mmap.hh"

#include <cmath>
#include <cstdlib>
#include <iostream>

using namespace std;

void test_config( double SAMPLE_RATE, double FREQUENCY, double DELAY, size_t LEN )
{
  TimeDomainSignal input { LEN, SAMPLE_RATE };
  TimeDomainSignal target_reference { LEN, SAMPLE_RATE };

  /* initialize input */
  for ( unsigned int i = 0; i < input.size(); i++ ) {
    const double t = input.index_to_time( i );

    input.at( i ) = sin( 2 * M_PI * FREQUENCY * t );
    target_reference.at( i ) = sin( 2 * M_PI * FREQUENCY * ( t - DELAY ) );
  }

  /* fractional delay */
  auto simulated_with_delay = delay( input, DELAY );

  for ( unsigned int i = 0; i < target_reference.size(); i++ ) {
    if ( abs( simulated_with_delay.at( i ).imag() ) > 1e-13 ) {
      throw runtime_error( "simulated delayed signal had imaginary component at index " + to_string( i ) );
    }

    if ( abs( target_reference.at( i ) - simulated_with_delay.at( i ) ) > 1e-11 ) {
      throw runtime_error( "simulated and reference delayed signal differ at index " + to_string( i ) );
    }
  }
}

void program_body( const string& wisdom_filename )
{
  ReadOnlyFile wisdom { wisdom_filename };
  FFTW fftw_state;
  fftw_state.load_wisdom( wisdom );

  /* test configurations chosen so that duration is multiple of sinusoid period */
  /* otherwise it's trickier to compute the correct "target reference" signal */
  test_config( 200, 10, 0.157272, 2000 );
  test_config( 200, 11, 0.157272, 2000 );
  test_config( 200, 12, 0.157272, 2000 );
  test_config( 200, 12, 1.33334, 2000 );
  test_config( 100, 13, 2.71828, 4000 );
  test_config( 128, 8, 1.914512, 2048 );
  test_config( 200000, 25000, 0.00007, 16384 );
}

int main( int argc, char* argv[] )
{
  try {
    ios::sync_with_stdio( false );

    if ( argc <= 0 ) {
      abort();
    }

    if ( argc != 2 ) {
      cerr << "Usage: " << argv[0] << " wisdom_filename\n";
      return EXIT_FAILURE;
    }

    program_body( argv[1] );
    return EXIT_SUCCESS;
  } catch ( const exception& e ) {
    cerr << e.what() << "\n";
    return EXIT_FAILURE;
  }
}
