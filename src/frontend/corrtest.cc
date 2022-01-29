#include "fft.hh"
#include "random.hh"
#include "signal.hh"

#include <array>
#include <cmath>
#include <cstdlib>
#include <iostream>

using namespace std;

constexpr double SAMPLE_RATE = 200'000;                /* Hz */
constexpr unsigned int TRANSMITTER_SIGNAL_LEN = 16384; /* normally distributed chips */
constexpr double TRANSMITTER_POWER = 10;               /* watts */
constexpr unsigned int TAG_CODE_LEN = 64;              /* binary chips */
constexpr unsigned int TAG_CODE_RATE_INVERSE = 8;      /* sample rate over tag code rate */

TimeDomainSignal synthesize_reflected_signal( const double tag_time_offset,
                                              const double path_delay,
                                              const double path_gain,
                                              const TimeDomainSignal& transmitter_signal,
                                              const TimeDomainSignal& tag_signal )
{
  if ( transmitter_signal.sample_rate() != tag_signal.sample_rate() ) {
    throw runtime_error( "synthesize_reflected_signal: sample-rate mismatch" );
  }

  auto delayed_tag_signal = delay( tag_signal, tag_time_offset );
  auto delayed_transmitter_signal = delay( transmitter_signal, path_delay );
  TimeDomainSignal reflected_signal { transmitter_signal.size(), transmitter_signal.sample_rate() };
  for ( unsigned int i = 0; i < delayed_tag_signal.size(); i++ ) {
    reflected_signal.at( i )
      = path_gain * delayed_transmitter_signal.at( i ) * delayed_tag_signal.at( i % delayed_tag_signal.size() );
  }
  return reflected_signal;
}

TimeDomainSignal decorrelate( const TimeDomainSignal& input, const TimeDomainSignal& nuisance )
{
  const double nuisance_power = accumulate(
    nuisance.signal().begin(), nuisance.signal().end(), 0.0, []( auto x, auto y ) { return x + norm( y ); } );

  const double correlation = input.correlation( nuisance );

  auto ret = input;
  for ( size_t i = 0; i < ret.size(); i++ ) {
    ret.at( i ) -= ( correlation / nuisance_power ) * nuisance.at( i );
  }

  const double new_correlation = ret.correlation( nuisance );
  if ( abs( new_correlation ) > 1e-8 ) {
    throw runtime_error( "ERR: correlation " + to_string( correlation ) + " -> " + to_string( new_correlation ) );
  }

  return ret;
}

void program_body()
{
  auto rng = get_random_generator();

  /* step 1: make signals */
  TimeDomainSignal transmitter_signal { TRANSMITTER_SIGNAL_LEN, SAMPLE_RATE };
  BasebandFrequencyDomainSignal frequency_domain { TRANSMITTER_SIGNAL_LEN, SAMPLE_RATE };
  BasebandFrequencyDomainSignal frequency_domain_copy { TRANSMITTER_SIGNAL_LEN, SAMPLE_RATE };
  TimeDomainSignal delayed_signal = { TRANSMITTER_SIGNAL_LEN, SAMPLE_RATE };
  ForwardFFT fft { transmitter_signal.size() };
  ReverseFFT ifft { frequency_domain.size() };

  /* make transmitter signal */
  normal_distribution<double> transmitter_signal_dist { 0.0, sqrt( TRANSMITTER_POWER ) };
  for ( auto& x : transmitter_signal.signal() ) {
    x = transmitter_signal_dist( rng );
  }

  /* remove mean from transmitter signal */
  double sum = 0;
  for ( const auto& x : transmitter_signal.signal() ) {
    sum += x.real();
  }
  const double mean = sum / transmitter_signal.size();
  for ( auto& x : transmitter_signal.signal() ) {
    x -= mean;
  }

  /* step 2: make tag code */
  array<bool, TAG_CODE_LEN> tag_code {};
  uniform_int_distribution<unsigned int> tag_code_one_bit_position_dist { 0, TAG_CODE_LEN - 1 };

  for ( unsigned int i = 0; i < TAG_CODE_LEN / 2; i++ ) {
    /* find a zero bit and set it to one */
    unsigned int one_position = -1;
    do {
      one_position = tag_code_one_bit_position_dist( rng );
    } while ( tag_code.at( one_position ) );
    tag_code.at( one_position ) = true;
  }

  /* step 2a: synthesize tag signal */
  TimeDomainSignal tag_signal { TAG_CODE_LEN * TAG_CODE_RATE_INVERSE, SAMPLE_RATE };

  for ( unsigned int chip_num = 0; chip_num < tag_code.size(); chip_num++ ) {
    for ( unsigned int sample_num = chip_num * TAG_CODE_RATE_INVERSE;
          sample_num < ( chip_num + 1 ) * TAG_CODE_RATE_INVERSE;
          sample_num++ ) {
      tag_signal.at( sample_num ) = tag_code.at( chip_num ) ? 1 : -1;
    }
  }

  /* make reflected signal */
  const auto reflected_signal = synthesize_reflected_signal( 0, 0, 1, transmitter_signal, tag_signal );
  fft.execute( reflected_signal, frequency_domain );
  frequency_domain.normalize();

  auto reflected_signal_plus_transmitter = reflected_signal;
  for ( size_t i = 0; i < reflected_signal_plus_transmitter.size(); i++ ) {
    reflected_signal_plus_transmitter.at( i ) += 16.0 * transmitter_signal.at( i );
  }

  const auto ref2 = decorrelate( reflected_signal_plus_transmitter, transmitter_signal );

  for ( double offset = -.0001; offset <= .0001; offset += .00000001 ) {
    frequency_domain_copy = frequency_domain;
    frequency_domain_copy.delay( offset );
    ifft.execute( frequency_domain_copy, delayed_signal );
    cout << offset << " " << ref2.correlation( delayed_signal ) << "\n";
  }
}

int main()
{
  try {
    ios::sync_with_stdio( false );
    program_body();
    return EXIT_SUCCESS;
  } catch ( const exception& e ) {
    cerr << e.what() << "\n";
    return EXIT_FAILURE;
  }
}
