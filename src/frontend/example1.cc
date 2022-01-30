#include "fft.hh"
#include "mmap.hh"
#include "random.hh"
#include "signal.hh"

#include <array>
#include <cmath>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <optional>

using namespace std;

constexpr double SAMPLE_RATE = 200'000;                        /* Hz */
constexpr unsigned int TRANSMITTER_SIGNAL_LEN = 16384;         /* normally distributed chips */
constexpr double TRANSMITTER_POWER = 10;                       /* watts */
constexpr unsigned int TAG_CODE_LEN = 64;                      /* binary chips */
constexpr unsigned int TAG_CODE_RATE_INVERSE = 8;              /* sample rate over tag code rate */
constexpr double MAX_PATH_DELAY = 4e-6;                        /* 4 microseconds -> about 1 km */
constexpr double PATH_GAIN = dB_to_amplitude_gain( -100 );     /* dB */
constexpr double LISTEN_DURATION = 60 * 60;                    /* seconds */
constexpr double NOISE_POWER = 1e-3 * dB_to_power_gain( -50 ); /* -50 dBm of receiver noise */
constexpr double SUBDIVISION_STEPS = 4.0;
constexpr double DIRECT_PATH_GAIN = 1.0;

constexpr double ACTUAL_TAG_SAMPLE_OFFSET = TAG_CODE_LEN * TAG_CODE_RATE_INVERSE / 2.0;
constexpr double ACTUAL_PATH_DELAY = MAX_PATH_DELAY / 2.0;

constexpr double SPEED_OF_LIGHT = 299'792'458; /* meters per second in vacuum */

TimeDomainSignal make_transmitter_signal( rng_t& rng );

TimeDomainSignal make_tag_signal( rng_t& rng );

TimeDomainSignal synthesize_reflected_signal( const double tag_time_offset,
                                              const double path_delay,
                                              const double path_gain,
                                              const TimeDomainSignal& transmitter_signal,
                                              const TimeDomainSignal& tag_signal );

TimeDomainSignal decorrelate( const TimeDomainSignal& input, const TimeDomainSignal& nuisance );

pair<double, double> find_best_fit( const TimeDomainSignal& receiver_signal_minus_transmitter,
                                    const TimeDomainSignal& transmitter_signal,
                                    const TimeDomainSignal& tag_signal );

void program_body( const char* const wisdom_filename )
{
  /* load pre-planned FFTs from file */
  ReadOnlyFile wisdom { wisdom_filename };
  FFTW fftw_state;
  fftw_state.load_wisdom( wisdom );

  auto rng = get_random_generator();

  /* step 1: make transmitter signal */
  auto transmitter_signal = make_transmitter_signal( rng );

  /* step 2: make tag signal */
  auto tag_signal = make_tag_signal( rng );

  /* step 3: choose values of unknowns */

  /* Unknown #1 (tag time offset relative to beginning of tag signal) */
  const double tag_time_offset = ACTUAL_TAG_SAMPLE_OFFSET / SAMPLE_RATE;

  /* Unknown #2 (path delay from transmitter->tag->receiver) */
  const double path_delay = ACTUAL_PATH_DELAY;

  /* step 4: synthesize reflected signal */
  const auto reflected_signal
    = synthesize_reflected_signal( tag_time_offset, path_delay, PATH_GAIN, transmitter_signal, tag_signal );

  /* step 5: synthesize receiver signal (before noise) */
  TimeDomainSignal receiver_signal { TRANSMITTER_SIGNAL_LEN, SAMPLE_RATE };
  for ( unsigned int i = 0; i < receiver_signal.size(); i++ ) {
    receiver_signal.at( i ) = reflected_signal.at( i ) + DIRECT_PATH_GAIN * transmitter_signal.at( i );
  }

  /* step 6: simulate listening with signal auto-interfering coherently and noise incoherently */
  const double signal_repetitions = LISTEN_DURATION * SAMPLE_RATE / TRANSMITTER_SIGNAL_LEN;
  normal_distribution<double> noise_distribution { 0, sqrt( signal_repetitions ) * sqrt( NOISE_POWER ) };
  for ( auto& x : receiver_signal.signal() ) {
    x *= signal_repetitions;
    x += noise_distribution( rng );
  }

  /* step 6b: computationally remove direct-path transmitter signal from receiver signal */
  const auto receiver_signal_minus_transmitter = decorrelate( receiver_signal, transmitter_signal );

  /* step 7: search for best values of the two unknowns */
  const auto [inferred_path_delay, inferred_tag_offset]
    = find_best_fit( receiver_signal_minus_transmitter, transmitter_signal, tag_signal );

  /* step 8: print diagnostics */
  cerr << "Correct tag offset: " << tag_time_offset << "\n";
  cerr << "Inferred tag offset: " << inferred_tag_offset << "\n";
  cerr << "Tag offset error: " << 1e9 * abs( tag_time_offset - inferred_tag_offset ) << " ns\n";

  cerr << "\n";

  cerr << "Correct path delay: " << path_delay << "\n";
  cerr << "Inferred path delay: " << inferred_path_delay << "\n";

  cerr << "Path delay error: " << 1e9 * abs( path_delay - inferred_path_delay ) << " ns (approx ";

  const auto old_precision = cerr.precision( 2 );
  cerr << fixed << SPEED_OF_LIGHT * abs( path_delay - inferred_path_delay ) << " meters)\n" << defaultfloat;
  cerr.precision( old_precision );

  cerr << "\n";

  const auto local_reference_inferred
    = synthesize_reflected_signal( inferred_tag_offset, inferred_path_delay, 1, transmitter_signal, tag_signal );
  const double inferred_correlation
    = receiver_signal_minus_transmitter.normalized_correlation( local_reference_inferred );

  cerr << "Correlation at best-fit constants: " << inferred_correlation << "\n";

  const auto local_reference_exact
    = synthesize_reflected_signal( tag_time_offset, path_delay, 1, transmitter_signal, tag_signal );
  const double correct_correlation
    = receiver_signal_minus_transmitter.normalized_correlation( local_reference_exact );

  cerr << "Correlation at true constants: " << correct_correlation << "\n";

  cerr << "Correlation error percentage: "
       << 100 * ( inferred_correlation - correct_correlation ) / correct_correlation << "%\n";
}

TimeDomainSignal make_transmitter_signal( rng_t& rng )
{
  TimeDomainSignal transmitter_signal { TRANSMITTER_SIGNAL_LEN, SAMPLE_RATE };

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

  return transmitter_signal;
}

TimeDomainSignal make_tag_signal( rng_t& rng )
{
  /* first: generate random tag code */
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

  /* now: synthesize tag signal from tag code */
  TimeDomainSignal tag_signal { TAG_CODE_LEN * TAG_CODE_RATE_INVERSE, SAMPLE_RATE };

  for ( unsigned int chip_num = 0; chip_num < tag_code.size(); chip_num++ ) {
    for ( unsigned int sample_num = chip_num * TAG_CODE_RATE_INVERSE;
          sample_num < ( chip_num + 1 ) * TAG_CODE_RATE_INVERSE;
          sample_num++ ) {
      tag_signal.at( sample_num ) = tag_code.at( chip_num ) ? 1 : -1;
    }
  }

  return tag_signal;
}

TimeDomainSignal synthesize_reflected_signal( const double tag_time_offset,
                                              const double path_delay,
                                              const double path_gain,
                                              const TimeDomainSignal& transmitter_signal,
                                              const TimeDomainSignal& tag_signal )
{
  static ForwardFFT tag_fft { tag_signal.size() }, transmitter_fft { transmitter_signal.size() };
  static ReverseFFT tag_ifft { tag_signal.size() }, transmitter_ifft { transmitter_signal.size() };

  if ( transmitter_signal.sample_rate() != tag_signal.sample_rate() ) {
    throw runtime_error( "synthesize_reflected_signal: sample-rate mismatch" );
  }

  auto delayed_tag_signal = delay( tag_signal, tag_time_offset, tag_fft, tag_ifft );
  auto delayed_transmitter_signal = delay( transmitter_signal, path_delay, transmitter_fft, transmitter_ifft );
  TimeDomainSignal reflected_signal { transmitter_signal.size(), transmitter_signal.sample_rate() };
  for ( unsigned int i = 0; i < delayed_tag_signal.size(); i++ ) {
    reflected_signal.at( i )
      = path_gain * delayed_transmitter_signal.at( i ) * delayed_tag_signal.at( i % delayed_tag_signal.size() );
  }
  return reflected_signal;
}

TimeDomainSignal decorrelate( const TimeDomainSignal& input, const TimeDomainSignal& nuisance )
{
  const double nuisance_power = nuisance.power();

  const double correlation = input.correlation( nuisance );

  auto ret = input;
  for ( size_t i = 0; i < ret.size(); i++ ) {
    ret.at( i ) -= ( correlation / nuisance_power ) * nuisance.at( i );
  }

  return ret;
}

pair<double, double> find_best_fit( const TimeDomainSignal& receiver_signal_minus_transmitter,
                                    const TimeDomainSignal& transmitter_signal,
                                    const TimeDomainSignal& tag_signal )
{
  double min_path_delay = 0, max_path_delay = MAX_PATH_DELAY;
  double min_tag_offset = 0, max_tag_offset = double( tag_signal.size() ) / SAMPLE_RATE;
  while ( ( max_path_delay - min_path_delay > 1e-12 ) or ( max_tag_offset - min_tag_offset > 1e-12 ) ) {
    double max_correlation_this_round = 0.0;
    optional<double> best_path_delay;
    optional<double> best_tag_offset;

    cerr << ".";

    const double path_stepsize = ( max_path_delay - min_path_delay ) / SUBDIVISION_STEPS;
    const double tag_stepsize = ( max_tag_offset - min_tag_offset ) / SUBDIVISION_STEPS;
    for ( double guess_path_delay = min_path_delay; guess_path_delay <= max_path_delay + path_stepsize / 2.0;
          guess_path_delay += path_stepsize ) {
      for ( double guess_tag_offset = min_tag_offset; guess_tag_offset <= max_tag_offset + path_stepsize / 2.0;
            guess_tag_offset += tag_stepsize ) {
        const auto local_reference
          = synthesize_reflected_signal( guess_tag_offset, guess_path_delay, 1, transmitter_signal, tag_signal );
        const double correlation = receiver_signal_minus_transmitter.normalized_correlation( local_reference );
        if ( correlation > max_correlation_this_round ) {
          max_correlation_this_round = correlation;
          best_path_delay.emplace( guess_path_delay );
          best_tag_offset.emplace( guess_tag_offset );
        }
      }
    }

    min_path_delay = best_path_delay.value() - 1.01 * path_stepsize;
    max_path_delay = best_path_delay.value() + 1.01 * path_stepsize;

    min_tag_offset = best_tag_offset.value() - 1.01 * tag_stepsize;
    max_tag_offset = best_tag_offset.value() + 1.01 * tag_stepsize;
  }

  const double inferred_path_delay = ( min_path_delay + max_path_delay ) / 2.0;
  const double inferred_tag_offset = ( min_tag_offset + max_tag_offset ) / 2.0;

  cerr << "\n";

  return { inferred_path_delay, inferred_tag_offset };
}

int main( int argc, char* argv[] )
{
  try {
    ios::sync_with_stdio( false );

    if ( argc <= 0 ) {
      abort();
    }

    if ( argc != 2 ) {
      cerr << "Usage: " << argv[0] << " wisdom_file\n";
      return EXIT_FAILURE;
    }

    program_body( argv[1] );
    return EXIT_SUCCESS;
  } catch ( const exception& e ) {
    cerr << e.what() << "\n";
    return EXIT_FAILURE;
  }
}
