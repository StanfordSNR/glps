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
constexpr unsigned int TAG_CODE_LEN = 256;                     /* binary chips */
constexpr unsigned int TAG_CODE_RATE_INVERSE = 32;             /* sample rate over tag code rate */
constexpr double MAX_PATH_DELAY = 4e-6;                        /* 4 microseconds -> about 1 km */
constexpr double PATH_GAIN = dB_to_amplitude_gain( -100 );     /* dB */
constexpr double LISTEN_DURATION = 60 * 60;                    /* seconds */
constexpr double NOISE_POWER = 1e-3 * dB_to_power_gain( -50 ); /* -50 dBm of receiver noise */
constexpr double SUBDIVISION_STEPS = 4.0;                      /* steps of division for fitting algorithm */
constexpr double DIRECT_PATH_GAIN = 1.0;                       /* gain on direct path */
constexpr double DIRECT_PATH_DELAY = 0;                        /* delay of direct path (s) */

/* true value of unknowns */
constexpr double ACTUAL_TAG_SAMPLE_OFFSET = TAG_CODE_LEN * TAG_CODE_RATE_INVERSE / 2.0;
constexpr double ACTUAL_PATH_DELAY = MAX_PATH_DELAY / 2.0;

constexpr double SPEED_OF_LIGHT = 299'792'458; /* meters per second in vacuum */

/* helper routine to make the transmitter signal */
TimeDomainSignal make_transmitter_signal( rng_t& rng );

/* helper routine to make the tag signal */
TimeDomainSignal make_tag_signal( rng_t& rng );

/* helper routine to modulate the transmitter signal by the tag signal */
TimeDomainSignal synthesize_reflected_signal( const double tag_time_offset,
                                              const double path_delay,
                                              const double path_gain,
                                              const TimeDomainSignal& transmitter_signal,
                                              const TimeDomainSignal& tag_signal );

/* learn (and reverse) the channel transfer function by comparing transmitted and received signals,
   relying on the periodic property of the transmitter signal (it repeats within half its length),
   and the fact that reflected signal has no correlation with transmitter signal because of
   the structure of the tag signal (it's opposite in the second half of its length) */
TimeDomainSignal remove_direct_path( const TimeDomainSignal& received_signal,
                                     const TimeDomainSignal& transmitter_signal );

/* find values of tag_time_offset and path_delay that produce a local-reference reflected signal that maximizes
 * correlation with what was received */
pair<double, double> find_best_fit( const TimeDomainSignal& receiver_signal_minus_transmitter,
                                    const TimeDomainSignal& transmitter_signal,
                                    const TimeDomainSignal& tag_signal );

/* calculate and print post-detection SNR statistics */
void print_post_detection_snr( const string_view name,
                               const double tag_offset,
                               const double path_delay,
                               const TimeDomainSignal& transmitter_signal,
                               const TimeDomainSignal& tag_signal,
                               const TimeDomainSignal& receiver_signal_minus_transmitter );

/* run one realization of the simulation */
void run_simulation( rng_t& rng );

/* main loop */
void program_body( const char* const wisdom_filename, const char* const count_str )
{
  /* load pre-planned FFTs from file */
  ReadOnlyFile wisdom { wisdom_filename };
  FFTW fftw_state;
  fftw_state.load_wisdom( wisdom );

  auto rng = get_random_generator();

  const unsigned int realization_count = stoul( count_str );

  for ( unsigned int i = 0; i < realization_count; i++ ) {
    run_simulation( rng );
  }
}

void run_simulation( rng_t& rng )
{
  /* step 1: make transmitter signal */
  auto transmitter_signal = make_transmitter_signal( rng );

  /* step 2: make tag signal */
  auto tag_signal = make_tag_signal( rng );

  /* step 3: choose values of unknowns */

  /* Unknown #1 (tag time offset relative to beginning of tag signal) */
  const double actual_tag_time_offset = ACTUAL_TAG_SAMPLE_OFFSET / SAMPLE_RATE;

  /* Unknown #2 (path delay from transmitter->tag->receiver) */
  const double actual_path_delay = ACTUAL_PATH_DELAY;

  /* step 4: synthesize reflected signal */
  const auto reflected_signal = synthesize_reflected_signal(
    actual_tag_time_offset, actual_path_delay, PATH_GAIN, transmitter_signal, tag_signal );

  if ( abs( reflected_signal.normalized_correlation( transmitter_signal ) ) > 1e-10 ) {
    throw runtime_error( "Spurious correlation between reflected and transmitter signals" );
  }

  /* step 5: synthesize receiver signal (before noise) with signal adding coherently based on listen duration */
  const double signal_repetitions = LISTEN_DURATION * SAMPLE_RATE / TRANSMITTER_SIGNAL_LEN;

  const auto reflected_signal_after_listening = reflected_signal * signal_repetitions;
  const auto direct_path_after_listening
    = delay( transmitter_signal * DIRECT_PATH_GAIN * signal_repetitions, DIRECT_PATH_DELAY );

  TimeDomainSignal receiver_signal_before_noise = reflected_signal_after_listening + direct_path_after_listening;

  /* step 6: simulate noise adding incoherently */
  normal_distribution<double> noise_distribution { 0, sqrt( signal_repetitions ) * sqrt( NOISE_POWER ) };

  TimeDomainSignal noise_signal { TRANSMITTER_SIGNAL_LEN, SAMPLE_RATE };
  for ( auto& x : noise_signal.signal() ) {
    x = noise_distribution( rng );
  }

  const auto receiver_signal = receiver_signal_before_noise + noise_signal;

  /* interlude: print signal statistics */
  cout << "Power gain from accumulating coherently for " << LISTEN_DURATION << " seconds (" << signal_repetitions
       << " repetitions): about " << amplitude_gain_to_dB( signal_repetitions ) << " dB\n";

  cout << "\n";

  cout << "   Reflected signal power: " << power_gain_to_dB( 1e3 * reflected_signal.power() ) << " dBm\n";

  cout << "   After accumulation at receiver, reflected signal power: "
       << power_gain_to_dB( 1e3 * reflected_signal_after_listening.power() ) << " dBm\n";

  cout << "\n";

  cout << "   Transmitter power: " << power_gain_to_dB( 1e3 * transmitter_signal.power() ) << " dBm\n";

  cout << "   Direct path delay: " << 1e9 * DIRECT_PATH_DELAY << " ns\n";

  cout << "   Direct path gain: " << amplitude_gain_to_dB( DIRECT_PATH_GAIN ) << " dB\n";

  cout << "   After accumulation at receiver, transmitter power [direct path]: "
       << power_gain_to_dB( 1e3 * direct_path_after_listening.power() ) << " dBm\n";

  cout << "\n";

  cout << "Power gain from accumulating in quadrature (incoherently) for " << LISTEN_DURATION << " seconds ("
       << signal_repetitions << " repetitions): about " << amplitude_gain_to_dB( sqrt( signal_repetitions ) )
       << " dB\n";

  cout << "\n";

  cout << "   Noise power: " << power_gain_to_dB( 1e3 * NOISE_POWER ) << " dBm\n";

  cout << "   After accumulation at receiver, noise power: " << power_gain_to_dB( 1e3 * noise_signal.power() )
       << " dBm\n";

  cout << "\n";

  cout << "Expected processing gain from accumulating for " << LISTEN_DURATION
       << " seconds: " << amplitude_gain_to_dB( sqrt( signal_repetitions ) ) << " dB\n";

  cout << "\n";

  cout << "Incident SINR [reflected_signal / (direct path + noise)]: "
       << power_gain_to_dB( reflected_signal_after_listening.power()
                            / ( direct_path_after_listening + noise_signal ).power() )
       << " dB\n";

  /* step 6b: computationally remove direct-path transmitter signal from receiver signal */
  const auto receiver_signal_minus_transmitter = remove_direct_path( receiver_signal, transmitter_signal );

  if ( abs( receiver_signal_minus_transmitter.normalized_correlation( transmitter_signal ) ) > 1e-11 ) {
    throw runtime_error( "could not remove direct path" );
  }

  const auto pre_detection_residue = receiver_signal_minus_transmitter - reflected_signal_after_listening;

  cout << "Pre-detection SNR [reflected_signal / (reconstructed_reflected_signal - reflected_signal): "
       << power_gain_to_dB( reflected_signal_after_listening.power() / pre_detection_residue.power() ) << " dB\n";

  cout << "\n";

  /* step 7: search for best values of the two unknowns */
  const auto [inferred_path_delay, inferred_tag_offset]
    = find_best_fit( receiver_signal_minus_transmitter, transmitter_signal, tag_signal );

  /* step 8: print diagnostics */
  cout << "\n";
  cout << "   Correct tag offset: " << actual_tag_time_offset * 1e3 << " ms\n";
  cout << "   Inferred tag offset: " << inferred_tag_offset * 1e3 << " ms\n";
  cout << "   Tag offset error: " << 1e9 * abs( actual_tag_time_offset - inferred_tag_offset ) << " ns\n";

  cout << "\n";

  cout << "   Correct path delay: " << actual_path_delay * 1e6 << " μs\n";
  cout << "   Inferred path delay: " << inferred_path_delay * 1e6 << " μs\n";

  cout << "   Path delay error: " << 1e9 * abs( actual_path_delay - inferred_path_delay ) << " ns (approx ";

  const auto old_precision = cout.precision( 2 );
  cout << fixed << SPEED_OF_LIGHT * abs( actual_path_delay - inferred_path_delay ) << " meters)\n" << defaultfloat;
  cout.precision( old_precision );

  cout << "\n";

  print_post_detection_snr( "Empirical",
                            inferred_tag_offset,
                            inferred_path_delay,
                            transmitter_signal,
                            tag_signal,
                            receiver_signal_minus_transmitter );

  cout << "\n";

  print_post_detection_snr( "Oracular",
                            actual_tag_time_offset,
                            actual_path_delay,
                            transmitter_signal,
                            tag_signal,
                            receiver_signal_minus_transmitter );
}

TimeDomainSignal make_transmitter_signal( rng_t& rng )
{
  TimeDomainSignal transmitter_signal { TRANSMITTER_SIGNAL_LEN, SAMPLE_RATE };

  normal_distribution<double> transmitter_signal_dist { 0.0, sqrt( TRANSMITTER_POWER ) };
  for ( size_t i = 0; i < TRANSMITTER_SIGNAL_LEN / 2; i++ ) {
    const auto x = transmitter_signal_dist( rng );
    transmitter_signal.at( i ) = x;
    transmitter_signal.at( i + TRANSMITTER_SIGNAL_LEN / 2 ) = x;
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

  /* scale to achieve exact power goal */
  const double current_power = transmitter_signal.power();
  const double scaling_factor = sqrt( TRANSMITTER_POWER / current_power );
  for ( auto& x : transmitter_signal.signal() ) {
    x *= scaling_factor;
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
  TimeDomainSignal tag_signal { TRANSMITTER_SIGNAL_LEN, SAMPLE_RATE };

  for ( size_t sample_num = 0; sample_num < tag_signal.size(); sample_num++ ) {
    const size_t chip_num = ( sample_num / TAG_CODE_RATE_INVERSE ) % TAG_CODE_LEN;

    tag_signal.at( sample_num ) = tag_code.at( chip_num ) ? 1 : -1;

    if ( sample_num >= tag_signal.size() / 2 ) {
      tag_signal.at( sample_num ) *= -1;
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
  if ( ( transmitter_signal.size() != tag_signal.size() )
       or ( transmitter_signal.sample_rate() != tag_signal.sample_rate() ) ) {
    throw runtime_error( "synthesize_reflected_signal: signalmismatch" );
  }

  static ForwardFFT fft { tag_signal.size() };
  static ReverseFFT ifft { tag_signal.size() };

  auto delayed_tag_signal = delay( tag_signal, tag_time_offset, fft, ifft );
  auto delayed_transmitter_signal = delay( transmitter_signal, path_delay, fft, ifft );
  TimeDomainSignal reflected_signal { transmitter_signal.size(), transmitter_signal.sample_rate() };
  for ( unsigned int i = 0; i < reflected_signal.size(); i++ ) {
    reflected_signal.at( i ) = path_gain * delayed_transmitter_signal.at( i ) * delayed_tag_signal.at( i );
  }
  return reflected_signal;
}

TimeDomainSignal remove_direct_path( const TimeDomainSignal& received_signal,
                                     const TimeDomainSignal& transmitter_signal )
{
  if ( ( received_signal.size() != transmitter_signal.size() )
       or ( received_signal.sample_rate() != transmitter_signal.sample_rate() ) ) {
    throw runtime_error( "remove_direct_path: signal mismatch" );
  }

  const size_t size = received_signal.size();

  if ( size % 2 ) {
    throw runtime_error( "remove_direct_path: signal size must be even" );
  }

  const size_t halfsize = received_signal.size() / 2;
  const double rate = received_signal.sample_rate();

  TimeDomainSignal received_folded { halfsize, rate };
  for ( size_t i = 0; i < halfsize; i++ ) {
    received_folded.at( i ) = received_signal.at( i ) + received_signal.at( i + halfsize );
  }

  TimeDomainSignal transmitter_half { halfsize, rate };
  for ( size_t i = 0; i < halfsize; i++ ) {
    transmitter_half.at( i ) = transmitter_signal.at( i ) + transmitter_signal.at( i + halfsize );
    if ( transmitter_signal.at( i ) != transmitter_signal.at( i + halfsize ) ) {
      throw runtime_error( "remove_direct_path: transmitter signal not periodic in half its length" );
    }
  }

  ForwardFFT fft_half { halfsize };
  ForwardFFT fft_full { size };
  ReverseFFT ifft_full { size };

  BasebandFrequencyDomainSignal received_f { size, rate };
  BasebandFrequencyDomainSignal transmitter_f { size, rate };
  BasebandFrequencyDomainSignal received_folded_f { halfsize, rate };
  BasebandFrequencyDomainSignal transmitter_half_f { halfsize, rate };

  fft_full.execute( received_signal, received_f );
  fft_full.execute( transmitter_signal, transmitter_f );

  fft_half.execute( received_folded, received_folded_f );
  fft_half.execute( transmitter_half, transmitter_half_f );

  /* This is the meat of the approach: learn the gain of the direct path by averaging over the two
     (opposite-polarity) phases of the tag signal -- the transmitter signal repeats twice
     within the full interval, once modulated by the positive tag signal and once modulated by the
     negative. This is invariant to shifts in the tag code.
     Then subtract that gain [times the transmitter signal] from the received signal]. */

  for ( size_t i = 0; i < size; i++ ) {
    const complex direct_gain_estimate = received_folded_f.at( i / 2 ) / transmitter_half_f.at( i / 2 );
    received_f.at( i ) = ( received_f.at( i ) - direct_gain_estimate * transmitter_f.at( i ) ) / double( size );
  }

  received_f.verify_hermitian();

  TimeDomainSignal ret { size, rate };

  ifft_full.execute( received_f, ret );

  return ret;
}

pair<double, double> find_best_fit( const TimeDomainSignal& receiver_signal_minus_transmitter,
                                    const TimeDomainSignal& transmitter_signal,
                                    const TimeDomainSignal& tag_signal )
{
  double min_path_delay = 0, max_path_delay = MAX_PATH_DELAY;
  double min_tag_offset = 0, max_tag_offset = double( tag_signal.size() ) / SAMPLE_RATE;
  cerr << "Finding best fit... ";

  while ( ( max_path_delay - min_path_delay > 1e-12 ) or ( max_tag_offset - min_tag_offset > 1e-12 ) ) {
    double max_correlation_this_round = 0.0;
    optional<double> best_path_delay;
    optional<double> best_tag_offset;

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

  cerr << "done.\n";

  return { inferred_path_delay, inferred_tag_offset };
}

void print_post_detection_snr( const string_view name,
                               const double tag_offset,
                               const double path_delay,
                               const TimeDomainSignal& transmitter_signal,
                               const TimeDomainSignal& tag_signal,
                               const TimeDomainSignal& receiver_signal_minus_transmitter )
{
  const auto local_reference
    = synthesize_reflected_signal( tag_offset, path_delay, 1, transmitter_signal, tag_signal );

  const double power_correction_factor = receiver_signal_minus_transmitter.power() / local_reference.power();

  const auto local_reference_for_comparison = local_reference * sqrt( power_correction_factor );

  const double residual_power = ( receiver_signal_minus_transmitter - local_reference_for_comparison ).power();

  const double correlation
    = receiver_signal_minus_transmitter.normalized_correlation( local_reference_for_comparison );

  cout << name << " correlation: " << correlation << " => " << power_gain_to_dB( .5 / ( 1 - correlation ) )
       << " \"correlation SNR\"\n";

  cout << name
       << " post-detection SNR: " << power_gain_to_dB( receiver_signal_minus_transmitter.power() / residual_power )
       << " dB\n";
}

int main( int argc, char* argv[] )
{
  try {
    ios::sync_with_stdio( false );

    if ( argc <= 0 ) {
      abort();
    }

    if ( argc != 3 ) {
      cerr << "Usage: " << argv[0] << " wisdom_file num_realizations\n";
      return EXIT_FAILURE;
    }

    program_body( argv[1], argv[2] );
    return EXIT_SUCCESS;
  } catch ( const exception& e ) {
    cerr << e.what() << "\n";
    return EXIT_FAILURE;
  }
}
