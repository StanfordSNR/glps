#include "fft.hh"
#include "exception.hh"

#include <thread>

using namespace std;

/* make sure that global FFTW state is cleaned up when program exits */
class FFTW
{
public:
  FFTW()
  {
    if ( not fftw_init_threads() ) {
      throw unix_error( "fftw_init_threads" );
    }

    if ( thread::hardware_concurrency() > 1 ) {
      fftw_plan_with_nthreads( thread::hardware_concurrency() );
    }
  }

  ~FFTW() { fftw_cleanup_threads(); }
};

FFTW global_fftw_state;

FFTPlan::FFTPlan( Signal& input, Signal& output, const int sign )
  : size_( input.size() )
  , plan_( notnull( "fftw_plan_dft_1d",
                    fftw_plan_dft_1d( size_,
                                      reinterpret_cast<fftw_complex*>( input.data() ),
                                      reinterpret_cast<fftw_complex*>( output.data() ),
                                      sign,
                                      FFTW_ESTIMATE | FFTW_PRESERVE_INPUT ) ) )
{
  if ( output.size() != size_ ) {
    throw runtime_error( "output size cannot be less than input size" );
  }

  if ( fftw_alignment_of( reinterpret_cast<double*>( input.data() ) ) ) {
    throw runtime_error( "input not sufficiently aligned" );
  }

  if ( fftw_alignment_of( reinterpret_cast<double*>( output.data() ) ) ) {
    throw runtime_error( "output not sufficiently aligned" );
  }
}

FFTPlan::~FFTPlan()
{
  fftw_destroy_plan( plan_ );
}

void FFTPlan::execute( const Signal& input, Signal& output )
{
  if ( input.size() != size_ or output.size() != size_ ) {
    throw runtime_error( "size mismatch" );
  }

  if ( fftw_alignment_of( const_cast<double*>( reinterpret_cast<const double*>( input.data() ) ) ) ) {
    throw runtime_error( "input not sufficiently aligned" );
  }

  if ( fftw_alignment_of( const_cast<double*>( reinterpret_cast<const double*>( output.data() ) ) ) ) {
    throw runtime_error( "output not sufficiently aligned" );
  }

  /* okay to cast to mutable because plan was FFTW_PRESERVE_INPUT */
  fftw_execute_dft( plan_,
                    const_cast<fftw_complex*>( reinterpret_cast<const fftw_complex*>( input.data() ) ),
                    reinterpret_cast<fftw_complex*>( output.data() ) );
}

void ForwardFFT::execute( const TimeDomainSignal& input, BasebandFrequencyDomainSignal& output )
{
  if ( input.sample_rate() != output.sample_rate() ) {
    throw runtime_error( "sample-rate mismatch" );
  }

  plan_.execute( input.signal(), output.signal() );
}

void ReverseFFT::execute( const BasebandFrequencyDomainSignal& input, TimeDomainSignal& output )
{
  if ( input.sample_rate() != output.sample_rate() ) {
    throw runtime_error( "sample-rate mismatch" );
  }

  plan_.execute( input.signal(), output.signal() );
}

TimeDomainSignal delay( const TimeDomainSignal& signal, const double tau )
{
  TimeDomainSignal signal_copy { signal.size(), signal.sample_rate() };
  BasebandFrequencyDomainSignal frequency_domain { signal.size(), signal.sample_rate() };
  ForwardFFT fft { signal_copy, frequency_domain };
  ReverseFFT ifft { frequency_domain, signal_copy };

  signal_copy = signal;
  fft.execute( signal_copy, frequency_domain );
  frequency_domain.delay_and_normalize( tau );
  ifft.execute( frequency_domain, signal_copy );
  return signal_copy;
}
