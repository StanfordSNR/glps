#include "signal.hh"

#include <numeric>

using namespace std;

void BasebandFrequencyDomainSignal::delay( const double tau )
{
  const size_t length = size();
  for ( size_t i = 0; i < length; i++ ) {
    const double f = index_to_frequency( i );
    const double delay_in_radians = -2 * M_PI * tau * f;
    const complex multiplier = exp( complex { 0.0, delay_in_radians } );

    signal_[i] *= multiplier;
  }
}

void BasebandFrequencyDomainSignal::normalize()
{
  const double normalization = 1.0 / size();
  for ( auto& x : signal_ ) {
    x *= normalization;
  }
}

void BasebandFrequencyDomainSignal::delay_and_normalize( const double tau )
{
  const size_t length = size();
  const double normalization = 1.0 / length;
  for ( size_t i = 0; i < length; i++ ) {
    const double f = index_to_frequency( i );
    const double delay_in_radians = -2 * M_PI * tau * f;
    const complex multiplier = exp( complex { 0.0, delay_in_radians } );

    signal_[i] *= multiplier * normalization;
  }
}

double TimeDomainSignal::power() const
{
  return accumulate( signal().begin(), signal().end(), 0.0, []( auto x, auto y ) { return x + norm( y ); } );
}

double TimeDomainSignal::correlation( const TimeDomainSignal& other ) const
{
  if ( size() != other.size() ) {
    throw runtime_error( "correlation: size mismatch" );
  }

  if ( sample_rate() != other.sample_rate() ) {
    throw runtime_error( "correlation: sample-rate mismatch" );
  }

  complex<double> correlation = 0;

  const size_t length = size();
  for ( size_t i = 0; i < length; i++ ) {
    correlation += conj( signal_[i] ) * other.signal_[i];
  }

  return correlation.real();
}

double TimeDomainSignal::normalized_correlation( const TimeDomainSignal& other ) const
{
  return correlation( other ) / sqrt( power() * other.power() );
}
