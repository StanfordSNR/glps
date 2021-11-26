#pragma once

#include "signal.hh"

#include <complex>
#include <fftw3.h> // must be included last, after <complex>

/* N.B.: Plan must be created before input arrays are initialized with data */
/* See https://www.fftw.org/fftw3_doc/Complex-One_002dDimensional-DFTs.html */

class FFTPlan
{
  size_t size_;
  fftw_plan plan_;

public:
  FFTPlan( Signal& input, Signal& output, const int sign );
  ~FFTPlan();
  void execute( const Signal& input, Signal& output );

  FFTPlan( const FFTPlan& other ) = delete;
  FFTPlan& operator=( const FFTPlan& other ) = delete;
};

class ForwardFFT
{
  FFTPlan plan_;

public:
  ForwardFFT( TimeDomainSignal& input, BasebandFrequencyDomainSignal& output )
    : plan_( input.signal(), output.signal(), FFTW_FORWARD )
  {}

  void execute( const TimeDomainSignal& input, BasebandFrequencyDomainSignal& output );
};

class ReverseFFT
{
  FFTPlan plan_;

public:
  ReverseFFT( BasebandFrequencyDomainSignal& input, TimeDomainSignal& output )
    : plan_( input.signal(), output.signal(), FFTW_BACKWARD )
  {}

  void execute( const BasebandFrequencyDomainSignal& input, TimeDomainSignal& output );
};

TimeDomainSignal delay( const TimeDomainSignal& signal, const double tau );
