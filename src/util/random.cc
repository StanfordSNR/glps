#include "random.hh"

#include <algorithm>
#include <array>

using namespace std;

rng_t get_random_generator()
{
  auto rd = random_device();
  array<uint32_t, mt19937::state_size> seed_data {};
  generate( seed_data.begin(), seed_data.end(), [&] { return rd(); } );
  seed_seq seed( seed_data.begin(), seed_data.end() );
  return mt19937 { seed };
}
