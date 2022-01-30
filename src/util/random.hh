#include <random>

// Borrowing from
// https://kristerw.blogspot.com/2017/05/seeding-stdmt19937-random-number-engine.html
// and https://www.pcg-random.org/posts/cpps-random_device.html via rsw

using rng_t = std::mt19937;

rng_t get_random_generator();
