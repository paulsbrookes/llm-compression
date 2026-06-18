// Milestone 4: rANS (range Asymmetric Numeral System) for fp16 samples.
//
// rANS is what modern ML weight/codec stacks actually use: it reaches the same
// near-optimal rate as arithmetic coding but decodes with table lookups and no
// per-symbol division on the hot path. Same static model as the range coder
// (freqs summing to M = 2^24). rANS encodes the stream in reverse and decodes
// forward; renormalisation moves 32-bit words. Escape symbols are followed by
// the raw 16-bit pattern coded uniformly within the same M.
//
//   build: g++ -O2 -std=c++17 rans.cpp -o rans
//   run:   ./rans [sigma=1.0] [mu=0.0] [N=1000000] [seed=12345]
#include "half.h"
#include "model.h"
#include "pmf.h"
#include <cstdint>
#include <cstdio>
#include <random>
#include <string>
#include <vector>

constexpr uint32_t SCALE = 24;                 // M = 2^SCALE, must match model bits
constexpr uint64_t M = 1ull << SCALE;
constexpr uint64_t RANS_L = 1ull << 31;        // lower bound of normalised state

int main(int argc, char** argv) {
    const double sigma = argc > 1 ? std::stod(argv[1]) : 1.0;
    const double mu    = argc > 2 ? std::stod(argv[2]) : 0.0;
    const long N       = argc > 3 ? std::stol(argv[3]) : 1000000;
    const uint64_t seed = argc > 4 ? std::stoull(argv[4]) : 12345ull;

    auto P = fp16pmf::compute(mu, sigma);
    const double H = fp16pmf::entropy(P);
    const auto mdl = freqmodel::build(P, SCALE);
    const auto& freq = mdl.freq; const auto& cum = mdl.cum;
    using freqmodel::ESCAPE;

    // Sample stream.
    std::mt19937_64 rng(seed);
    std::normal_distribution<double> gauss(mu, sigma);
    std::vector<uint16_t> data((size_t)N);
    for (long i = 0; i < N; ++i) data[i] = half::from_float((float)gauss(rng));

    // ---- encode (reverse order) ----
    std::vector<uint32_t> words; words.reserve((size_t)N / 2 + 16);
    uint64_t x = RANS_L;
    auto put = [&](uint32_t start, uint32_t f) {
        const uint64_t x_max = ((RANS_L >> SCALE) << 32) * f;   // emit 32 bits if state too big
        if (x >= x_max) { words.push_back((uint32_t)x); x >>= 32; }
        x = ((x / f) << SCALE) + (x % f) + start;
    };
    long escapes = 0;
    for (long i = N - 1; i >= 0; --i) {
        uint16_t s = data[i];
        if (freq[s]) put(cum[s], freq[s]);
        else { put((uint32_t)s * 256u, 256u); put(mdl.cum[ESCAPE], freq[ESCAPE]); ++escapes; }
        //       ^ raw 16-bit pattern (freq 256 over M => 16 bits), encoded BEFORE escape
        //         so it decodes AFTER the escape symbol
    }
    words.push_back((uint32_t)x); words.push_back((uint32_t)(x >> 32));   // flush state

    // ---- decode (forward) ----
    long idx = (long)words.size() - 1;
    uint64_t dx = ((uint64_t)words[idx] << 32) | words[idx - 1]; idx -= 2;
    auto get = [&](uint32_t start, uint32_t f, uint32_t slot) {
        dx = (uint64_t)f * (dx >> SCALE) + slot - start;
        if (dx < RANS_L) { dx = (dx << 32) | words[idx]; --idx; }
    };
    bool lossless = true;
    for (long i = 0; i < N; ++i) {
        uint32_t slot = (uint32_t)(dx & (M - 1));
        int s = mdl.findSym(slot);
        get(cum[s], freq[s], slot);
        uint16_t out;
        if (s == ESCAPE) {
            uint32_t slot2 = (uint32_t)(dx & (M - 1));
            out = (uint16_t)(slot2 >> 8);                 // pattern = slot2 / 256
            get((uint32_t)out * 256u, 256u, slot2);
        } else out = (uint16_t)s;
        if (out != data[i]) { lossless = false; break; }
    }

    const double achieved = 32.0 * words.size() / (double)N;
    printf("rANS coder  |  N(mu=%.4g, sigma=%.4g)  |  N=%ld samples\n", mu, sigma, N);
    printf("  coded alphabet (freq>=1)  : %d symbols (+ escape)\n", mdl.coded);
    printf("  escape emissions          : %ld (%.4f%%)\n", escapes, 100.0 * escapes / N);
    printf("  lossless round-trip       : %s\n", lossless ? "PASS" : "FAIL");
    printf("  -------------------------------------------\n");
    printf("  entropy floor   H         : %8.4f bits/sample\n", H);
    printf("  achieved (sampled stream) : %8.4f bits/sample\n", achieved);
    printf("  overhead over floor       : %8.4f bits (%.3f%%)\n", achieved - H, 100.0 * (achieved - H) / H);
    return 0;
}
