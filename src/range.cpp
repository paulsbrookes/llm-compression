// Milestone 3: range coder (arithmetic coding) for fp16 samples of N(mu,sigma^2).
//
// Same static model as Huffman (built from the exact pmf), but frequencies are
// quantised at high resolution (TOT = 2^24) instead of being rounded to whole
// bits, so the achieved rate sits much closer to the entropy floor H. Symbols
// too rare to earn a frequency >= 1 fold into an ESCAPE symbol (escape codeword
// + the 16-bit pattern sent as two uniform bytes), keeping the codec lossless
// for any input. We sample, encode, decode, verify the round-trip, and compare.
//
//   build: g++ -O2 -std=c++17 range.cpp -o range
//   run:   ./range [sigma=1.0] [mu=0.0] [N=1000000] [seed=12345]
#include "half.h"
#include "model.h"
#include "pmf.h"
#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <random>
#include <string>
#include <vector>

constexpr int ESCAPE = 1 << 16;
constexpr uint32_t TOTBITS = 24, TOT = 1u << TOTBITS;   // frequency resolution
constexpr uint64_t TOP = 1ULL << 56, BOT = 1ULL << 48;  // 64-bit renorm bounds (TOT < BOT)

// ---- Subbotin carryless range coder, 64-bit --------------------------------
// range floor is BOT=2^48, so range/tot stays >= 2^24: high frequency
// resolution AND negligible arithmetic-rounding loss at the same time.
struct RangeEnc {
    std::vector<uint8_t>& out; uint64_t low = 0, range = ~0ULL;
    explicit RangeEnc(std::vector<uint8_t>& o) : out(o) {}
    void encode(uint64_t cum, uint64_t freq, uint64_t tot) {
        range /= tot; low += cum * range; range *= freq;
        while ((low ^ (low + range)) < TOP || (range < BOT && ((range = (~low + 1) & (BOT - 1)), true)))
            { out.push_back((uint8_t)(low >> 56)); low <<= 8; range <<= 8; }
    }
    void raw8(uint32_t b) { encode(b, 1, 256); }        // one uniform byte
    void flush() { for (int i = 0; i < 8; ++i) { out.push_back((uint8_t)(low >> 56)); low <<= 8; } }
};
struct RangeDec {
    const std::vector<uint8_t>& in; size_t pos = 0; uint64_t low = 0, range = ~0ULL, code = 0;
    explicit RangeDec(const std::vector<uint8_t>& i) : in(i) {}
    uint8_t next() { return pos < in.size() ? in[pos++] : 0; }
    void init() { for (int i = 0; i < 8; ++i) code = (code << 8) | next(); }
    uint64_t getFreq(uint64_t tot) { range /= tot; uint64_t f = (code - low) / range; return f < tot ? f : tot - 1; }
    void decode(uint64_t cum, uint64_t freq) {
        low += cum * range; range *= freq;
        while ((low ^ (low + range)) < TOP || (range < BOT && ((range = (~low + 1) & (BOT - 1)), true)))
            { code = (code << 8) | next(); low <<= 8; range <<= 8; }
    }
    uint32_t raw8() { uint64_t b = getFreq(256); decode(b, 1); return (uint32_t)b; }
};

int main(int argc, char** argv) {
    const double sigma = argc > 1 ? std::stod(argv[1]) : 1.0;
    const double mu    = argc > 2 ? std::stod(argv[2]) : 0.0;
    const long N       = argc > 3 ? std::stol(argv[3]) : 1000000;
    const uint64_t seed = argc > 4 ? std::stoull(argv[4]) : 12345ull;

    auto P = fp16pmf::compute(mu, sigma);
    const double H = fp16pmf::entropy(P);

    // Static frequency model (shared with rANS), quantised at TOT = 2^TOTBITS.
    const auto M = freqmodel::build(P, TOTBITS);
    const auto& freq = M.freq; const auto& cum = M.cum;
    const int coded = M.coded;
    auto findSym = [&](uint32_t f) { return M.findSym(f); };

    // Sample stream.
    std::mt19937_64 rng(seed);
    std::normal_distribution<double> gauss(mu, sigma);
    std::vector<uint16_t> data((size_t)N);
    for (long i = 0; i < N; ++i) data[i] = half::from_float((float)gauss(rng));

    // Encode.
    std::vector<uint8_t> buf; RangeEnc enc(buf); long escapes = 0;
    for (uint16_t s : data) {
        if (freq[s]) enc.encode(cum[s], freq[s], TOT);
        else { enc.encode(cum[ESCAPE], freq[ESCAPE], TOT); enc.raw8(s >> 8); enc.raw8(s & 0xFF); ++escapes; }
    }
    enc.flush();

    // Decode + verify.
    RangeDec dec(buf); dec.init(); bool lossless = true;
    for (long i = 0; i < N; ++i) {
        int s = findSym(dec.getFreq(TOT));
        dec.decode(cum[s], freq[s]);
        uint16_t out = (s == ESCAPE) ? (uint16_t)((dec.raw8() << 8) | dec.raw8()) : (uint16_t)s;
        if (out != data[i]) { lossless = false; break; }
    }

    const double achieved = 8.0 * buf.size() / (double)N;
    printf("Range coder  |  N(mu=%.4g, sigma=%.4g)  |  N=%ld samples\n", mu, sigma, N);
    printf("  coded alphabet (freq>=1)  : %d symbols (+ escape)\n", coded);
    printf("  escape emissions          : %ld (%.4f%%)\n", escapes, 100.0 * escapes / N);
    printf("  lossless round-trip       : %s\n", lossless ? "PASS" : "FAIL");
    printf("  -------------------------------------------\n");
    printf("  entropy floor   H         : %8.4f bits/sample\n", H);
    printf("  achieved (sampled stream) : %8.4f bits/sample\n", achieved);
    printf("  overhead over floor       : %8.4f bits (%.3f%%)\n", achieved - H, 100.0 * (achieved - H) / H);
    return 0;
}
