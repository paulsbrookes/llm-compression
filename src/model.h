#pragma once
#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <vector>

// Shared static frequency model for the entropy coders (range, rANS).
// Quantises an exact pmf into integer frequencies summing to exactly TOT=2^bits.
// Symbols too rare to earn freq>=1 fold into ESCAPE (id 65536); the coder emits
// the escape symbol followed by the raw 16-bit pattern, so it stays lossless.
namespace freqmodel {

constexpr int ESCAPE = 1 << 16;

struct Model {
    uint32_t TOT = 0;
    int coded = 0;                          // symbols with freq >= 1 (excl. escape)
    std::vector<uint32_t> freq, cum;        // indexed by symbol id (size ESCAPE+1)
    std::vector<uint32_t> starts;           // cum-start per coded symbol, ascending
    std::vector<int> syms;                  // symbol id per coded symbol

    // Map a cumulative-frequency value in [0,TOT) to its symbol id.
    int findSym(uint32_t f) const {
        int i = (int)(std::upper_bound(starts.begin(), starts.end(), f) - starts.begin()) - 1;
        return syms[i];
    }
};

inline Model build(const std::array<double, 1 << 16>& P, uint32_t bits) {
    Model m;
    m.TOT = 1u << bits;
    m.freq.assign(ESCAPE + 1, 0);
    m.cum.assign(ESCAPE + 1, 0);
    double esc = 0;
    for (uint32_t h = 0; h < (1u << 16); ++h) {
        if (P[h] <= 0.0) continue;
        uint32_t f = (uint32_t)llround(P[h] * m.TOT);
        if (f == 0) esc += P[h]; else { m.freq[h] = f; ++m.coded; }
    }
    m.freq[ESCAPE] = std::max(1u, (uint32_t)llround(esc * m.TOT));
    // fix rounding drift by adjusting the most frequent symbol so freqs sum to TOT
    uint64_t sum = 0; int am = ESCAPE;
    for (int s = 0; s <= ESCAPE; ++s) { sum += m.freq[s]; if (m.freq[s] > m.freq[am]) am = s; }
    m.freq[am] = (uint32_t)((int64_t)m.freq[am] + ((int64_t)m.TOT - (int64_t)sum));
    // cumulative table + decode index
    uint32_t c = 0;
    for (int s = 0; s <= ESCAPE; ++s) {
        if (!m.freq[s]) continue;
        m.cum[s] = c; m.starts.push_back(c); m.syms.push_back(s); c += m.freq[s];
    }
    return m;
}

} // namespace freqmodel
