#pragma once
#include "half.h"
#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <utility>
#include <vector>

// Exact pmf of N(mu, sigma^2) rounded-to-nearest into fp16. P[h] is the Gaussian
// mass over pattern h's round-to-nearest cell (midpoints to its neighbours);
// tail mass past the largest finite magnitude folds into +/-Inf; NaN -> 0.
namespace fp16pmf {

inline double Phi(double x, double mu, double sigma) {
    return 0.5 * std::erfc(-((x - mu) / sigma) / std::sqrt(2.0));
}

inline std::array<double, 1 << 16> compute(double mu, double sigma) {
    std::vector<std::pair<double, uint16_t>> finite;
    finite.reserve(1 << 16);
    for (uint32_t h = 0; h < (1u << 16); ++h) {
        if (half::is_nan((uint16_t)h) || half::is_inf((uint16_t)h)) continue;
        finite.emplace_back(half::to_double((uint16_t)h), (uint16_t)h);
    }
    std::sort(finite.begin(), finite.end(),
              [](auto& a, auto& b) { return a.first < b.first; });

    std::vector<double> val;                  // unique sorted real values
    std::vector<std::vector<uint16_t>> pats;  // patterns sharing that value (only +0/-0 collide)
    for (auto& [v, p] : finite) {
        if (!val.empty() && v == val.back()) pats.back().push_back(p);
        else { val.push_back(v); pats.push_back({p}); }
    }
    const int n = (int)val.size();

    std::array<double, 1 << 16> P{};
    auto mid = [&](int i, int j) { return 0.5 * (val[i] + val[j]); };
    const double loEdge = val[0]     - 0.5 * (val[1]     - val[0]);
    const double hiEdge = val[n - 1] + 0.5 * (val[n - 1] - val[n - 2]);
    P[half::NEG_INF] = Phi(loEdge, mu, sigma);
    P[half::POS_INF] = 1.0 - Phi(hiEdge, mu, sigma);

    for (int i = 0; i < n; ++i) {
        const double lo = (i == 0)     ? loEdge : mid(i - 1, i);
        const double hi = (i == n - 1) ? hiEdge : mid(i, i + 1);
        if (val[i] == 0.0 && pats[i].size() == 2) {
            const double atZero = Phi(0.0, mu, sigma);
            for (uint16_t p : pats[i])
                P[p] = (p == half::NEG_ZERO) ? atZero - Phi(lo, mu, sigma)
                                             : Phi(hi, mu, sigma) - atZero;
        } else {
            P[pats[i][0]] = Phi(hi, mu, sigma) - Phi(lo, mu, sigma);
        }
    }
    return P;
}

inline double entropy(const std::array<double, 1 << 16>& P) {
    double H = 0.0;
    for (double p : P) if (p > 0.0) H -= p * std::log2(p);
    return H;
}

} // namespace fp16pmf
