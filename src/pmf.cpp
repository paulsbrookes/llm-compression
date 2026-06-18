// Milestone 1: entropy floor H and per-field (sign/exponent/mantissa) breakdown
// for N(mu, sigma^2) rounded into fp16.
//
//   build: g++ -O2 -std=c++17 pmf.cpp -o pmf
//   run:   ./pmf [sigma=1.0] [mu=0.0] [pmf.csv]
#include "pmf.h"
#include "half.h"
#include <array>
#include <cmath>
#include <cstdio>
#include <string>

static double xlog2x(double p) { return p > 0.0 ? -p * std::log2(p) : 0.0; }

int main(int argc, char** argv) {
    const double sigma = argc > 1 ? std::stod(argv[1]) : 1.0;
    const double mu    = argc > 2 ? std::stod(argv[2]) : 0.0;
    const std::string csv = argc > 3 ? argv[3] : "pmf.csv";

    auto P = fp16pmf::compute(mu, sigma);

    double H = 0.0, sum = 0.0;
    std::array<double, 2>    Psign{};
    std::array<double, 32>   Pexp{};
    std::array<double, 1024> Pmant{};
    for (uint32_t h = 0; h < (1u << 16); ++h) {
        const double p = P[h];
        if (p <= 0.0) continue;
        sum += p;
        H   += xlog2x(p);
        Psign[half::sign_bit((uint16_t)h)] += p;
        Pexp[half::exponent((uint16_t)h)]  += p;
        Pmant[half::mantissa((uint16_t)h)] += p;
    }
    double Hsign = 0, Hexp = 0, Hmant = 0;
    for (double p : Psign) Hsign += xlog2x(p);
    for (double p : Pexp)  Hexp  += xlog2x(p);
    for (double p : Pmant) Hmant += xlog2x(p);

    printf("N(mu=%.4g, sigma=%.4g) rounded to fp16\n", mu, sigma);
    printf("  probability mass captured : %.10f\n", sum);
    printf("  entropy H                 : %8.4f bits/sample\n", H);
    printf("  naive storage             : %8.4f bits/sample\n", 16.0);
    printf("  savings vs naive          : %8.4f bits/sample\n", 16.0 - H);
    printf("  --- per-field marginal entropy ---\n");
    printf("  sign     : %7.4f bits\n", Hsign);
    printf("  exponent : %7.4f bits\n", Hexp);
    printf("  mantissa : %7.4f bits\n", Hmant);
    printf("  sum (independent-field coder) : %7.4f bits  (>= H, gap = correlation)\n",
           Hsign + Hexp + Hmant);

    if (FILE* f = std::fopen(csv.c_str(), "w")) {
        std::fprintf(f, "pattern,value,prob\n");
        for (uint32_t h = 0; h < (1u << 16); ++h)
            if (P[h] > 0.0)
                std::fprintf(f, "%u,%.17g,%.17g\n", h, half::to_double((uint16_t)h), P[h]);
        std::fclose(f);
        printf("  wrote pmf -> %s\n", csv.c_str());
    }
    return 0;
}
