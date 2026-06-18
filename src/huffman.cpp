// Milestone 2: Huffman codec for fp16 samples of N(mu, sigma^2).
//
// The code is built from the *exact* pmf (a static code matched to the source),
// so its expected length L satisfies the source-coding bound H <= L < H+1.
// Symbols with probability below EPS are folded into an ESCAPE leaf (emit escape
// codeword + 16 raw bits), which bounds tree depth and makes the codec lossless
// for ANY fp16 input. We then sample a real stream, encode/decode it, verify the
// round-trip is exact, and compare achieved bits/sample to L and to the floor H.
//
//   build: g++ -O2 -std=c++17 huffman.cpp -o huffman
//   run:   ./huffman [sigma=1.0] [mu=0.0] [N=1000000] [seed=12345]
#include "half.h"
#include "pmf.h"
#include <cstdint>
#include <cstdio>
#include <queue>
#include <random>
#include <string>
#include <vector>

constexpr int ESCAPE = 1 << 16;                 // pseudo-symbol id, outside uint16 range
constexpr double EPS = 1e-12;                    // fold rarer symbols into ESCAPE

struct BitWriter {
    std::vector<uint8_t> buf;
    uint8_t cur = 0; int n = 0; long long bits = 0;
    void put(int b) { cur = (uint8_t)((cur << 1) | (b & 1)); if (++n == 8) { buf.push_back(cur); cur = 0; n = 0; } ++bits; }
    void put_code(const std::vector<bool>& c) { for (bool b : c) put(b); }
    void put_raw16(uint16_t v) { for (int i = 15; i >= 0; --i) put((v >> i) & 1); }
    void flush() { if (n) { buf.push_back((uint8_t)(cur << (8 - n))); cur = 0; n = 0; } }
};

struct BitReader {
    const std::vector<uint8_t>& buf; size_t pos = 0; int n = 0;
    explicit BitReader(const std::vector<uint8_t>& b) : buf(b) {}
    int get() { int b = (buf[pos] >> (7 - n)) & 1; if (++n == 8) { n = 0; ++pos; } return b; }
    uint16_t get_raw16() { uint16_t v = 0; for (int i = 0; i < 16; ++i) v = (uint16_t)((v << 1) | get()); return v; }
};

struct Node { double w; int sym, l, r; };

int main(int argc, char** argv) {
    const double sigma = argc > 1 ? std::stod(argv[1]) : 1.0;
    const double mu    = argc > 2 ? std::stod(argv[2]) : 0.0;
    const long N       = argc > 3 ? std::stol(argv[3]) : 1000000;
    const uint64_t seed = argc > 4 ? std::stoull(argv[4]) : 12345ull;

    // Self-test: every non-NaN pattern must survive half -> float -> half.
    for (uint32_t h = 0; h < (1u << 16); ++h)
        if (!half::is_nan((uint16_t)h) && half::from_float((float)half::to_double((uint16_t)h)) != (uint16_t)h) {
            printf("FP16 round-trip self-test FAILED at pattern %u\n", h); return 1;
        }

    auto P = fp16pmf::compute(mu, sigma);
    const double H = fp16pmf::entropy(P);

    // Build leaves: symbols with P>=EPS, plus one ESCAPE leaf carrying the rest.
    std::vector<Node> nodes;
    auto pq_cmp = [&](int a, int b) { return nodes[a].w > nodes[b].w; };
    std::priority_queue<int, std::vector<int>, decltype(pq_cmp)> pq(pq_cmp);
    double escapeW = 1e-300;                     // tiny floor so ESCAPE always has a codeword
    int realSyms = 0;
    for (uint32_t h = 0; h < (1u << 16); ++h) {
        if (P[h] <= 0.0) continue;
        if (P[h] >= EPS) { nodes.push_back({P[h], (int)h, -1, -1}); pq.push((int)nodes.size() - 1); ++realSyms; }
        else escapeW += P[h];
    }
    nodes.push_back({escapeW, ESCAPE, -1, -1}); pq.push((int)nodes.size() - 1);

    while (pq.size() > 1) {
        int a = pq.top(); pq.pop();
        int b = pq.top(); pq.pop();
        nodes.push_back({nodes[a].w + nodes[b].w, -1, a, b});
        pq.push((int)nodes.size() - 1);
    }
    const int root = pq.top();

    // Assign codewords by DFS.
    std::vector<std::vector<bool>> code(1 << 16);   // by pattern
    std::vector<bool> escCode;
    int maxLen = 0;
    std::vector<std::pair<int, std::vector<bool>>> stack = {{root, {}}};
    while (!stack.empty()) {
        auto [idx, path] = stack.back(); stack.pop_back();
        const Node& nd = nodes[idx];
        if (nd.sym >= 0) {
            maxLen = std::max(maxLen, (int)path.size());
            if (nd.sym == ESCAPE) escCode = path; else code[nd.sym] = path;
        } else {
            auto lp = path; lp.push_back(0); stack.push_back({nd.l, lp});
            auto rp = path; rp.push_back(1); stack.push_back({nd.r, rp});
        }
    }

    // Expected length under the true pmf (the theoretical Huffman performance).
    double L = escapeW * (escCode.size() + 16);
    for (uint32_t h = 0; h < (1u << 16); ++h)
        if (P[h] >= EPS) L += P[h] * code[h].size();

    // Sample a real stream, encode, decode, verify lossless.
    std::mt19937_64 rng(seed);
    std::normal_distribution<double> gauss(mu, sigma);
    std::vector<uint16_t> data((size_t)N);
    for (long i = 0; i < N; ++i) data[i] = half::from_float((float)gauss(rng));

    BitWriter bw; long escapes = 0;
    for (uint16_t s : data) {
        if (!code[s].empty()) { bw.put_code(code[s]); }     // coded symbol (root is internal => len >= 1)
        else { bw.put_code(escCode); bw.put_raw16(s); ++escapes; }  // sub-EPS / unseen -> escape + raw
    }
    bw.flush();

    BitReader br(bw.buf);
    bool lossless = true;
    for (long i = 0; i < N; ++i) {
        int idx = root;
        while (nodes[idx].sym < 0) idx = br.get() ? nodes[idx].r : nodes[idx].l;
        uint16_t out = nodes[idx].sym == ESCAPE ? br.get_raw16() : (uint16_t)nodes[idx].sym;
        if (out != data[i]) { lossless = false; break; }
    }

    const double achieved = (double)bw.bits / (double)N;
    printf("Huffman codec  |  N(mu=%.4g, sigma=%.4g)  |  N=%ld samples\n", mu, sigma, N);
    printf("  alphabet (P>=%.0e)        : %d symbols (+ escape)\n", EPS, realSyms);
    printf("  max codeword length       : %d bits\n", maxLen);
    printf("  escape emissions          : %ld (%.4f%%)\n", escapes, 100.0 * escapes / N);
    printf("  lossless round-trip       : %s\n", lossless ? "PASS" : "FAIL");
    printf("  -------------------------------------------\n");
    printf("  entropy floor   H         : %8.4f bits/sample\n", H);
    printf("  Huffman expected L        : %8.4f bits/sample   (H <= L < H+1: %s)\n",
           L, (H <= L + 1e-9 && L < H + 1.0) ? "ok" : "VIOLATED");
    printf("  achieved (sampled stream) : %8.4f bits/sample\n", achieved);
    printf("  naive storage             : %8.4f bits/sample\n", 16.0);
    printf("  overhead over floor       : %8.4f bits (%.2f%%)\n", achieved - H, 100.0 * (achieved - H) / H);
    return 0;
}
