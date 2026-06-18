// Read a .safetensors file, and for a selection of weight matrices compute
// distribution stats + a histogram, plus the empirical entropy of their bf16
// patterns (ties our codec work to real LLM weights). Emits JSON for the HTML
// dashboard.
//
//   build: g++ -O2 -std=c++17 weights.cpp -o weights
//   run:   ./weights <model.safetensors> [out.json]
//          ./weights <model.safetensors> --list      (just list tensors)
#include "half.h"
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <map>
#include <string>
#include <vector>

// ---- tiny JSON parser (enough for a safetensors header) --------------------
struct JVal {
    enum T { OBJ, ARR, STR, NUM } t = OBJ;
    std::map<std::string, JVal> obj;
    std::vector<JVal> arr;
    std::string str;
    double num = 0;
};
struct JParse {
    const char* p;
    void ws() { while (*p == ' ' || *p == '\n' || *p == '\t' || *p == '\r') ++p; }
    std::string pstr() { std::string s; ++p; while (*p != '"') { if (*p == '\\') ++p; s += *p++; } ++p; return s; }
    JVal val() {
        ws();
        JVal v;
        if (*p == '{') { v.t = JVal::OBJ; ++p; ws(); while (*p != '}') { ws(); std::string k = pstr(); ws(); ++p; /*:*/ v.obj[k] = val(); ws(); if (*p == ',') ++p; ws(); } ++p; }
        else if (*p == '[') { v.t = JVal::ARR; ++p; ws(); while (*p != ']') { v.arr.push_back(val()); ws(); if (*p == ',') ++p; ws(); } ++p; }
        else if (*p == '"') { v.t = JVal::STR; v.str = pstr(); }
        else { v.t = JVal::NUM; char* e; v.num = std::strtod(p, &e); p = e; }
        return v;
    }
};

// ---- dtype decoding --------------------------------------------------------
static int elem_size(const std::string& dt) {
    if (dt == "BF16" || dt == "F16" || dt == "I16") return 2;
    if (dt == "F32" || dt == "I32") return 4;
    if (dt == "F64") return 8;
    if (dt == "I8" || dt == "U8") return 1;
    return 0;
}
static float decode(const uint8_t* b, const std::string& dt) {
    if (dt == "BF16") { uint32_t x = (uint32_t)(b[0] | (b[1] << 8)) << 16; float f; std::memcpy(&f, &x, 4); return f; }
    if (dt == "F16")  { uint16_t h = (uint16_t)(b[0] | (b[1] << 8)); return (float)half::to_double(h); }
    if (dt == "F32")  { float f; std::memcpy(&f, b, 4); return f; }
    return 0.0f;
}
// bf16 pattern, for the empirical-entropy histogram (always 16-bit key).
static uint16_t bf16_pattern(const uint8_t* b, const std::string& dt) {
    if (dt == "BF16" || dt == "F16") return (uint16_t)(b[0] | (b[1] << 8));
    return half::from_float(decode(b, dt));   // re-round F32 weights to bf16-ish key
}

int main(int argc, char** argv) {
    if (argc < 2) { printf("usage: %s <model.safetensors> [out.json|--list]\n", argv[0]); return 1; }
    const std::string path = argv[1];
    const bool listOnly = argc > 2 && std::string(argv[2]) == "--list";
    const std::string out = (argc > 2 && !listOnly) ? argv[2] : "weights.json";

    std::ifstream f(path, std::ios::binary);
    if (!f) { printf("cannot open %s\n", path.c_str()); return 1; }
    uint64_t hlen = 0; f.read((char*)&hlen, 8);
    std::string hdr(hlen, '\0'); f.read(&hdr[0], (std::streamsize)hlen);
    const uint64_t dataStart = 8 + hlen;

    JParse jp{hdr.c_str()};
    JVal root = jp.val();

    // List of (name, dtype, shape, offset, nbytes), skipping __metadata__.
    struct Tensor { std::string name, dtype; std::vector<long> shape; uint64_t off, nbytes; };
    std::vector<Tensor> tensors;
    for (auto& [name, v] : root.obj) {
        if (name == "__metadata__") continue;
        Tensor t; t.name = name; t.dtype = v.obj["dtype"].str;
        for (auto& d : v.obj["shape"].arr) t.shape.push_back((long)d.num);
        const uint64_t s = (uint64_t)v.obj["data_offsets"].arr[0].num;
        const uint64_t e = (uint64_t)v.obj["data_offsets"].arr[1].num;
        t.off = dataStart + s; t.nbytes = e - s;
        tensors.push_back(std::move(t));
    }
    printf("parsed %zu tensors from %s\n", tensors.size(), path.c_str());

    if (listOnly) {
        for (auto& t : tensors) {
            printf("  %-48s %-5s [", t.name.c_str(), t.dtype.c_str());
            for (size_t i = 0; i < t.shape.size(); ++i) printf("%s%ld", i ? "," : "", t.shape[i]);
            printf("]\n");
        }
        return 0;
    }

    // Pick a representative set (first match per substring that exists).
    const std::vector<std::string> want = {
        "embed_tokens.weight",
        "layers.0.self_attn.q_proj.weight", "layers.0.self_attn.k_proj.weight",
        "layers.0.self_attn.v_proj.weight",
        "layers.0.mlp.gate_proj.weight", "layers.0.mlp.down_proj.weight",
        "layers.0.input_layernorm.weight", "layers.23.mlp.down_proj.weight",
    };
    std::vector<const Tensor*> sel;
    for (auto& w : want)
        for (auto& t : tensors)
            if (t.name.find(w) != std::string::npos) { sel.push_back(&t); break; }

    std::ofstream js(out);
    js << "{\n  \"model\": \"" << path << "\",\n  \"tensors\": [\n";
    for (size_t si = 0; si < sel.size(); ++si) {
        const Tensor& t = *sel[si];
        const int es = elem_size(t.dtype);
        const uint64_t n = es ? t.nbytes / es : 0;
        std::vector<uint8_t> buf(t.nbytes);
        f.seekg((std::streamoff)t.off); f.read((char*)buf.data(), (std::streamsize)t.nbytes);

        // Pass 1: sum (for mean), min/max, bf16-pattern histogram.
        const uint64_t cnt = n;
        double sum = 0; float lo = 1e30f, hi = -1e30f;
        std::vector<uint64_t> pat(1 << 16, 0);
        for (uint64_t i = 0; i < n; ++i) {
            const uint8_t* b = buf.data() + i * es;
            float x = decode(b, t.dtype);
            ++pat[bf16_pattern(b, t.dtype)];
            sum += x; lo = std::min(lo, x); hi = std::max(hi, x);
        }
        const double mean = cnt ? sum / cnt : 0.0;

        // Pass 2: central moments about the mean (robust two-pass, buffer is in RAM).
        double m2 = 0, m4 = 0;
        for (uint64_t i = 0; i < n; ++i) {
            const double d = decode(buf.data() + i * es, t.dtype) - mean;
            const double d2 = d * d; m2 += d2; m4 += d2 * d2;
        }
        const double var = cnt ? m2 / cnt : 0.0, sd = std::sqrt(var);
        const double kurt = (cnt && var > 0) ? (m4 / cnt) / (var * var) : 0.0; // raw (Gaussian = 3)

        double bf16H = 0.0;
        for (uint64_t c : pat) if (c) { double p = (double)c / cnt; bf16H -= p * std::log2(p); }

        // Pass 2: value histogram over [lo, hi].
        const int B = 256;
        std::vector<uint64_t> bins(B, 0);
        const double span = (hi > lo) ? (hi - lo) : 1.0;
        for (uint64_t i = 0; i < n; ++i) {
            float x = decode(buf.data() + i * es, t.dtype);
            int k = (int)((x - lo) / span * B); if (k < 0) k = 0; if (k >= B) k = B - 1;
            ++bins[k];
        }

        js << "    {\n";
        js << "      \"name\": \"" << t.name << "\",\n";
        js << "      \"dtype\": \"" << t.dtype << "\",\n";
        js << "      \"shape\": ["; for (size_t i = 0; i < t.shape.size(); ++i) js << (i ? "," : "") << t.shape[i]; js << "],\n";
        js << "      \"count\": " << cnt << ",\n";
        js << "      \"mean\": " << mean << ", \"std\": " << sd << ",\n";
        js << "      \"min\": " << lo << ", \"max\": " << hi << ",\n";
        js << "      \"kurtosis\": " << kurt << ",\n";
        js << "      \"bf16_entropy_bits\": " << bf16H << ",\n";
        js << "      \"hist\": { \"lo\": " << lo << ", \"hi\": " << hi << ", \"bins\": [";
        for (int k = 0; k < B; ++k) js << (k ? "," : "") << bins[k];
        js << "] }\n    }" << (si + 1 < sel.size() ? "," : "") << "\n";

        printf("  %-44s %-5s n=%-9llu mean=%+.4g std=%.4g kurt=%.2f bf16H=%.3f bits\n",
               t.name.c_str(), t.dtype.c_str(), (unsigned long long)cnt, mean, sd, kurt, bf16H);
    }
    js << "  ]\n}\n";
    printf("wrote %s\n", out.c_str());
    return 0;
}
