#pragma once
#include <cstdint>
#include <cmath>

// Minimal IEEE-754 half-precision (fp16) helpers operating on raw uint16_t bit
// patterns. We use double for the decoded value so downstream probability math
// has headroom.
namespace half {

inline bool is_nan(uint16_t h) { return ((h >> 10) & 0x1F) == 0x1F && (h & 0x3FF) != 0; }
inline bool is_inf(uint16_t h) { return ((h >> 10) & 0x1F) == 0x1F && (h & 0x3FF) == 0; }
inline int  sign_bit(uint16_t h) { return (h >> 15) & 1; }
inline int  exponent(uint16_t h) { return (h >> 10) & 0x1F; }
inline int  mantissa(uint16_t h) { return h & 0x3FF; }

// Decode a finite or infinite pattern to its real value. (NaN -> NaN.)
inline double to_double(uint16_t h) {
    const int s = sign_bit(h), e = exponent(h), m = mantissa(h);
    const double sign = s ? -1.0 : 1.0;
    if (e == 0)            return sign * std::ldexp((double)m, -24);          // zero / subnormal
    if (e == 0x1F)         return m ? NAN : sign * INFINITY;                  // inf / nan
    return sign * std::ldexp((double)(m | 0x400), e - 25);                    // normal
}

constexpr uint16_t POS_INF = 0x7C00;
constexpr uint16_t NEG_INF = 0xFC00;
constexpr uint16_t POS_ZERO = 0x0000;
constexpr uint16_t NEG_ZERO = 0x8000;

// Round a float to the nearest fp16 (ties-to-even), matching the pmf's rounding.
inline uint16_t from_float(float f) {
    uint32_t x;
    static_assert(sizeof(x) == sizeof(f), "");
    __builtin_memcpy(&x, &f, sizeof(x));
    const uint32_t sign = (x >> 16) & 0x8000u;
    const uint32_t aexp = (x >> 23) & 0xFF;
    uint32_t mant = x & 0x7FFFFFu;

    if (aexp == 0xFF) return (uint16_t)(sign | (mant ? 0x7E00 : 0x7C00)); // nan / inf
    const int e = (int)aexp - 127 + 15;                                  // target half exponent
    if (e >= 0x1F) return (uint16_t)(sign | 0x7C00);                     // overflow -> inf
    if (e > 0) {                                                         // normal
        const uint32_t hm = mant >> 13, rem = mant & 0x1FFF;
        uint16_t h = (uint16_t)(sign | (e << 10) | hm);
        if (rem > 0x1000 || (rem == 0x1000 && (hm & 1))) ++h;           // ties-to-even (carry ok)
        return h;
    }
    if (e < -10) return (uint16_t)sign;                                  // underflow -> signed zero
    // subnormal: hm = round(significand24 / 2^shift), ties-to-even.
    const uint32_t sig = mant | 0x800000u;
    const int shift = 14 - e;                                            // 14..24
    uint32_t hm = sig >> shift;
    const uint32_t rem = sig & ((1u << shift) - 1), half = 1u << (shift - 1);
    if (rem > half || (rem == half && (hm & 1))) ++hm;                   // carry into exp=1 is correct
    return (uint16_t)(sign | hm);
}

} // namespace half
