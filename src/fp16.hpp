#pragma once

#include <cstdint>
#include <cstring>

namespace kokoro {

// Portable fp32 -> fp16 (round-to-nearest-even). Avoids needing __fp16.
inline uint16_t fp32_to_fp16(float f) {
  uint32_t x;
  std::memcpy(&x, &f, 4);
  uint32_t sign = (x >> 16) & 0x8000u;
  int32_t  exp  = static_cast<int32_t>((x >> 23) & 0xFFu) - 127 + 15;
  uint32_t mant = x & 0x7FFFFFu;
  uint16_t h;
  if (exp >= 31) {
    h = static_cast<uint16_t>(sign | 0x7C00u | (mant ? 0x200u : 0)); // inf/nan
  } else if (exp <= 0) {
    if (exp < -10) { h = static_cast<uint16_t>(sign); }
    else {
      mant |= 0x800000u;
      uint32_t shift = static_cast<uint32_t>(14 - exp);
      uint32_t hmant = mant >> shift;
      // round
      if ((mant >> (shift - 1)) & 1) hmant += 1;
      h = static_cast<uint16_t>(sign | hmant);
    }
  } else {
    uint32_t hmant = mant >> 13;
    if (mant & 0x1000u) {
      hmant += 1;
      if (hmant & 0x400u) { hmant = 0; exp += 1; }
    }
    if (exp >= 31) h = static_cast<uint16_t>(sign | 0x7C00u);
    else h = static_cast<uint16_t>(sign | (static_cast<uint32_t>(exp) << 10) | hmant);
  }
  return h;
}

inline float fp16_to_fp32(uint16_t h) {
  uint32_t sign = (h >> 15) & 1u;
  uint32_t exp  = (h >> 10) & 0x1Fu;
  uint32_t mant = h & 0x3FFu;
  uint32_t out;
  if (exp == 0) {
    if (mant == 0) out = sign << 31;
    else {
      while (!(mant & 0x400u)) { mant <<= 1; --exp; }
      ++exp; mant &= 0x3FFu;
      out = (sign << 31) | ((exp + 112u) << 23) | (mant << 13);
    }
  } else if (exp == 31) {
    out = (sign << 31) | (0xFFu << 23) | (mant << 13);
  } else {
    out = (sign << 31) | ((exp + 112u) << 23) | (mant << 13);
  }
  float r;
  std::memcpy(&r, &out, 4);
  return r;
}

} // namespace kokoro
