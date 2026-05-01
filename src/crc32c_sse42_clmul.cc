// Copyright 2025 The CRC32C Authors. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file. See the AUTHORS file for names of contributors.

#include "./crc32c_sse42_clmul.h"

// In a separate source file to allow this PCLMUL-accelerated CRC32C function
// to be compiled with appropriate compiler flags (-msse4.2 -mpclmul).

// This implementation uses PCLMULQDQ (carry-less multiplication) to fold
// parallel CRC stripes, replacing the skip-table approach. Based on the
// techniques in:
//   "Fast CRC Computation for Generic Polynomials Using PCLMULQDQ Instruction"
//   V. Gopal, E. Ozturk, et al., 2009

#include <cstddef>
#include <cstdint>

#include "./crc32c_internal.h"
#include "./crc32c_prefetch.h"
#include "./crc32c_read_le.h"
#include "./crc32c_round_up.h"
#include "crc32c/crc32c_config.h"

#if HAVE_SSE42 && HAVE_PCLMUL && (defined(_M_X64) || defined(__x86_64__))

#if defined(_MSC_VER)
#include <intrin.h>
#include <wmmintrin.h>
#else  // !defined(_MSC_VER)
#include <nmmintrin.h>
#include <wmmintrin.h>
#endif  // defined(_MSC_VER)

namespace crc32c {

namespace {

// ---------------------------------------------------------------------------
// PCLMULQDQ folding helpers
// ---------------------------------------------------------------------------

// Fold two CRC stripes (crc0, crc1) using PCLMULQDQ with constants K1, K2.
// Returns: crc32q(0, pclmul(crc0, K1)) ^ crc2 ^ crc32q(0, pclmul(crc1, K2))
// crc2 is consumed (XORed in, not used as a CRC accumulator).
inline uint64_t FoldTwoStripes(uint64_t crc0, uint64_t crc1, uint64_t crc2,
                               uint64_t K1, uint64_t K2) {
  __m128i xmm_crc0 = _mm_cvtsi64_si128(static_cast<int64_t>(crc0));
  __m128i xmm_crc1 = _mm_cvtsi64_si128(static_cast<int64_t>(crc1));
  __m128i xmm_K1 = _mm_cvtsi64_si128(static_cast<int64_t>(K1));
  __m128i xmm_K2 = _mm_cvtsi64_si128(static_cast<int64_t>(K2));

  // pclmulqdq xmm, xmm with imm8=0x00: multiply low 64 bits
  __m128i xmm_t0 = _mm_clmulepi64_si128(xmm_crc0, xmm_K1, 0x00);
  __m128i xmm_t1 = _mm_clmulepi64_si128(xmm_crc1, xmm_K2, 0x00);

  uint64_t t0 = static_cast<uint64_t>(_mm_cvtsi128_si64(xmm_t0));
  uint64_t t1 = static_cast<uint64_t>(_mm_cvtsi128_si64(xmm_t1));

  // Fold: result = crc32q(0, t0) ^ crc2 ^ crc32q(0, t1)
  // All three terms are XORed together; crc2 is NOT used as an accumulator.
  uint64_t folded_crc0 = _mm_crc32_u64(0, t0);
  uint64_t folded_crc1 = _mm_crc32_u64(0, t1);
  return folded_crc0 ^ crc2 ^ folded_crc1;
}

// ---------------------------------------------------------------------------
// Block-level functions.
// Each processes exactly the specified number of bytes using 3 parallel
// CRC stripes folded with PCLMULQDQ.
// ---------------------------------------------------------------------------

// Prefetch horizon in bytes: how far ahead of the current read pointer we
// request prefetch. 256 matches the SSE4.2 skip-table path.
constexpr ptrdiff_t kClmulPrefetchHorizon = 256;

// Process exactly 256 bytes in 3 stripes of (11, 10, 10) quadwords.
// Stripe layout: [0..88) [88..168) [176..256)
inline uint32_t ExtendSse42Clmul256(const uint8_t* data, uint32_t crc) {
  uint64_t crc0 = static_cast<uint64_t>(crc);
  uint64_t crc1 = 0;
  uint64_t crc2 = 0;

  // Stripe offsets: stripe0 at data+0, stripe1 at data+88, stripe2 at data+176
  const uint8_t* s0 = data;
  const uint8_t* s1 = data + 88;
  const uint8_t* s2 = data + 176;

  // 11 quadwords for stripe0, 10 for stripes 1 and 2
  for (int i = 0; i < 10; ++i) {
    crc0 = _mm_crc32_u64(crc0, ReadUint64LE(s0)); s0 += 8;
    crc1 = _mm_crc32_u64(crc1, ReadUint64LE(s1)); s1 += 8;
    crc2 = _mm_crc32_u64(crc2, ReadUint64LE(s2)); s2 += 8;
  }
  // Last quadword: stripe0 gets an 11th, stripe1 absorbs the "crc2" role
  crc0 = _mm_crc32_u64(crc0, ReadUint64LE(s0));
  crc1 = _mm_crc32_u64(crc1, ReadUint64LE(s1));

  // Magic constants for 256-byte folding
  constexpr uint64_t K1 = 0x1b3d8f29ULL;
  constexpr uint64_t K2 = 0x39d3b296ULL;
  return static_cast<uint32_t>(FoldTwoStripes(crc0, crc1, crc2, K1, K2));
}

// Process exactly 1024 bytes: 5 loops of 3×(8×8bytes) + 3×(3×8bytes) tail
// Stripe layout: [0..344) [344..680) [680..1024)
inline uint32_t ExtendSse42Clmul1024(const uint8_t* data, uint32_t crc) {
  uint64_t crc0 = static_cast<uint64_t>(crc);
  uint64_t crc1 = 0;
  uint64_t crc2 = 0;

  const uint8_t* p = data;
  const uint8_t* s1 = data + 344;
  const uint8_t* s2 = data + 680;

  // 5 loops, each processing 64 bytes per stripe (8 quadwords)
  for (int loop = 0; loop < 5; ++loop) {
    RequestPrefetch(p + kClmulPrefetchHorizon);
    RequestPrefetch(s1 + kClmulPrefetchHorizon);
    RequestPrefetch(s2 + kClmulPrefetchHorizon);

    for (int i = 0; i < 8; ++i) {
      crc0 = _mm_crc32_u64(crc0, ReadUint64LE(p));   p += 8;
      crc1 = _mm_crc32_u64(crc1, ReadUint64LE(s1));  s1 += 8;
      crc2 = _mm_crc32_u64(crc2, ReadUint64LE(s2));  s2 += 8;
    }
  }

  // Tail: stripe0 gets 3 quads, stripe1 gets 2 quads, stripe2 gets 3 quads
  // After the loop, total per stripe: stripe0=344B, stripe1=336B, stripe2=344B
  // Total = 344 + 336 + 344 = 1024 bytes
  crc0 = _mm_crc32_u64(crc0, ReadUint64LE(p));   p += 8;
  crc1 = _mm_crc32_u64(crc1, ReadUint64LE(s1));  s1 += 8;
  crc2 = _mm_crc32_u64(crc2, ReadUint64LE(s2));  s2 += 8;

  crc0 = _mm_crc32_u64(crc0, ReadUint64LE(p));   p += 8;
  crc1 = _mm_crc32_u64(crc1, ReadUint64LE(s1));  s1 += 8;
  crc2 = _mm_crc32_u64(crc2, ReadUint64LE(s2));  s2 += 8;

  crc0 = _mm_crc32_u64(crc0, ReadUint64LE(p));
  crc2 = _mm_crc32_u64(crc2, ReadUint64LE(s2));  s2 += 8;

  // Magic constants for 1024-byte folding
  constexpr uint64_t K1 = 0xe417f38aULL;
  constexpr uint64_t K2 = 0x8f158014ULL;
  return static_cast<uint32_t>(FoldTwoStripes(crc0, crc1, crc2, K1, K2));
}

// Process exactly 3072 bytes: 16 loops of 3×(8×8bytes)
// Stripe layout: [0..1024) [1024..2048) [2048..3072)
inline uint32_t ExtendSse42Clmul3072(const uint8_t* data, uint32_t crc) {
  uint64_t crc0 = static_cast<uint64_t>(crc);
  uint64_t crc1 = 0;
  uint64_t crc2 = 0;

  const uint8_t* p = data;
  const uint8_t* s1 = data + 1024;
  const uint8_t* s2 = data + 2048;

  // 16 loops, each processing 64 bytes per stripe (8 quadwords)
  for (int loop = 0; loop < 16; ++loop) {
    RequestPrefetch(p + kClmulPrefetchHorizon);
    RequestPrefetch(s1 + kClmulPrefetchHorizon);
    RequestPrefetch(s2 + kClmulPrefetchHorizon);

    for (int i = 0; i < 8; ++i) {
      crc0 = _mm_crc32_u64(crc0, ReadUint64LE(p));   p += 8;
      crc1 = _mm_crc32_u64(crc1, ReadUint64LE(s1));  s1 += 8;
      crc2 = _mm_crc32_u64(crc2, ReadUint64LE(s2));  s2 += 8;
    }
  }

  // Magic constants for 3072-byte folding
  constexpr uint64_t K1 = 0xa51b6135ULL;
  constexpr uint64_t K2 = 0x170076faULL;
  return static_cast<uint32_t>(FoldTwoStripes(crc0, crc1, crc2, K1, K2));
}

// Process exactly 4032 bytes: 21 loops of 3×(8×8bytes)
// Stripe layout: [0..1344) [1344..2688) [2688..4032)
// Constant computed by tools/compute_folding_constants.cc
inline uint32_t ExtendSse42Clmul4032(const uint8_t* data, uint32_t crc) {
  uint64_t crc0 = static_cast<uint64_t>(crc);
  uint64_t crc1 = 0;
  uint64_t crc2 = 0;

  const uint8_t* p = data;
  const uint8_t* s1 = data + 1344;
  const uint8_t* s2 = data + 2688;

  // 21 loops, each processing 64 bytes per stripe (8 quadwords)
  for (int loop = 0; loop < 21; ++loop) {
    RequestPrefetch(p + kClmulPrefetchHorizon);
    RequestPrefetch(s1 + kClmulPrefetchHorizon);
    RequestPrefetch(s2 + kClmulPrefetchHorizon);

    for (int i = 0; i < 8; ++i) {
      crc0 = _mm_crc32_u64(crc0, ReadUint64LE(p));   p += 8;
      crc1 = _mm_crc32_u64(crc1, ReadUint64LE(s1));  s1 += 8;
      crc2 = _mm_crc32_u64(crc2, ReadUint64LE(s2));  s2 += 8;
    }
  }

  // Magic constants for 4032-byte folding
  constexpr uint64_t K1 = 0x889774e1ULL;
  constexpr uint64_t K2 = 0xc9c8b782ULL;
  return static_cast<uint32_t>(FoldTwoStripes(crc0, crc1, crc2, K1, K2));
}

// Process exactly 16128 bytes: 84 loops of 3×(8×8bytes)
// Stripe layout: [0..5376) [5376..10752) [10752..16128)
// Constant computed by tools/compute_folding_constants.cc
inline uint32_t ExtendSse42Clmul16128(const uint8_t* data, uint32_t crc) {
  uint64_t crc0 = static_cast<uint64_t>(crc);
  uint64_t crc1 = 0;
  uint64_t crc2 = 0;

  const uint8_t* p = data;
  const uint8_t* s1 = data + 5376;
  const uint8_t* s2 = data + 10752;

  // 84 loops, each processing 64 bytes per stripe (8 quadwords)
  for (int loop = 0; loop < 84; ++loop) {
    RequestPrefetch(p + kClmulPrefetchHorizon);
    RequestPrefetch(s1 + kClmulPrefetchHorizon);
    RequestPrefetch(s2 + kClmulPrefetchHorizon);

    for (int i = 0; i < 8; ++i) {
      crc0 = _mm_crc32_u64(crc0, ReadUint64LE(p));   p += 8;
      crc1 = _mm_crc32_u64(crc1, ReadUint64LE(s1));  s1 += 8;
      crc2 = _mm_crc32_u64(crc2, ReadUint64LE(s2));  s2 += 8;
    }
  }

  // Magic constants for 16128-byte folding
  constexpr uint64_t K1 = 0x57e82916ULL;
  constexpr uint64_t K2 = 0x0e0126b2ULL;
  return static_cast<uint32_t>(FoldTwoStripes(crc0, crc1, crc2, K1, K2));
}

}  // namespace

// ---------------------------------------------------------------------------
// Public entry point
// ---------------------------------------------------------------------------

uint32_t ExtendSse42Clmul(uint32_t crc, const uint8_t* data, size_t count) {
  // Invert bits at entry (match the CRC32C convention used by ExtendPortable
  // and ExtendSse42).
  uint32_t c = crc ^ kCRC32Xor;

  const uint8_t* p = data;
  intptr_t length = static_cast<intptr_t>(count);

  // Small input: fall through to the byte/quadword loop below
  if (length < 8) {
    while (length-- > 0) {
      c = _mm_crc32_u8(c, *p++);
    }
    return c ^ kCRC32Xor;
  }

  // Align to 8-byte boundary
  intptr_t alignment = reinterpret_cast<intptr_t>(p) & 0x7;
  intptr_t leading = (8 - alignment) & 0x7;
  length -= leading;
  while (leading-- > 0) {
    c = _mm_crc32_u8(c, *p++);
  }

  // Process large blocks with PCLMUL folding (largest first).
  // Block sizes are aligned with the SSE4.2 skip-table hierarchy
  // (16320/4080/1008) to keep per-byte folding overhead comparable.
  while (length >= 16128) {
    c = ExtendSse42Clmul16128(p, c);
    p += 16128;
    length -= 16128;
  }
  while (length >= 4032) {
    c = ExtendSse42Clmul4032(p, c);
    p += 4032;
    length -= 4032;
  }
  while (length >= 3072) {
    c = ExtendSse42Clmul3072(p, c);
    p += 3072;
    length -= 3072;
  }
  while (length >= 1024) {
    c = ExtendSse42Clmul1024(p, c);
    p += 1024;
    length -= 1024;
  }
  while (length >= 256) {
    c = ExtendSse42Clmul256(p, c);
    p += 256;
    length -= 256;
  }

  // Process remaining aligned quadwords
  while (length >= 8) {
    c = static_cast<uint32_t>(
        _mm_crc32_u64(static_cast<uint64_t>(c), ReadUint64LE(p)));
    p += 8;
    length -= 8;
  }

  // Process trailing bytes
  while (length-- > 0) {
    c = _mm_crc32_u8(c, *p++);
  }

  return c ^ kCRC32Xor;
}

}  // namespace crc32c

#endif  // HAVE_SSE42 && HAVE_PCLMUL && (defined(_M_X64) || defined(__x86_64__))
