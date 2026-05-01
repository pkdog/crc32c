// Copyright 2025 The CRC32C Authors. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file. See the AUTHORS file for names of contributors.

#ifndef CRC32C_CRC32C_SSE42_CLMUL_H_
#define CRC32C_CRC32C_SSE42_CLMUL_H_

// X86-specific code that uses PCLMULQDQ to accelerate CRC32C stripe folding.

#include <cstddef>
#include <cstdint>

#include "crc32c/crc32c_config.h"

// The PCLMUL-accelerated implementation depends on SSE4.2 and is only
// enabled for 64-bit builds.
#if HAVE_SSE42 && HAVE_PCLMUL && (defined(_M_X64) || defined(__x86_64__))

namespace crc32c {

// PCLMUL+SSE4.2-accelerated implementation in crc32c_sse42_clmul.cc
uint32_t ExtendSse42Clmul(uint32_t crc, const uint8_t* data, size_t count);

}  // namespace crc32c

#endif  // HAVE_SSE42 && HAVE_PCLMUL && (defined(_M_X64) || defined(__x86_64__))

#endif  // CRC32C_CRC32C_SSE42_CLMUL_H_
