/*************************************************************************
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * See LICENSE.txt for more information
 *************************************************************************/

#ifndef INSPECTOR_INSPECTOR_PROM_STATS_H_
#define INSPECTOR_INSPECTOR_PROM_STATS_H_

#include <algorithm>
#include <stddef.h>
#include <stdint.h>
#include <utility>
#include <vector>

static inline uint64_t inspectorPromMix64(uint64_t value) {
  value += 0x9e3779b97f4a7c15ULL;
  value = (value ^ (value >> 30)) * 0xbf58476d1ce4e5b9ULL;
  value = (value ^ (value >> 27)) * 0x94d049bb133111ebULL;
  return value ^ (value >> 31);
}

static inline void inspectorPromBoundedSampleUpdate(
    std::vector<uint64_t>& samples,
    size_t capacity,
    uint64_t seen,
    uint64_t sequence,
    size_t messageBytes,
    uint64_t value) {
  if (capacity == 0 || seen == 0) return;
  if (samples.size() < capacity) {
    samples.push_back(value);
    return;
  }
  uint64_t sampleIndex
    = inspectorPromMix64(sequence ^ messageBytes ^ seen) % seen;
  if (sampleIndex < capacity) {
    samples[static_cast<size_t>(sampleIndex)] = value;
  }
}

template <typename Item, typename Slower>
static inline void inspectorPromBoundedTopK(std::vector<Item>& items,
                                            Item item,
                                            size_t capacity,
                                            Slower slower) {
  if (capacity == 0) return;
  items.push_back(std::move(item));
  std::sort(items.begin(), items.end(), slower);
  if (items.size() > capacity) items.resize(capacity);
}

#endif  // INSPECTOR_INSPECTOR_PROM_STATS_H_
