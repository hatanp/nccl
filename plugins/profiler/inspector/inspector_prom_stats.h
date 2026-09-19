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

enum inspectorPromSemanticFamily {
  inspectorPromFamilyUnknown = 0,
  inspectorPromFamilyDp,
  inspectorPromFamilyEdp,
  inspectorPromFamilyEp,
  inspectorPromFamilyGlobal,
  inspectorPromFamilyPpLocal,
  inspectorPromFamilyPpCrossNode
};

struct inspectorPromTopologySizes {
  int world = 0;
  int dp = 0;
  int edp = 0;
  int ep = 0;
  int pp = 0;
};

static inline inspectorPromSemanticFamily inspectorPromClassifyFamily(
    bool isP2p,
    int nranks,
    int nnodes,
    const inspectorPromTopologySizes& sizes) {
  if (isP2p) {
    return nnodes > 1 ? inspectorPromFamilyPpCrossNode
                      : inspectorPromFamilyPpLocal;
  }
  if (sizes.world > 0 && nranks == sizes.world) {
    return inspectorPromFamilyGlobal;
  }
  if (sizes.dp > 0 && nranks == sizes.dp) {
    return inspectorPromFamilyDp;
  }
  if (sizes.edp > 0 && nranks == sizes.edp) {
    return inspectorPromFamilyEdp;
  }
  if (sizes.ep > 0 && nranks == sizes.ep) {
    return inspectorPromFamilyEp;
  }
  return inspectorPromFamilyUnknown;
}

static inline const char* inspectorPromSemanticFamilyName(
    inspectorPromSemanticFamily family) {
  switch (family) {
    case inspectorPromFamilyDp: return "dp";
    case inspectorPromFamilyEdp: return "edp";
    case inspectorPromFamilyEp: return "ep";
    case inspectorPromFamilyGlobal: return "global";
    case inspectorPromFamilyPpLocal: return "pp_local";
    case inspectorPromFamilyPpCrossNode: return "pp_cross_node";
    default: return "unknown";
  }
}

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
