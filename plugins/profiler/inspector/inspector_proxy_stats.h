#ifndef NCCL_INSPECTOR_PROXY_STATS_H_
#define NCCL_INSPECTOR_PROXY_STATS_H_

#include <stdint.h>
#include "inspector_parent_identity.h"

enum inspectorProxyWaitPhase {
  inspectorProxySendGpuWait = 0,
  inspectorProxySendPeerWait = 1,
  inspectorProxySendWait = 2,
  inspectorProxyRecvWait = 3,
  inspectorProxyRecvFlushWait = 4,
  inspectorProxyRecvGpuWait = 5,
  inspectorProxyWaitPhaseCount = 6,
};

struct inspectorProxyStepTimeline {
  bool isSend = false;
  uint64_t startUsecs = 0;
  uint64_t stopUsecs = 0;
  uint64_t states[3] = {0, 0, 0};
};

struct inspectorProxyWaitDurations {
  uint64_t usecs[inspectorProxyWaitPhaseCount] = {0, 0, 0, 0, 0, 0};
  uint32_t validMask = 0;
};

// One exact witness for a phase maximum, never an independently merged set of
// identity fields. Host timestamps use inspectorGetTime(), not GPU pTimer.
struct inspectorProxyPhasePeak {
  uint64_t parentIdentityId;
  uint64_t startUsecs;
  uint64_t stopUsecs;
  uint64_t sequence;
  uint64_t parentType;
  int channel;
  int peer;
  int transferStep;
  bool identityKnown;
};

static inline bool inspectorProxySelectPeak(
    inspectorProxyPhasePeak& target,
    const inspectorProxyPhasePeak& candidate) {
  if (candidate.startUsecs == 0 || candidate.stopUsecs < candidate.startUsecs) return false;
  const uint64_t duration = candidate.stopUsecs - candidate.startUsecs;
  const uint64_t current = target.stopUsecs - target.startUsecs;
  if (target.startUsecs != 0) {
    if (duration < current) return false;
    if (duration == current) {
      // Deterministic tie choice also makes merge independent of dump order.
      const uint64_t left[] = {candidate.startUsecs, candidate.parentType,
        candidate.sequence, uint64_t(candidate.channel), uint64_t(candidate.peer),
        uint64_t(candidate.transferStep), uint64_t(candidate.identityKnown)};
      const uint64_t right[] = {target.startUsecs, target.parentType,
        target.sequence, uint64_t(target.channel), uint64_t(target.peer),
        uint64_t(target.transferStep), uint64_t(target.identityKnown)};
      bool earlier = false;
      for (unsigned i = 0; i < sizeof(left) / sizeof(left[0]); i++) {
        if (left[i] == right[i]) continue;
        earlier = left[i] < right[i];
        break;
      }
      if (!earlier) return false;
    }
  }
  target = candidate;
  return true;
}

// Select the original witness with exactly the existing comparison policy,
// then copy its parent values as one unit. No live parent pointer is retained.
static inline bool inspectorProxySelectPeakWithParent(
    inspectorProxyPhasePeak& target, inspectorParentIdentity& targetParent,
    const inspectorProxyPhasePeak& candidate,
    const inspectorParentIdentity& candidateParent) {
  if (!inspectorProxySelectPeak(target, candidate)) return false;
  targetParent = candidate.parentIdentityId == candidateParent.id
    ? candidateParent : inspectorParentIdentity{};
  return true;
}

static inline bool inspectorProxyOrderedDelta(
    uint64_t start,
    uint64_t stop,
    uint64_t* duration) {
  if (duration == nullptr || start == 0 || stop < start) return false;
  *duration = stop - start;
  return true;
}

static inline inspectorProxyWaitDurations inspectorProxyComputeWaitDurations(
    const inspectorProxyStepTimeline& timeline) {
  inspectorProxyWaitDurations result;
  if (timeline.isSend) {
    const inspectorProxyWaitPhase phases[3] = {
      inspectorProxySendGpuWait,
      inspectorProxySendPeerWait,
      inspectorProxySendWait,
    };
    const uint64_t starts[3] = {
      timeline.states[0], timeline.states[1], timeline.states[2]
    };
    const uint64_t stops[3] = {
      timeline.states[1], timeline.states[2], timeline.stopUsecs
    };
    for (int index = 0; index < 3; index++) {
      uint64_t duration = 0;
      if (inspectorProxyOrderedDelta(starts[index], stops[index], &duration)) {
        result.usecs[phases[index]] = duration;
        result.validMask |= (1u << phases[index]);
      }
    }
  } else {
    const inspectorProxyWaitPhase phases[3] = {
      inspectorProxyRecvWait,
      inspectorProxyRecvFlushWait,
      inspectorProxyRecvGpuWait,
    };
    const uint64_t starts[3] = {
      timeline.states[0], timeline.states[1], timeline.states[2]
    };
    const uint64_t stops[3] = {
      timeline.states[1], timeline.states[2], timeline.stopUsecs
    };
    for (int index = 0; index < 3; index++) {
      uint64_t duration = 0;
      if (inspectorProxyOrderedDelta(starts[index], stops[index], &duration)) {
        result.usecs[phases[index]] = duration;
        result.validMask |= (1u << phases[index]);
      }
    }
  }
  return result;
}

static inline const char* inspectorProxyWaitPhaseName(
    inspectorProxyWaitPhase phase) {
  switch (phase) {
    case inspectorProxySendGpuWait: return "send_gpu_wait";
    case inspectorProxySendPeerWait: return "send_peer_wait";
    case inspectorProxySendWait: return "send_wait";
    case inspectorProxyRecvWait: return "recv_wait";
    case inspectorProxyRecvFlushWait: return "recv_flush_wait";
    case inspectorProxyRecvGpuWait: return "recv_gpu_wait";
    default: return "unknown";
  }
}

#endif
