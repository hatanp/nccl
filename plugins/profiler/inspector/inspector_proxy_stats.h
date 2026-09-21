#ifndef NCCL_INSPECTOR_PROXY_STATS_H_
#define NCCL_INSPECTOR_PROXY_STATS_H_

#include <stdint.h>

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
