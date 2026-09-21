/*************************************************************************
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * See LICENSE.txt for more license information
 *************************************************************************/

#include "inspector_prom.h"
#include "inspector.h"
#include "inspector_cudawrap.h"
#include "inspector_prom_stats.h"
#include "inspector_proxy_pool.h"
#include "inspector_proxy_stats.h"
#include "inspector_ring.h"
#include "profiler.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <inttypes.h>
#include <algorithm>
#include <atomic>
#include <map>
#include <limits>
#include <mutex>
#include <set>
#include <string>
#include <vector>
#include <cmath>
#include <unistd.h>
#include <sys/stat.h>
#include <errno.h>
#include <time.h>
#include <cuda_runtime.h>

// External references from inspector.cc
extern struct inspectorState g_state;
extern inspectorResult_t inspectorCommInfoListFinalize(struct inspectorCommInfoList* commList);
extern const char* inspectorTimingSourceToString(inspectorTimingSource_t timingSource);

extern const char* ncclFuncToString(ncclFunc_t fn);

struct inspectorPromSlowEvent {
  uint64_t sequence = 0;
  uint64_t execTimeUsecs = 0;
  uint64_t startTimestampUsecs = 0;
  uint64_t stopTimestampUsecs = 0;
  int peer = -1;
};

struct inspectorPromBucketAgg {
  uint64_t count = 0;
  double execTimeSum = 0.0;
  uint64_t execTimeMin = std::numeric_limits<uint64_t>::max();
  uint64_t execTimeMax = 0;
  std::vector<uint64_t> execTimes;
  std::vector<inspectorPromSlowEvent> slowest;
  double algoLogSum = 0.0;
  uint64_t algoCount = 0;
  double busLogSum = 0.0;
  uint64_t busCount = 0;
};

struct inspectorPromGlobalSlowEvent {
  std::string labels;
  inspectorPromSlowEvent event;
};

struct inspectorPromCollBucketKey {
  int nranks;
  int nnodes;
  ncclFunc_t func;
  size_t msgSizeRangeBytes;
  size_t msgSizeBytes;
  int rank;
  std::string commId;
  std::string commName;
  std::string algoProto;
  std::string timingSource;

  bool operator<(const inspectorPromCollBucketKey& other) const {
    if (nranks != other.nranks) return nranks < other.nranks;
    if (nnodes != other.nnodes) return nnodes < other.nnodes;
    if (func != other.func) return func < other.func;
    if (msgSizeRangeBytes != other.msgSizeRangeBytes) {
      return msgSizeRangeBytes < other.msgSizeRangeBytes;
    }
    if (msgSizeBytes != other.msgSizeBytes) return msgSizeBytes < other.msgSizeBytes;
    if (rank != other.rank) return rank < other.rank;
    if (commId != other.commId) return commId < other.commId;
    if (commName != other.commName) return commName < other.commName;
    if (algoProto != other.algoProto) return algoProto < other.algoProto;
    return timingSource < other.timingSource;
  }
};

struct inspectorPromP2pBucketKey {
  int nranks;
  int nnodes;
  ncclFunc_t func;
  size_t msgSizeRangeBytes;
  size_t msgSizeBytes;
  int rank;
  int peer;
  std::string commId;
  std::string commName;
  std::string timingSource;

  bool operator<(const inspectorPromP2pBucketKey& other) const {
    if (nranks != other.nranks) return nranks < other.nranks;
    if (nnodes != other.nnodes) return nnodes < other.nnodes;
    if (func != other.func) return func < other.func;
    if (msgSizeRangeBytes != other.msgSizeRangeBytes) {
      return msgSizeRangeBytes < other.msgSizeRangeBytes;
    }
    if (msgSizeBytes != other.msgSizeBytes) return msgSizeBytes < other.msgSizeBytes;
    if (rank != other.rank) return rank < other.rank;
    if (peer != other.peer) return peer < other.peer;
    if (commId != other.commId) return commId < other.commId;
    if (commName != other.commName) return commName < other.commName;
    return timingSource < other.timingSource;
  }
};

struct inspectorPromStepFamilyKey {
  int64_t step;
  inspectorPromSemanticFamily family;
  ncclFunc_t func;

  bool operator<(const inspectorPromStepFamilyKey& other) const {
    if (step != other.step) return step < other.step;
    if (family != other.family) return family < other.family;
    return func < other.func;
  }
};

struct inspectorPromStepFamilyAgg {
  uint64_t count = 0;
  double execTimeSum = 0.0;
  uint64_t execTimeMax = 0;
  uint64_t firstStartTimestampUsecs = std::numeric_limits<uint64_t>::max();
  uint64_t lastStopTimestampUsecs = 0;
};

struct inspectorPromStepP2pEvent {
  ncclFunc_t func;
  uint64_t sequence;
  size_t messageSizeBytes;
  uint64_t commHash;
  int commRank;
  int peer;
  int nranks;
  int nnodes;
  uint64_t startTimestampUsecs;
  uint64_t stopTimestampUsecs;
  uint64_t execTimeUsecs;
  inspectorTimingSource_t timingSource;
};

struct inspectorPromStepProxyKey {
  int64_t step;
  inspectorPromSemanticFamily family;
  ncclFunc_t func;
  size_t messageSizeBytes;
  bool isSend;
  std::string commId;
  std::string commName;
  int commRank;
  int nranks;
  int nnodes;

  bool operator<(const inspectorPromStepProxyKey& other) const {
    if (step != other.step) return step < other.step;
    if (family != other.family) return family < other.family;
    if (func != other.func) return func < other.func;
    if (messageSizeBytes != other.messageSizeBytes) {
      return messageSizeBytes < other.messageSizeBytes;
    }
    if (isSend != other.isSend) return isSend < other.isSend;
    if (commId != other.commId) return commId < other.commId;
    if (commName != other.commName) return commName < other.commName;
    if (commRank != other.commRank) return commRank < other.commRank;
    if (nranks != other.nranks) return nranks < other.nranks;
    return nnodes < other.nnodes;
  }
};

struct inspectorPromStepProxyAgg {
  uint64_t count = 0;
  uint64_t transferBytes = 0;
  uint64_t unknownTransferSizes = 0;
  uint64_t phaseCount[inspectorProxyWaitPhaseCount] = {0, 0, 0, 0, 0, 0};
  uint64_t phaseSumUsecs[inspectorProxyWaitPhaseCount] = {0, 0, 0, 0, 0, 0};
  uint64_t phaseMaxUsecs[inspectorProxyWaitPhaseCount] = {0, 0, 0, 0, 0, 0};
  uint64_t missingTransitions = 0;
};

using inspectorPromCollBucketMap = std::map<inspectorPromCollBucketKey, inspectorPromBucketAgg>;
using inspectorPromP2pBucketMap = std::map<inspectorPromP2pBucketKey, inspectorPromBucketAgg>;

struct inspectorPromDevice {
  std::string deviceUuidStr;
  std::string nodeName;
  std::string gpuName;
  FILE* file = nullptr;
  inspectorPromCollBucketMap collBuckets;
  inspectorPromP2pBucketMap p2pBuckets;
  std::vector<inspectorPromGlobalSlowEvent> collSlowest;
  std::vector<inspectorPromGlobalSlowEvent> p2pSlowest;
  std::map<inspectorPromStepFamilyKey, inspectorPromStepFamilyAgg> stepFamilies;
  std::map<int64_t, std::vector<inspectorPromStepP2pEvent>> stepP2pEvents;
  std::map<int64_t, uint64_t> stepP2pDropped;
  std::map<inspectorPromStepProxyKey, inspectorPromStepProxyAgg> stepProxy;
  uint64_t collOverwritten = 0;
  uint64_t p2pOverwritten = 0;
  bool hasData = false;
};
static const int kInspectorPromFormatMinor = 5;
static const size_t kInspectorPromPercentileSampleCapacity = 256;
static const size_t kInspectorPromGlobalSlowEventCapacity = 4;
static std::mutex gInspectorPromStreamingMutex;
static std::map<std::string, inspectorPromDevice> gInspectorPromStreamingDevices;
static std::map<inspectorPromStepProxyKey,
                inspectorPromStepProxyAgg> gInspectorPromDetachedProxy;
static std::atomic<int64_t> gInspectorPromCurrentStep {-1};
static std::set<int64_t> gInspectorPromRetainedSteps;

static void inspectorPromInitDevice(inspectorPromDevice& device,
                                    const inspectorCommInfo* commInfo);

static bool inspectorPromStepEnabled() {
  static const bool enabled = []() {
    const char* value = getenv("NCCL_INSPECTOR_STEP_ENABLE");
    return value != nullptr && strcmp(value, "1") == 0;
  }();
  return enabled;
}

static int inspectorPromEnvSize(const char* name) {
  const char* value = getenv(name);
  if (value == nullptr || value[0] == '\0') return 0;
  char* end = nullptr;
  long parsed = strtol(value, &end, 10);
  if (end == value || *end != '\0' || parsed <= 0
      || parsed > std::numeric_limits<int>::max()) {
    return 0;
  }
  return static_cast<int>(parsed);
}

static inspectorPromTopologySizes inspectorPromGetTopologySizes() {
  static const inspectorPromTopologySizes sizes = []() {
    inspectorPromTopologySizes configured;
    configured.world = inspectorPromEnvSize("NCCL_INSPECTOR_WORLD_SIZE");
    configured.dp = inspectorPromEnvSize("NCCL_INSPECTOR_DP_SIZE");
    configured.edp = inspectorPromEnvSize("NCCL_INSPECTOR_EDP_SIZE");
    configured.ep = inspectorPromEnvSize("NCCL_INSPECTOR_EP_SIZE");
    configured.pp = inspectorPromEnvSize("NCCL_INSPECTOR_PP_SIZE");
    return configured;
  }();
  return sizes;
}

static void inspectorPromOperationEnvelopeUsecs(
    const inspectorCompletedOpInfo& op,
    uint64_t* startUsecs,
    uint64_t* stopUsecs) {
  *startUsecs = std::numeric_limits<uint64_t>::max();
  *stopUsecs = 0;
  if (op.timingSource == inspectorTimingSourceKernelGpu
      || op.timingSource == inspectorTimingSourceKernelCpu) {
    uint32_t channels = std::min(op.evtTrk.nChannels,
                                 static_cast<uint32_t>(MAX_CHANNELS));
    for (uint32_t channel = 0; channel < channels; channel++) {
      const inspectorEventTraceInfo* trace
        = op.evtTrk.kernelCh[channel].evntTrace;
      uint64_t start = trace[NCCL_INSP_EVT_TRK_KERNEL_START].ts;
      uint64_t stop = trace[NCCL_INSP_EVT_TRK_KERNEL_STOP].ts;
      if (start > 0) *startUsecs = std::min(*startUsecs, start);
      if (stop > 0) *stopUsecs = std::max(*stopUsecs, stop);
    }
  }
  if (*startUsecs == std::numeric_limits<uint64_t>::max()
      || *stopUsecs == 0) {
    *startUsecs = op.evtTrk.evntTrace[NCCL_INSP_EVT_TRK_OP_START].ts;
    *stopUsecs = op.evtTrk.evntTrace[NCCL_INSP_EVT_TRK_OP_STOP].ts;
  }
}

static size_t inspectorPromStepCapacity() {
  static const size_t capacity = []() {
    int configured = inspectorPromEnvSize("NCCL_INSPECTOR_STEP_CAPACITY");
    return configured > 0 ? static_cast<size_t>(configured) : size_t{128};
  }();
  return capacity;
}

static size_t inspectorPromStepP2pCapacity() {
  static const size_t capacity = []() {
    int configured = inspectorPromEnvSize("NCCL_INSPECTOR_STEP_P2P_CAPACITY");
    return configured > 0 ? static_cast<size_t>(configured) : size_t{1024};
  }();
  return capacity;
}

extern "C" __attribute__((visibility("default")))
int ncclInspectorStepBegin(int64_t step, uint64_t wallTimeNs) {
  (void)wallTimeNs;
  if (!inspectorPromStepEnabled()) return 0;
  std::lock_guard<std::mutex> lock(gInspectorPromStreamingMutex);
  if (gInspectorPromRetainedSteps.count(step) == 0
      && gInspectorPromRetainedSteps.size() >= inspectorPromStepCapacity()) {
    gInspectorPromCurrentStep.store(-1, std::memory_order_relaxed);
    return 0;
  }
  gInspectorPromRetainedSteps.insert(step);
  gInspectorPromCurrentStep.store(step, std::memory_order_relaxed);
  return 0;
}

extern "C" __attribute__((visibility("default")))
int ncclInspectorStepEnd(int64_t step, uint64_t wallTimeNs) {
  (void)wallTimeNs;
  if (!inspectorPromStepEnabled()) return 0;
  std::lock_guard<std::mutex> lock(gInspectorPromStreamingMutex);
  if (gInspectorPromCurrentStep.load(std::memory_order_relaxed) == step) {
    gInspectorPromCurrentStep.store(-1, std::memory_order_relaxed);
  }
  return 0;
}

static void inspectorPromStepUpdate(inspectorPromDevice& device,
                                    const inspectorCommInfo* commInfo,
                                    const inspectorCompletedOpInfo& op) {
  int64_t currentStep = gInspectorPromCurrentStep.load(std::memory_order_relaxed);
  if (!inspectorPromStepEnabled() || currentStep < 0) return;
  inspectorPromTopologySizes sizes = inspectorPromGetTopologySizes();
  inspectorPromSemanticFamily family = inspectorPromClassifyFamily(
    op.isP2p, commInfo->nranks, commInfo->nnodes, sizes);
  if (family == inspectorPromFamilyUnknown) return;
  inspectorPromStepFamilyKey key {currentStep, family, op.func};
  inspectorPromStepFamilyAgg& agg = device.stepFamilies[key];
  uint64_t operationStartUsecs;
  uint64_t operationStopUsecs;
  inspectorPromOperationEnvelopeUsecs(
    op, &operationStartUsecs, &operationStopUsecs);
  agg.count++;
  agg.execTimeSum += static_cast<double>(op.execTimeUsecs);
  agg.execTimeMax = std::max(agg.execTimeMax, op.execTimeUsecs);
  agg.firstStartTimestampUsecs = std::min(
    agg.firstStartTimestampUsecs, operationStartUsecs);
  agg.lastStopTimestampUsecs = std::max(
    agg.lastStopTimestampUsecs, operationStopUsecs);
  if (op.isP2p && family == inspectorPromFamilyPpCrossNode) {
    std::vector<inspectorPromStepP2pEvent>& events
      = device.stepP2pEvents[currentStep];
    if (events.size() < inspectorPromStepP2pCapacity()) {
      events.push_back(inspectorPromStepP2pEvent {
        op.func,
        op.sn,
        op.msgSizeBytes,
        commInfo->commHash,
        commInfo->rank,
        op.peer,
        commInfo->nranks,
        commInfo->nnodes,
        operationStartUsecs,
        operationStopUsecs,
        op.execTimeUsecs,
        op.timingSource
      });
    } else {
      device.stepP2pDropped[currentStep]++;
    }
  }
}

int64_t inspectorPromCurrentStep() {
  if (!inspectorPromStepEnabled()) return -1;
  return gInspectorPromCurrentStep.load(std::memory_order_relaxed);
}

inspectorResult_t inspectorPromRecordProxyOp(
    const inspectorProxyOpInfo* op) {
  if (op == nullptr || op->applicationStep < 0) {
    return inspectorSuccess;
  }
  inspectorPromTopologySizes sizes = inspectorPromGetTopologySizes();
  inspectorPromSemanticFamily family = op->detached
    ? inspectorPromFamilyPxn
    : inspectorPromClassifyFamily(
        op->parentType == ncclProfileP2p, op->nranks, op->nnodes, sizes);
  if (family == inspectorPromFamilyUnknown) return inspectorSuccess;
  std::lock_guard<std::mutex> lock(gInspectorPromStreamingMutex);
  const bool hasComm = !op->detached && op->commInfo != nullptr;
  inspectorPromStepProxyKey key {
    op->applicationStep, family, op->func, op->messageSizeBytes, op->isSend != 0,
    hasComm ? op->commInfo->commHashStr : "unknown",
    hasComm && op->commInfo->commName && op->commInfo->commName[0]
      ? op->commInfo->commName : "unknown",
    hasComm ? op->commInfo->rank : -1,
    op->nranks,
    op->nnodes
  };
  inspectorPromStepProxyAgg* aggPtr = nullptr;
  inspectorPromDevice* devicePtr = nullptr;
  if (op->detached) {
    aggPtr = &gInspectorPromDetachedProxy[key];
  } else {
    if (op->commInfo == nullptr) return inspectorSuccess;
    inspectorPromDevice& device =
      gInspectorPromStreamingDevices[op->commInfo->deviceUuidStr];
    inspectorPromInitDevice(device, op->commInfo);
    aggPtr = &device.stepProxy[key];
    devicePtr = &device;
  }
  inspectorPromStepProxyAgg& agg = *aggPtr;
  agg.count += op->proxyStepCount;
  agg.transferBytes += op->transferSizeBytes;
  agg.unknownTransferSizes += op->unknownTransferSizes;
  agg.missingTransitions += op->missingTransitions;
  for (int phase = 0; phase < inspectorProxyWaitPhaseCount; phase++) {
    agg.phaseCount[phase] += op->phaseCount[phase];
    agg.phaseSumUsecs[phase] += op->phaseSumUsecs[phase];
    agg.phaseMaxUsecs[phase] = std::max(
      agg.phaseMaxUsecs[phase], op->phaseMaxUsecs[phase]);
  }
  if (devicePtr != nullptr) devicePtr->hasData = true;
  return inspectorSuccess;
}

static void inspectorPromAggUpdate(inspectorPromBucketAgg& agg,
                                   const inspectorCompletedOpInfo& op) {
  agg.count++;
  agg.execTimeSum += static_cast<double>(op.execTimeUsecs);
  agg.execTimeMin = std::min(agg.execTimeMin, op.execTimeUsecs);
  agg.execTimeMax = std::max(agg.execTimeMax, op.execTimeUsecs);
  if (inspectorPromStreamingEnabled()) {
    inspectorPromBoundedSampleUpdate(agg.execTimes,
                                     kInspectorPromPercentileSampleCapacity,
                                     agg.count, op.sn, op.msgSizeBytes,
                                     op.execTimeUsecs);
  } else {
    agg.execTimes.push_back(op.execTimeUsecs);
    inspectorPromSlowEvent event;
    event.sequence = op.sn;
    event.execTimeUsecs = op.execTimeUsecs;
    event.startTimestampUsecs
      = op.evtTrk.evntTrace[NCCL_INSP_EVT_TRK_OP_START].ts;
    event.stopTimestampUsecs
      = op.evtTrk.evntTrace[NCCL_INSP_EVT_TRK_OP_STOP].ts;
    event.peer = op.peer;
    inspectorPromBoundedTopK(
      agg.slowest, std::move(event), kInspectorPromGlobalSlowEventCapacity,
      [](const inspectorPromSlowEvent& lhs, const inspectorPromSlowEvent& rhs) {
        return lhs.execTimeUsecs > rhs.execTimeUsecs;
      });
  }

  if (op.algoBwGbs > 0.0) {
    // Geometric mean trick: sum log(x), then exp(sum/count) later.
    agg.algoLogSum += std::log(op.algoBwGbs);
    agg.algoCount++;
  }
  if (op.busBwGbs > 0.0) {
    // log(a*b*...)=sum(log(a)), so we accumulate logs for GM.
    agg.busLogSum += std::log(op.busBwGbs);
    agg.busCount++;
  }
}

static uint64_t inspectorPromPercentile(const std::vector<uint64_t>& values,
                                        double percentile) {
  if (values.empty()) return 0;
  std::vector<uint64_t> sorted(values);
  std::sort(sorted.begin(), sorted.end());
  size_t index = static_cast<size_t>(std::ceil(percentile * sorted.size()));
  if (index == 0) index = 1;
  return sorted[std::min(index - 1, sorted.size() - 1)];
}

/*
 * Description:
 *   Formats message size as a power-of-two range label (e.g., 1-2KB, 2-4MB).
 *
 * Thread Safety:
 *   Not thread-safe. Onus of thread safety is on the caller/owner of the buffer.
 *
 * Input:
 *   size_t bytes - number of bytes.
 *   char* output - output buffer for formatted string.
 *   size_t outputSize - size of output buffer.
 *
 * Output:
 *   Range label string is written to buffer.
 *
 * Return:
 *   None.
 */
static void inspectorPromFormatMessageSizeRange(size_t bytes,
                                                char* output,
                                                size_t outputSize) {
  if (bytes == 0) {
    snprintf(output, outputSize, "0B");
    return;
  }

  const char* units[] = {"B", "KB", "MB", "GB", "TB"};
  size_t unitIndex = 0;
  double byteValue = static_cast<double>(bytes);

  while (byteValue >= 1024.0 && unitIndex < 4) {
    byteValue /= 1024.0;
    unitIndex++;
  }

  double value = byteValue;
  if (value < 1.0) {
    value = 1.0;
  }

  double lower = std::floor(value);
  if (lower < 1.0) {
    lower = 1.0;
  }
  double upper = lower + 1.0;

  snprintf(output, outputSize, "%.0f-%.0f%s", lower, upper, units[unitIndex]);
}

/*
 * Description:
 *   Returns the lower-bound (in bytes) of the decimal range bucket.
 *
 * Return:
 *   size_t - lower-bound of the bucket in bytes.
 */
static size_t inspectorPromMessageSizeRangeLowerBound(size_t bytes) {
  if (bytes == 0) {
    return 0;
  }

  size_t unitIndex = 0;
  double value = static_cast<double>(bytes);
  while (value >= 1024.0 && unitIndex < 4) {
    value /= 1024.0;
    unitIndex++;
  }

  if (value < 1.0) {
    value = 1.0;
  }

  double lower = std::floor(value);
  if (lower < 1.0) {
    lower = 1.0;
  }

  size_t unitBytes = static_cast<size_t>(std::pow(1024.0, unitIndex));
  return static_cast<size_t>(lower) * unitBytes;
}


/*
 * Description:
 *
 *   Formats labels for Prometheus metrics from a collective bucket key.
 *
 * Thread Safety:
 *
 *   Not thread-safe. Onus of thread safety is on the caller/owner of
 *   the buffer.
 *
 * Input:
 *   char* labels - output buffer for formatted labels.
 *   size_t labelSize - size of labels buffer.
 *   int nranks - number of ranks in communicator.
 *   int nnodes - number of nodes in communicator.
 *   ncclFunc_t func - collective operation type.
 *   size_t msgSizeRangeBytes - message size range lower bound in bytes.
 *
 * Output:
 *   Formatted labels string is written to buffer.
 *
 * Return:
 *   inspectorResult_t - success or error code.
 */
static void inspectorPromGetVersion(char* version, size_t versionSize) {
  if (version == nullptr || versionSize == 0) {
    return;
  }
  snprintf(version, versionSize, "v%d.%d",
           NCCL_PROFILER_INTERFACE_VERSION,
           kInspectorPromFormatMinor);
  version[versionSize - 1] = '\0';
}

static inspectorResult_t inspectorPromGetLabelsColl(char* labels,
                                                    size_t labelSize,
                                                    const char* nodeName,
                                                    const char* gpuName,
                                                    const char* commId,
                                                    const char* commName,
                                                    const char* version,
                                                    int rank,
                                                    int nranks,
                                                    int nnodes,
                                                    ncclFunc_t func,
                                                    const char* algoProto,
                                                    size_t msgSizeRangeBytes,
                                                    size_t msgSizeBytes,
                                                    const char* timingSource) {
  const char* jobId = getenv("SLURM_JOB_ID");
  const char* worldRank = getenv("SLURM_PROCID");
  const char* localRank = getenv("SLURM_LOCALID");
  char msgSizeStr[32];
  inspectorPromFormatMessageSizeRange(msgSizeRangeBytes,
                                      msgSizeStr,
                                      sizeof(msgSizeStr));

  if (inspectorPromStreamingEnabled()) {
    int compactRet = snprintf(
      labels, labelSize,
      "version=\"%s\",comm_id=\"%s\",comm_name=\"%s\",comm_rank=\"%d\","
      "n_nodes=\"%d\",nranks=\"%d\",collective=\"%s\","
      "message_size_bytes=\"%zu\",algo_proto=\"%s\",timing_source=\"%s\"",
      (version && version[0]) ? version : "unknown",
      (commId && commId[0]) ? commId : "unknown",
      (commName && commName[0]) ? commName : "unknown",
      rank, nnodes, nranks, ncclFuncToString(func), msgSizeBytes,
      algoProto ? algoProto : "unknown",
      timingSource ? timingSource : "unknown");
    return (compactRet < 0 || (size_t)compactRet >= labelSize)
      ? inspectorMemoryError : inspectorSuccess;
  }

  int ret = snprintf(labels, labelSize,
                     "version=\"%s\",slurm_job_id=\"%s\",world_rank=\"%s\","
                     "local_rank=\"%s\",node=\"%s\",gpu=\"%s\","
                     "comm_id=\"%s\",comm_name=\"%s\",comm_rank=\"%d\","
                     "n_nodes=\"%d\",nranks=\"%d\","
                     "collective=\"%s\",message_size=\"%s\","
                     "message_size_bytes=\"%zu\",algo_proto=\"%s\","
                     "timing_source=\"%s\"",
                     (version && version[0]) ? version : "unknown",
                     jobId ? jobId : "unknown",
                     worldRank ? worldRank : "unknown",
                     localRank ? localRank : "unknown",
                     (nodeName && nodeName[0]) ? nodeName : "unknown",
                     (gpuName && gpuName[0]) ? gpuName : "unknown",
                     (commId && commId[0]) ? commId : "unknown",
                     (commName && commName[0]) ? commName : "unknown",
                     rank,
                     nnodes,
                     nranks,
                     ncclFuncToString(func),
                     msgSizeStr,
                     msgSizeBytes,
                     algoProto ? algoProto : "unknown",
                     timingSource ? timingSource : "unknown");

  if (ret < 0 || (size_t)ret >= labelSize) {
    return inspectorMemoryError;
  }
  return inspectorSuccess;
}

/*
 * Description:
 *
 *   Formats labels for Prometheus metrics from a P2P bucket key.
 *
 * Thread Safety:
 *   Not thread-safe.
 *
 * Input:
 *   char* labels - output buffer for formatted labels.
 *   size_t labelSize - size of labels buffer.
 *   int nranks - number of ranks in communicator.
 *   int nnodes - number of nodes in communicator.
 *   ncclFunc_t func - P2P operation type.
 *   size_t msgSizeRangeBytes - message size range lower bound in bytes.
 *
 * Output:
 *   Formatted labels string is written to buffer.
 *
 * Return:
 *   inspectorResult_t - success or error code.
 */
static inspectorResult_t inspectorPromGetLabelsP2p(char* labels,
                                                   size_t labelSize,
                                                   const char* nodeName,
                                                   const char* gpuName,
                                                   const char* commId,
                                                   const char* commName,
                                                   const char* version,
                                                   int rank,
                                                   int peer,
                                                   int nranks,
                                                   int nnodes,
                                                   ncclFunc_t func,
                                                   size_t msgSizeRangeBytes,
                                                   size_t msgSizeBytes,
                                                   const char* timingSource) {
  const char* jobId = getenv("SLURM_JOB_ID");
  const char* worldRank = getenv("SLURM_PROCID");
  const char* localRank = getenv("SLURM_LOCALID");
  char msgSizeStr[32];
  inspectorPromFormatMessageSizeRange(msgSizeRangeBytes,
                                      msgSizeStr,
                                      sizeof(msgSizeStr));

  if (inspectorPromStreamingEnabled()) {
    int compactRet = snprintf(
      labels, labelSize,
      "version=\"%s\",comm_id=\"%s\",comm_name=\"%s\",comm_rank=\"%d\","
      "peer=\"%d\",n_nodes=\"%d\",nranks=\"%d\",p2p_operation=\"%s\","
      "message_size_bytes=\"%zu\",timing_source=\"%s\"",
      (version && version[0]) ? version : "unknown",
      (commId && commId[0]) ? commId : "unknown",
      (commName && commName[0]) ? commName : "unknown",
      rank, peer, nnodes, nranks, ncclFuncToString(func), msgSizeBytes,
      timingSource ? timingSource : "unknown");
    return (compactRet < 0 || (size_t)compactRet >= labelSize)
      ? inspectorMemoryError : inspectorSuccess;
  }

  int ret = snprintf(labels,
                     labelSize,
                     "version=\"%s\",slurm_job_id=\"%s\",world_rank=\"%s\","
                     "local_rank=\"%s\",node=\"%s\",gpu=\"%s\","
                     "comm_id=\"%s\",comm_name=\"%s\",comm_rank=\"%d\","
                     "peer=\"%d\",n_nodes=\"%d\",nranks=\"%d\","
                     "p2p_operation=\"%s\",message_size=\"%s\","
                     "message_size_bytes=\"%zu\",timing_source=\"%s\"",
                     (version && version[0]) ? version : "unknown",
                     jobId ? jobId : "unknown",
                     worldRank ? worldRank : "unknown",
                     localRank ? localRank : "unknown",
                     (nodeName && nodeName[0]) ? nodeName : "unknown",
                     (gpuName && gpuName[0]) ? gpuName : "unknown",
                     (commId && commId[0]) ? commId : "unknown",
                     (commName && commName[0]) ? commName : "unknown",
                     rank,
                     peer,
                     nnodes,
                     nranks,
                     ncclFuncToString(func),
                     msgSizeStr,
                     msgSizeBytes,
                     timingSource ? timingSource : "unknown");

  if (ret < 0 || (size_t)ret >= labelSize) {
    return inspectorMemoryError;
  }
  return inspectorSuccess;
}

static void inspectorPromGetNodeName(char* nodeName, size_t nodeNameSize) {
  if (nodeName == nullptr || nodeNameSize == 0) {
    return;
  }
  if (gethostname(nodeName, nodeNameSize - 1) != 0) {
    snprintf(nodeName, nodeNameSize, "unknown");
    return;
  }
  nodeName[nodeNameSize - 1] = '\0';
}

static void inspectorPromInitDevice(inspectorPromDevice& device,
                                    const inspectorCommInfo* commInfo) {
  if (device.deviceUuidStr.empty()) {
    device.deviceUuidStr = commInfo->deviceUuidStr;
  }
  if (device.nodeName.empty()) {
    char nodeName[256];
    inspectorPromGetNodeName(nodeName, sizeof(nodeName));
    device.nodeName = nodeName;
  }
  if (device.gpuName.empty()) {
    char gpuName[16];
    snprintf(gpuName, sizeof(gpuName), "GPU%d", commInfo->cudaDeviceId);
    device.gpuName = gpuName;
  }
}

static void inspectorPromAddGlobalSlowEvent(
    std::vector<inspectorPromGlobalSlowEvent>& slowest,
    const char* labels,
    const inspectorCompletedOpInfo& op) {
  inspectorPromGlobalSlowEvent item;
  item.labels = labels;
  item.event.sequence = op.sn;
  item.event.execTimeUsecs = op.execTimeUsecs;
  item.event.startTimestampUsecs
    = op.evtTrk.evntTrace[NCCL_INSP_EVT_TRK_OP_START].ts;
  item.event.stopTimestampUsecs
    = op.evtTrk.evntTrace[NCCL_INSP_EVT_TRK_OP_STOP].ts;
  item.event.peer = op.peer;
  inspectorPromBoundedTopK(
    slowest, std::move(item), kInspectorPromGlobalSlowEventCapacity,
    [](const inspectorPromGlobalSlowEvent& lhs,
       const inspectorPromGlobalSlowEvent& rhs) {
      return lhs.event.execTimeUsecs > rhs.event.execTimeUsecs;
    });
}

inspectorResult_t inspectorPromRecordCompleted(inspectorCommInfo* commInfo,
                                               const inspectorCompletedOpInfo* op) {
  if (commInfo == nullptr || op == nullptr) return inspectorMemoryError;

  std::lock_guard<std::mutex> lock(gInspectorPromStreamingMutex);
  inspectorPromDevice& device
    = gInspectorPromStreamingDevices[commInfo->deviceUuidStr];
  inspectorPromInitDevice(device, commInfo);
  const char* commName
    = (commInfo->commName && commInfo->commName[0]) ? commInfo->commName : "unknown";
  size_t msgSizeRangeBytes
    = inspectorPromMessageSizeRangeLowerBound(op->msgSizeBytes);
  char labels[1024];
  char version[16];
  inspectorPromGetVersion(version, sizeof(version));

  if (op->isP2p) {
    inspectorPromP2pBucketKey key {
      commInfo->nranks,
      commInfo->nnodes,
      op->func,
      msgSizeRangeBytes,
      op->msgSizeBytes,
      commInfo->rank,
      op->peer,
      commInfo->commHashStr,
      commName,
      inspectorTimingSourceToString(op->timingSource)
    };
    inspectorPromAggUpdate(device.p2pBuckets[key], *op);
    INS_CHK(inspectorPromGetLabelsP2p(labels, sizeof(labels),
                                      device.nodeName.c_str(),
                                      device.gpuName.c_str(),
                                      key.commId.c_str(), key.commName.c_str(),
                                      version, key.rank, key.peer, key.nranks,
                                      key.nnodes, key.func, key.msgSizeRangeBytes,
                                      key.msgSizeBytes, key.timingSource.c_str()));
    inspectorPromAddGlobalSlowEvent(device.p2pSlowest, labels, *op);
  } else {
    const char* algo = op->algo[0] ? op->algo : "unknown";
    const char* proto = op->proto[0] ? op->proto : "unknown";
    inspectorPromCollBucketKey key {
      commInfo->nranks,
      commInfo->nnodes,
      op->func,
      msgSizeRangeBytes,
      op->msgSizeBytes,
      commInfo->rank,
      commInfo->commHashStr,
      commName,
      std::string(algo) + "_" + proto,
      inspectorTimingSourceToString(op->timingSource)
    };
    inspectorPromAggUpdate(device.collBuckets[key], *op);
    INS_CHK(inspectorPromGetLabelsColl(labels, sizeof(labels),
                                       device.nodeName.c_str(),
                                       device.gpuName.c_str(),
                                       key.commId.c_str(), key.commName.c_str(),
                                       version, key.rank, key.nranks, key.nnodes,
                                       key.func, key.algoProto.c_str(),
                                       key.msgSizeRangeBytes, key.msgSizeBytes,
                                       key.timingSource.c_str()));
    inspectorPromAddGlobalSlowEvent(device.collSlowest, labels, *op);
  }

  inspectorPromStepUpdate(device, commInfo, *op);

  device.hasData = true;
  return inspectorSuccess;
}

static void inspectorPromTakeStreamingDevices(
    std::map<std::string, inspectorPromDevice>& devices,
    std::map<inspectorPromStepProxyKey,
             inspectorPromStepProxyAgg>& detachedProxy) {
  std::lock_guard<std::mutex> lock(gInspectorPromStreamingMutex);
  devices.swap(gInspectorPromStreamingDevices);
  detachedProxy.swap(gInspectorPromDetachedProxy);
}

/*
 * Description:
 *
 *   Generates GPU-specific Prometheus filename using pre-computed
 *   device UUID.  Each GPU gets its own file containing metrics from
 *   all communicators using that GPU.
 *
 * Thread Safety:
 *
 *   Not thread-safe. Onus of thread safety is on the caller/owner of
 *   the buffer.
 *
 * Input:
 *   const char* baseFilename - base output file path.
 *   const char* deviceUuidStr - pre-computed device UUID string.
 *   char* output - output buffer for filename.
 *   size_t outputSize - size of output buffer.
 *
 * Output:
 *   UUID-based filename is written to output buffer.
 *
 * Return:
 *   inspectorResult_t - success or error code.
 */
static inspectorResult_t inspectorPromGetFilename(const char* baseFilename,
                                                  const char* deviceUuidStr,
                                                  char* output,
                                                  size_t outputSize) {
  const char* jobId = getenv("SLURM_JOB_ID");
  if (inspectorPromStreamingEnabled()) {
    snprintf(output, outputSize,
             "%s/nccl_inspector_metrics_job%s_%s.prom",
             baseFilename, (jobId && jobId[0]) ? jobId : "unknown",
             deviceUuidStr);
  } else {
    snprintf(output, outputSize,
             "%s/nccl_inspector_metrics_%s.prom",
             baseFilename, deviceUuidStr);
  }

  return inspectorSuccess;
}

/*
 * Description:
 *
 *   Returns the file handle for a device bucket, creating one if
 *   needed.
 *
 * Thread Safety:
 *
 *   Not thread-safe. Onus of thread safety is on the caller/owner.
 *
 */
static inspectorResult_t inspectorPromGetDeviceFile(inspectorPromDevice& device,
                                                    const char* output_root,
                                                    uint64_t currentTime,
                                                    struct inspectorDumpThread* dumpThread,
                                                    FILE** fileOut) {
  if (device.file != nullptr) {
    *fileOut = device.file;
    return inspectorSuccess;
  }

  char filename[1024];
  INS_CHK(inspectorPromGetFilename(output_root,
                                   device.deviceUuidStr.c_str(),
                                   filename,
                                   sizeof(filename)));

  FILE* file
    = dumpThread ? dumpThread->getOrCreateFileHandle(device.deviceUuidStr.c_str(),
                                                     filename,
                                                     currentTime) : NULL;
  if (!file) {
    INFO_INSPECTOR("NCCL Inspector: Failed to get file handle for device UUID %s, file %s",
                   device.deviceUuidStr.c_str(), filename);
    return inspectorFileOpenError;
  }

  device.file = file;
  *fileOut = file;
  return inspectorSuccess;
}


/*
 * Description:
 *   Writes aggregated Prometheus metrics for a collective bucket.
 *
 * Thread Safety:
 *
 *   Not thread-safe. Onus of thread safety is on the caller/owner of
 *   the file handle.
 *
 */
static inspectorResult_t inspectorPromWriteCollBucket(FILE* file,
                                                      const inspectorPromDevice& device,
                                                      const inspectorPromCollBucketKey& key,
                                                      const inspectorPromBucketAgg& agg) {
  if (!file) {
    return inspectorFileOpenError;
  }

  char labels[1024];
  memset(labels, 0, sizeof(labels));

  char version[16];
  inspectorPromGetVersion(version, sizeof(version));

  INS_CHK(inspectorPromGetLabelsColl(labels,
                                     sizeof(labels),
                                     device.nodeName.c_str(),
                                     device.gpuName.c_str(),
                                     key.commId.c_str(),
                                     key.commName.c_str(),
                                     version,
                                     key.rank,
                                     key.nranks,
                                     key.nnodes,
                                     key.func,
                                     key.algoProto.c_str(),
                                     key.msgSizeRangeBytes,
                                     key.msgSizeBytes,
                                     key.timingSource.c_str()));

  double execMean = agg.count ? (agg.execTimeSum / agg.count) : 0.0;
  // GM = exp((1/n) * sum(log(x))) from log rules.
  double busMean = agg.busCount ? std::exp(agg.busLogSum / agg.busCount) : 0.0;

  uint64_t p50 = inspectorPromPercentile(agg.execTimes, 0.50);
  uint64_t p95 = inspectorPromPercentile(agg.execTimes, 0.95);
  uint64_t p99 = inspectorPromPercentile(agg.execTimes, 0.99);
  uint64_t execMin = agg.count ? agg.execTimeMin : 0;

  char buffer[8192];
  int written;
  if (inspectorPromStreamingEnabled()) {
    written = snprintf(buffer, sizeof(buffer),
                       "nccl_bus_bandwidth_gbs{%s} %.6g\n"
                       "nccl_collective_count{%s,percentile_samples=\"%zu\"} %" PRIu64 "\n"
                       "nccl_collective_exec_time_sum_microseconds{%s} %.6g\n"
                       "nccl_collective_exec_time_min_microseconds{%s} %" PRIu64 "\n"
                       "nccl_collective_exec_time_p50_microseconds{%s} %" PRIu64 "\n"
                       "nccl_collective_exec_time_p95_microseconds{%s} %" PRIu64 "\n"
                       "nccl_collective_exec_time_p99_microseconds{%s} %" PRIu64 "\n"
                       "nccl_collective_exec_time_max_microseconds{%s} %" PRIu64 "\n",
                       labels, busMean,
                       labels, agg.execTimes.size(), agg.count,
                       labels, agg.execTimeSum,
                       labels, execMin,
                       labels, p50,
                       labels, p95,
                       labels, p99,
                       labels, agg.execTimeMax);
  } else {
    written = snprintf(buffer, sizeof(buffer),
                       "nccl_bus_bandwidth_gbs{%s} %.6g\n"
                       "nccl_collective_count{%s} %" PRIu64 "\n"
                       "nccl_collective_exec_time_sum_microseconds{%s} %.6g\n"
                       "nccl_collective_exec_time_microseconds{%s} %.6g\n"
                       "nccl_collective_exec_time_min_microseconds{%s} %" PRIu64 "\n"
                       "nccl_collective_exec_time_p50_microseconds{%s} %" PRIu64 "\n"
                       "nccl_collective_exec_time_p95_microseconds{%s} %" PRIu64 "\n"
                       "nccl_collective_exec_time_p99_microseconds{%s} %" PRIu64 "\n"
                       "nccl_collective_exec_time_max_microseconds{%s} %" PRIu64 "\n",
                       labels, busMean,
                       labels, agg.count,
                       labels, agg.execTimeSum,
                       labels, execMean,
                       labels, execMin,
                       labels, p50,
                       labels, p95,
                       labels, p99,
                       labels, agg.execTimeMax);
  }

  if (written < 0 || (size_t)written >= sizeof(buffer)) {
    return inspectorMemoryError;
  }

  if (fwrite(buffer, 1, written, file) != (size_t)written) {
    return inspectorFileOpenError;
  }

  if (!inspectorPromStreamingEnabled()) {
    for (size_t i = 0; i < agg.slowest.size(); i++) {
      const inspectorPromSlowEvent& event = agg.slowest[i];
      written = snprintf(buffer, sizeof(buffer),
                         "nccl_collective_slow_event_exec_time_microseconds{%s,"
                         "top_index=\"%zu\",sequence=\"%" PRIu64 "\","
                         "start_timestamp_us=\"%" PRIu64 "\","
                         "stop_timestamp_us=\"%" PRIu64 "\"} %" PRIu64 "\n",
                         labels, i, event.sequence, event.startTimestampUsecs,
                         event.stopTimestampUsecs, event.execTimeUsecs);
      if (written < 0 || (size_t)written >= sizeof(buffer)) {
        return inspectorMemoryError;
      }
      if (fwrite(buffer, 1, written, file) != (size_t)written) {
        return inspectorFileOpenError;
      }
    }
  }

  fflush(file);
  return inspectorSuccess;
}

/*
 * Description:
 *   Writes aggregated Prometheus metrics for a P2P bucket.
 *
 * Thread Safety:
 *   Not thread-safe. Onus of thread safety is on the caller/owner of
 *   the file handle.
 */
static inspectorResult_t inspectorPromWriteP2pBucket(FILE* file,
                                                     const inspectorPromDevice& device,
                                                     const inspectorPromP2pBucketKey& key,
                                                     const inspectorPromBucketAgg& agg) {
  if (!file) {
    return inspectorFileOpenError;
  }

  char labels[1024];
  memset(labels, 0, sizeof(labels));

  char version[16];
  inspectorPromGetVersion(version, sizeof(version));

  INS_CHK(inspectorPromGetLabelsP2p(labels,
                                    sizeof(labels),
                                    device.nodeName.c_str(),
                                    device.gpuName.c_str(),
                                    key.commId.c_str(),
                                    key.commName.c_str(),
                                    version,
                                    key.rank,
                                    key.peer,
                                    key.nranks,
                                    key.nnodes,
                                    key.func,
                                    key.msgSizeRangeBytes,
                                    key.msgSizeBytes,
                                    key.timingSource.c_str()));

  double execMean = agg.count ? (agg.execTimeSum / agg.count) : 0.0;
  double busMean = agg.busCount ? std::exp(agg.busLogSum / agg.busCount) : 0.0;

  uint64_t p50 = inspectorPromPercentile(agg.execTimes, 0.50);
  uint64_t p95 = inspectorPromPercentile(agg.execTimes, 0.95);
  uint64_t p99 = inspectorPromPercentile(agg.execTimes, 0.99);
  uint64_t execMin = agg.count ? agg.execTimeMin : 0;

  char buffer[8192];
  int written;
  if (inspectorPromStreamingEnabled()) {
    written = snprintf(buffer, sizeof(buffer),
                       "nccl_p2p_bus_bandwidth_gbs{%s} %.6g\n"
                       "nccl_p2p_count{%s,percentile_samples=\"%zu\"} %" PRIu64 "\n"
                       "nccl_p2p_exec_time_sum_microseconds{%s} %.6g\n"
                       "nccl_p2p_exec_time_min_microseconds{%s} %" PRIu64 "\n"
                       "nccl_p2p_exec_time_p50_microseconds{%s} %" PRIu64 "\n"
                       "nccl_p2p_exec_time_p95_microseconds{%s} %" PRIu64 "\n"
                       "nccl_p2p_exec_time_p99_microseconds{%s} %" PRIu64 "\n"
                       "nccl_p2p_exec_time_max_microseconds{%s} %" PRIu64 "\n",
                       labels, busMean,
                       labels, agg.execTimes.size(), agg.count,
                       labels, agg.execTimeSum,
                       labels, execMin,
                       labels, p50,
                       labels, p95,
                       labels, p99,
                       labels, agg.execTimeMax);
  } else {
    written = snprintf(buffer, sizeof(buffer),
                       "nccl_p2p_bus_bandwidth_gbs{%s} %.6g\n"
                       "nccl_p2p_count{%s} %" PRIu64 "\n"
                       "nccl_p2p_exec_time_sum_microseconds{%s} %.6g\n"
                       "nccl_p2p_exec_time_microseconds{%s} %.6g\n"
                       "nccl_p2p_exec_time_min_microseconds{%s} %" PRIu64 "\n"
                       "nccl_p2p_exec_time_p50_microseconds{%s} %" PRIu64 "\n"
                       "nccl_p2p_exec_time_p95_microseconds{%s} %" PRIu64 "\n"
                       "nccl_p2p_exec_time_p99_microseconds{%s} %" PRIu64 "\n"
                       "nccl_p2p_exec_time_max_microseconds{%s} %" PRIu64 "\n",
                       labels, busMean,
                       labels, agg.count,
                       labels, agg.execTimeSum,
                       labels, execMean,
                       labels, execMin,
                       labels, p50,
                       labels, p95,
                       labels, p99,
                       labels, agg.execTimeMax);
  }

  if (written < 0 || (size_t)written >= sizeof(buffer)) {
    return inspectorMemoryError;
  }

  if (fwrite(buffer, 1, written, file) != (size_t)written) {
    return inspectorFileOpenError;
  }

  if (!inspectorPromStreamingEnabled()) {
    for (size_t i = 0; i < agg.slowest.size(); i++) {
      const inspectorPromSlowEvent& event = agg.slowest[i];
      written = snprintf(buffer, sizeof(buffer),
                         "nccl_p2p_slow_event_exec_time_microseconds{%s,"
                         "top_index=\"%zu\",sequence=\"%" PRIu64 "\",peer=\"%d\","
                         "start_timestamp_us=\"%" PRIu64 "\","
                         "stop_timestamp_us=\"%" PRIu64 "\"} %" PRIu64 "\n",
                         labels, i, event.sequence, event.peer,
                         event.startTimestampUsecs, event.stopTimestampUsecs,
                         event.execTimeUsecs);
      if (written < 0 || (size_t)written >= sizeof(buffer)) {
        return inspectorMemoryError;
      }
      if (fwrite(buffer, 1, written, file) != (size_t)written) {
        return inspectorFileOpenError;
      }
    }
  }

  fflush(file);
  return inspectorSuccess;
}

static inspectorResult_t inspectorPromWriteGlobalSlowEvents(
    FILE* file,
    const char* metric,
    const std::vector<inspectorPromGlobalSlowEvent>& slowest) {
  if (!file) return inspectorFileOpenError;

  char buffer[8192];
  for (size_t i = 0; i < slowest.size(); i++) {
    const inspectorPromGlobalSlowEvent& item = slowest[i];
    int written = snprintf(
      buffer, sizeof(buffer),
      "%s{%s,top_index=\"%zu\",sequence=\"%" PRIu64 "\","
      "start_timestamp_us=\"%" PRIu64 "\",stop_timestamp_us=\"%" PRIu64
      "\"} %" PRIu64 "\n",
      metric, item.labels.c_str(), i, item.event.sequence,
      item.event.startTimestampUsecs, item.event.stopTimestampUsecs,
      item.event.execTimeUsecs);
    if (written < 0 || (size_t)written >= sizeof(buffer)) {
      return inspectorMemoryError;
    }
    if (fwrite(buffer, 1, written, file) != (size_t)written) {
      return inspectorFileOpenError;
    }
  }
  return inspectorSuccess;
}

static inspectorResult_t inspectorPromWriteStepFamilies(
    FILE* file,
    const std::map<inspectorPromStepFamilyKey,
                   inspectorPromStepFamilyAgg>& stepFamilies) {
  if (!file) return inspectorFileOpenError;

  char buffer[1024];
  int infoWritten = snprintf(
    buffer, sizeof(buffer),
    "# nccl_inspector_step_info {\"capacity\":%zu,\"retained_steps\":%zu}\n",
    inspectorPromStepCapacity(), gInspectorPromRetainedSteps.size());
  if (infoWritten < 0 || (size_t)infoWritten >= sizeof(buffer)) {
    return inspectorMemoryError;
  }
  if (fwrite(buffer, 1, infoWritten, file) != (size_t)infoWritten) {
    return inspectorFileOpenError;
  }
  for (const auto& entry : stepFamilies) {
    const inspectorPromStepFamilyKey& key = entry.first;
    const inspectorPromStepFamilyAgg& agg = entry.second;
    uint64_t firstStart = agg.count ? agg.firstStartTimestampUsecs : 0;
    int written = snprintf(
      buffer, sizeof(buffer),
      "# nccl_inspector_step {\"step\":%" PRId64
      ",\"family\":\"%s\",\"operation\":\"%s\""
      ",\"count\":%" PRIu64 ",\"sum_us\":%.6g"
      ",\"max_us\":%" PRIu64
      ",\"first_start_us\":%" PRIu64 ",\"last_stop_us\":%" PRIu64
      "}\n",
      key.step, inspectorPromSemanticFamilyName(key.family),
      ncclFuncToString(key.func), agg.count, agg.execTimeSum,
      agg.execTimeMax, firstStart, agg.lastStopTimestampUsecs);
    if (written < 0 || (size_t)written >= sizeof(buffer)) {
      return inspectorMemoryError;
    }
    if (fwrite(buffer, 1, written, file) != (size_t)written) {
      return inspectorFileOpenError;
    }
  }
  return inspectorSuccess;
}

static inspectorResult_t inspectorPromWriteStepProxy(
    FILE* file,
    const std::map<inspectorPromStepProxyKey,
                   inspectorPromStepProxyAgg>& proxy) {
  if (!file) return inspectorFileOpenError;
  char buffer[3072];
  int infoWritten = snprintf(
    buffer, sizeof(buffer),
    "# nccl_inspector_step_proxy_info {\"records\":%zu,"
    "\"dropped_ops\":%" PRIu64 ",\"dropped_steps\":%" PRIu64
    ",\"detached_ops\":%" PRIu64
    ",\"phase_fields\":[\"send_gpu_wait\",\"send_peer_wait\","
    "\"send_wait\",\"recv_wait\",\"recv_flush_wait\","
    "\"recv_gpu_wait\"]}\n",
    proxy.size(), inspectorProxyPoolDroppedOps(), inspectorProxyPoolDroppedSteps(),
    inspectorProxyPoolDetachedOps());
  if (infoWritten < 0 || (size_t)infoWritten >= sizeof(buffer)) {
    return inspectorMemoryError;
  }
  if (fwrite(buffer, 1, infoWritten, file) != (size_t)infoWritten) {
    return inspectorFileOpenError;
  }
  for (const auto& entry : proxy) {
    const inspectorPromStepProxyKey& key = entry.first;
    const inspectorPromStepProxyAgg& agg = entry.second;
    int written = snprintf(
      buffer, sizeof(buffer),
      "# nccl_inspector_step_proxy {\"step\":%" PRId64
      ",\"family\":\"%s\",\"operation\":\"%s\""
      ",\"message_size_bytes\":%zu,\"direction\":\"%s\""
      ",\"comm_id\":\"%s\",\"comm_name\":\"%s\""
      ",\"comm_rank\":%d,\"nranks\":%d,\"n_nodes\":%d"
      ",\"count\":%" PRIu64 ",\"transfer_bytes\":%" PRIu64
      ",\"unknown_transfer_sizes\":%" PRIu64
      ",\"missing_transitions\":%" PRIu64
      ",\"phase_count\":[%" PRIu64 ",%" PRIu64 ",%" PRIu64
      ",%" PRIu64 ",%" PRIu64 ",%" PRIu64 "]"
      ",\"phase_sum_us\":[%" PRIu64 ",%" PRIu64 ",%" PRIu64
      ",%" PRIu64 ",%" PRIu64 ",%" PRIu64 "]"
      ",\"phase_max_us\":[%" PRIu64 ",%" PRIu64 ",%" PRIu64
      ",%" PRIu64 ",%" PRIu64 ",%" PRIu64 "]}\n",
      key.step, inspectorPromSemanticFamilyName(key.family),
      ncclFuncToString(key.func), key.messageSizeBytes,
      key.isSend ? "send" : "recv", key.commId.c_str(), key.commName.c_str(),
      key.commRank, key.nranks, key.nnodes, agg.count, agg.transferBytes,
      agg.unknownTransferSizes, agg.missingTransitions,
      agg.phaseCount[0], agg.phaseCount[1], agg.phaseCount[2],
      agg.phaseCount[3], agg.phaseCount[4], agg.phaseCount[5],
      agg.phaseSumUsecs[0], agg.phaseSumUsecs[1], agg.phaseSumUsecs[2],
      agg.phaseSumUsecs[3], agg.phaseSumUsecs[4], agg.phaseSumUsecs[5],
      agg.phaseMaxUsecs[0], agg.phaseMaxUsecs[1], agg.phaseMaxUsecs[2],
      agg.phaseMaxUsecs[3], agg.phaseMaxUsecs[4], agg.phaseMaxUsecs[5]);
    if (written < 0 || (size_t)written >= sizeof(buffer)) {
      return inspectorMemoryError;
    }
    if (fwrite(buffer, 1, written, file) != (size_t)written) {
      return inspectorFileOpenError;
    }
  }
  return inspectorSuccess;
}

static inspectorResult_t inspectorPromWriteStepP2pEvents(
    FILE* file,
    const std::map<int64_t, std::vector<inspectorPromStepP2pEvent>>& byStep,
    const std::map<int64_t, uint64_t>& droppedByStep) {
  if (!file) return inspectorFileOpenError;

  uint64_t retained = 0;
  uint64_t dropped = 0;
  for (const auto& entry : byStep) retained += entry.second.size();
  for (const auto& entry : droppedByStep) dropped += entry.second;
  char buffer[1024];
  int infoWritten = snprintf(
    buffer, sizeof(buffer),
    "# nccl_inspector_step_p2p_info {\"capacity_per_step\":%zu,"
    "\"retained_records\":%" PRIu64 ",\"dropped_records\":%" PRIu64
    ",\"event_fields\":[\"start_delta_us\",\"envelope_us\","
    "\"duration_us\",\"sequence\"]}\n",
    inspectorPromStepP2pCapacity(), retained, dropped);
  if (infoWritten < 0 || (size_t)infoWritten >= sizeof(buffer)) {
    return inspectorMemoryError;
  }
  if (fwrite(buffer, 1, infoWritten, file) != (size_t)infoWritten) {
    return inspectorFileOpenError;
  }

  struct GroupKey {
    ncclFunc_t func;
    size_t messageSizeBytes;
    uint64_t commHash;
    int commRank;
    int peer;
    int nranks;
    int nnodes;
    inspectorTimingSource_t timingSource;

    bool operator<(const GroupKey& other) const {
      if (commHash != other.commHash) return commHash < other.commHash;
      if (commRank != other.commRank) return commRank < other.commRank;
      if (peer != other.peer) return peer < other.peer;
      if (func != other.func) return func < other.func;
      if (messageSizeBytes != other.messageSizeBytes) {
        return messageSizeBytes < other.messageSizeBytes;
      }
      if (nranks != other.nranks) return nranks < other.nranks;
      if (nnodes != other.nnodes) return nnodes < other.nnodes;
      return timingSource < other.timingSource;
    }
  };

  for (const auto& stepEntry : byStep) {
    std::map<GroupKey, std::vector<inspectorPromStepP2pEvent>> groups;
    for (const inspectorPromStepP2pEvent& event : stepEntry.second) {
      groups[GroupKey {
        event.func, event.messageSizeBytes, event.commHash, event.commRank,
        event.peer, event.nranks, event.nnodes, event.timingSource
      }].push_back(event);
    }
    for (auto& groupEntry : groups) {
      const GroupKey& key = groupEntry.first;
      std::vector<inspectorPromStepP2pEvent>& events = groupEntry.second;
      std::sort(
        events.begin(), events.end(),
        [](const inspectorPromStepP2pEvent& lhs,
           const inspectorPromStepP2pEvent& rhs) {
          if (lhs.startTimestampUsecs != rhs.startTimestampUsecs) {
            return lhs.startTimestampUsecs < rhs.startTimestampUsecs;
          }
          return lhs.sequence < rhs.sequence;
        });
      uint64_t base = events.front().startTimestampUsecs;
      int prefixWritten = snprintf(
        buffer, sizeof(buffer),
        "# nccl_inspector_step_p2p {\"step\":%" PRId64
        ",\"operation\":\"%s\",\"message_size_bytes\":%zu"
        ",\"comm_id\":\"%016" PRIx64 "\",\"comm_rank\":%d"
        ",\"peer\":%d,\"nranks\":%d,\"n_nodes\":%d"
        ",\"timing_source\":\"%s\",\"start_base_us\":%" PRIu64
        ",\"events\":[",
        stepEntry.first, ncclFuncToString(key.func), key.messageSizeBytes,
        key.commHash, key.commRank, key.peer, key.nranks, key.nnodes,
        inspectorTimingSourceToString(key.timingSource), base);
      if (prefixWritten < 0 || (size_t)prefixWritten >= sizeof(buffer)) {
        return inspectorMemoryError;
      }
      if (fwrite(buffer, 1, prefixWritten, file) != (size_t)prefixWritten) {
        return inspectorFileOpenError;
      }
      for (size_t index = 0; index < events.size(); index++) {
        const inspectorPromStepP2pEvent& event = events[index];
        uint64_t envelope = event.stopTimestampUsecs >= event.startTimestampUsecs
          ? event.stopTimestampUsecs - event.startTimestampUsecs : 0;
        int eventWritten = snprintf(
          buffer, sizeof(buffer), "%s[%" PRIu64 ",%" PRIu64 ",%" PRIu64
          ",%" PRIu64 "]", index ? "," : "",
          event.startTimestampUsecs - base, envelope, event.execTimeUsecs,
          event.sequence);
        if (eventWritten < 0 || (size_t)eventWritten >= sizeof(buffer)) {
          return inspectorMemoryError;
        }
        if (fwrite(buffer, 1, eventWritten, file) != (size_t)eventWritten) {
          return inspectorFileOpenError;
        }
      }
      static const char suffix[] = "]}\n";
      if (fwrite(suffix, 1, sizeof(suffix) - 1, file) != sizeof(suffix) - 1) {
        return inspectorFileOpenError;
      }
    }
  }
  return inspectorSuccess;
}

/*
 * Description:
 *
 *   Dumps the state of a single communicator to Prometheus format.
 *
 * Thread Safety:
 *   Not thread-safe (should be called with proper locking).
 *
 * Input:
 *   struct inspectorCommInfo* commInfo - communicator info.
 *   const char* filename - output filename.
 *   bool* needs_writing - set to true if output was written.
 *
 * Output:
 *   Prometheus metrics are written to file if needed.
 *
 * Return:
 *   inspectorResult_t - success or error code.
 */
static inspectorResult_t inspectorPromCommInfoDumpColl(struct inspectorCommInfo* commInfo,
                                                       inspectorPromDevice& device,
                                                       uint64_t* overwritten,
                                                       bool* needs_writing) {
  if (commInfo == nullptr) {
    return inspectorSuccess;
  }

  thread_local std::vector<inspectorCompletedOpInfo> drainedColl;
  drainedColl.clear();

  inspectorLockWr(&commInfo->guard);
  if (overwritten) *overwritten += commInfo->completedCollRing.overwritten;
  commInfo->completedCollRing.overwritten = 0;
  if (commInfo->dump_coll || inspectorRingNonEmpty(&commInfo->completedCollRing)) {
    if (commInfo->completedCollRing.size > 0
        && drainedColl.capacity() < commInfo->completedCollRing.size) {
      drainedColl.reserve(commInfo->completedCollRing.size);
    }
    INS_CHK(inspectorRingDrain<inspectorCompletedOpInfo>(&commInfo->completedCollRing,
                                                        drainedColl));
    commInfo->dump_coll = inspectorRingNonEmpty(&commInfo->completedCollRing);
  }
  inspectorUnlockRWLock(&commInfo->guard);

  if (!drainedColl.empty()) {
    *needs_writing = true;
    for (size_t i = 0; i < drainedColl.size(); i++) {
      const inspectorCompletedOpInfo& collInfo = drainedColl[i];
      size_t msgSizeRangeBytes
        = inspectorPromMessageSizeRangeLowerBound(collInfo.msgSizeBytes);
      const char* commName
        = (commInfo->commName && commInfo->commName[0]) ? commInfo->commName : "unknown";
      const char* algo = collInfo.algo[0] ? collInfo.algo : "unknown";
      const char* proto = collInfo.proto[0] ? collInfo.proto : "unknown";
      std::string algoProto = std::string(algo) + "_" + proto;
      inspectorPromCollBucketKey key {
        commInfo->nranks,
        commInfo->nnodes,
        collInfo.func,
        msgSizeRangeBytes,
        collInfo.msgSizeBytes,
        commInfo->rank,
        commInfo->commHashStr,
        commName,
        algoProto,
        inspectorTimingSourceToString(collInfo.timingSource)
      };
      inspectorPromAggUpdate(device.collBuckets[key], collInfo);
      char labels[1024];
      char version[16];
      inspectorPromGetVersion(version, sizeof(version));
      INS_CHK(inspectorPromGetLabelsColl(labels, sizeof(labels),
                                         device.nodeName.c_str(),
                                         device.gpuName.c_str(),
                                         key.commId.c_str(), key.commName.c_str(),
                                         version, key.rank, key.nranks, key.nnodes,
                                         key.func, key.algoProto.c_str(),
                                         key.msgSizeRangeBytes, key.msgSizeBytes,
                                         key.timingSource.c_str()));
      if (inspectorPromStreamingEnabled()) {
        inspectorPromAddGlobalSlowEvent(device.collSlowest, labels, collInfo);
      }
    }
  }

  return inspectorSuccess;
}

static inspectorResult_t inspectorPromCommInfoDumpP2p(struct inspectorCommInfo* commInfo,
                                                      inspectorPromDevice& device,
                                                      uint64_t* overwritten,
                                                      bool* needs_writing) {
  if (commInfo == nullptr) {
    return inspectorSuccess;
  }

  thread_local std::vector<inspectorCompletedOpInfo> drainedP2p;
  drainedP2p.clear();

  inspectorLockWr(&commInfo->guard);
  if (overwritten) *overwritten += commInfo->completedP2pRing.overwritten;
  commInfo->completedP2pRing.overwritten = 0;
  if (commInfo->dump_p2p || inspectorRingNonEmpty(&commInfo->completedP2pRing)) {
    if (commInfo->completedP2pRing.size > 0
        && drainedP2p.capacity() < commInfo->completedP2pRing.size) {
      drainedP2p.reserve(commInfo->completedP2pRing.size);
    }
    INS_CHK(inspectorRingDrain<inspectorCompletedOpInfo>(&commInfo->completedP2pRing,
                                                        drainedP2p));
    commInfo->dump_p2p = inspectorRingNonEmpty(&commInfo->completedP2pRing);
  }
  inspectorUnlockRWLock(&commInfo->guard);

  if (!drainedP2p.empty()) {
    *needs_writing = true;
    for (size_t i = 0; i < drainedP2p.size(); i++) {
      const inspectorCompletedOpInfo& p2pInfo = drainedP2p[i];
      size_t msgSizeRangeBytes
        = inspectorPromMessageSizeRangeLowerBound(p2pInfo.msgSizeBytes);
      const char* commName
        = (commInfo->commName && commInfo->commName[0]) ? commInfo->commName : "unknown";
      inspectorPromP2pBucketKey key {
        commInfo->nranks,
        commInfo->nnodes,
        p2pInfo.func,
        msgSizeRangeBytes,
        p2pInfo.msgSizeBytes,
        commInfo->rank,
        p2pInfo.peer,
        commInfo->commHashStr,
        commName,
        inspectorTimingSourceToString(p2pInfo.timingSource)
      };
      inspectorPromAggUpdate(device.p2pBuckets[key], p2pInfo);
      char labels[1024];
      char version[16];
      inspectorPromGetVersion(version, sizeof(version));
      INS_CHK(inspectorPromGetLabelsP2p(labels, sizeof(labels),
                                        device.nodeName.c_str(),
                                        device.gpuName.c_str(),
                                        key.commId.c_str(), key.commName.c_str(),
                                        version, key.rank, key.peer, key.nranks,
                                        key.nnodes, key.func,
                                        key.msgSizeRangeBytes, key.msgSizeBytes,
                                        key.timingSource.c_str()));
      if (inspectorPromStreamingEnabled()) {
        inspectorPromAddGlobalSlowEvent(device.p2pSlowest, labels, p2pInfo);
      }
    }
  }

  return inspectorSuccess;
}

static inspectorResult_t inspectorPromCommInfoDump(struct inspectorCommInfo* commInfo,
                                                   inspectorPromDevice& device,
                                                   uint64_t* collOverwritten,
                                                   uint64_t* p2pOverwritten,
                                                   bool* needs_writing) {
  *needs_writing = false;

  INS_CHK(inspectorPromCommInfoDumpColl(commInfo, device,
                                        collOverwritten, needs_writing));
  INS_CHK(inspectorPromCommInfoDumpP2p(commInfo, device,
                                       p2pOverwritten, needs_writing));

  return inspectorSuccess;
}

/*
 * Description:
 *   Drains per-communicator rings into per-device bucket aggregations.
 *
 * Thread Safety:
 *   Not thread-safe (caller must hold commList lock).
 */
static inspectorResult_t inspectorPromFillDeviceBuckets(struct inspectorCommInfoList* commList,
                                                        std::map<std::string, inspectorPromDevice>& devices,
                                                        uint32_t* processedOut) {
  if (processedOut == nullptr) {
    return inspectorMemoryError;
  }
  *processedOut = 0;

  for (struct inspectorCommInfo* itr = commList->comms;
       itr != nullptr;
       itr = itr->next) {
    bool needs_writing;

    std::string deviceKey(itr->deviceUuidStr);
    inspectorPromDevice& device = devices[deviceKey];
    inspectorPromInitDevice(device, itr);

    INS_CHK(inspectorPromCommInfoDump(itr,
                                      device,
                                      &device.collOverwritten,
                                      &device.p2pOverwritten,
                                      &needs_writing));

    if (needs_writing) {
      device.hasData = true;
      (*processedOut)++;
      TRACE_INSPECTOR(
        "NCCL Inspector: Processed comm %u for CUDA device (rank %d)",
        *processedOut, itr->rank);
    }
  }

  return inspectorSuccess;
}

/*
 * Description:
 *   Writes bucketized Prometheus metrics for each device.
 *
 * Thread Safety:
 *   Not thread-safe (caller must hold commList lock).
 */
static inspectorResult_t inspectorPromWriteDeviceBuckets(std::map<std::string,
                                                         inspectorPromDevice>& devices,
                                                         const char* output_root,
                                                         uint64_t currentTime,
                                                         struct inspectorDumpThread* dumpThread) {
  for (auto& entry : devices) {
    inspectorPromDevice& device = entry.second;
    if (!device.hasData) {
      continue;
    }
    FILE* file = nullptr;
    INS_CHK(inspectorPromGetDeviceFile(device,
                                       output_root,
                                       currentTime,
                                       dumpThread,
                                       &file));
    if (!file) {
      continue;
    }
    if (inspectorPromStreamingEnabled()) {
      const char* jobId = getenv("SLURM_JOB_ID");
      const char* worldRank = getenv("SLURM_PROCID");
      const char* localRank = getenv("SLURM_LOCALID");
      char version[16];
      inspectorPromGetVersion(version, sizeof(version));
      char identity[1024];
      int identityWritten = snprintf(
        identity, sizeof(identity),
        "nccl_inspector_identity{version=\"%s\",slurm_job_id=\"%s\","
        "world_rank=\"%s\",local_rank=\"%s\",node=\"%s\",gpu=\"%s\"} 1\n",
        version, jobId ? jobId : "unknown", worldRank ? worldRank : "unknown",
        localRank ? localRank : "unknown", device.nodeName.c_str(),
        device.gpuName.c_str());
      if (identityWritten < 0 || (size_t)identityWritten >= sizeof(identity)) {
        return inspectorMemoryError;
      }
      if (fwrite(identity, 1, identityWritten, file)
          != (size_t)identityWritten) {
        return inspectorFileOpenError;
      }
    }
    char completeness[512];
    int completenessWritten = snprintf(
      completeness, sizeof(completeness),
      "nccl_inspector_collective_ring_overwritten{node=\"%s\",gpu=\"%s\"} %" PRIu64 "\n"
      "nccl_inspector_p2p_ring_overwritten{node=\"%s\",gpu=\"%s\"} %" PRIu64 "\n",
      device.nodeName.c_str(), device.gpuName.c_str(), device.collOverwritten,
      device.nodeName.c_str(), device.gpuName.c_str(), device.p2pOverwritten);
    if (completenessWritten < 0
        || (size_t)completenessWritten >= sizeof(completeness)) {
      return inspectorMemoryError;
    }
    if (fwrite(completeness, 1, completenessWritten, file)
        != (size_t)completenessWritten) {
      return inspectorFileOpenError;
    }
    for (const auto& collEntry : device.collBuckets) {
      INS_CHK(inspectorPromWriteCollBucket(file,
                                           device,
                                           collEntry.first,
                                           collEntry.second));
    }
    for (const auto& p2pEntry : device.p2pBuckets) {
      INS_CHK(inspectorPromWriteP2pBucket(file,
                                          device,
                                          p2pEntry.first,
                                          p2pEntry.second));
    }
    INS_CHK(inspectorPromWriteGlobalSlowEvents(
      file, "nccl_collective_slow_event_exec_time_microseconds",
      device.collSlowest));
    INS_CHK(inspectorPromWriteGlobalSlowEvents(
      file, "nccl_p2p_slow_event_exec_time_microseconds",
      device.p2pSlowest));
    INS_CHK(inspectorPromWriteStepFamilies(file, device.stepFamilies));
    INS_CHK(inspectorPromWriteStepProxy(file, device.stepProxy));
    INS_CHK(inspectorPromWriteStepP2pEvents(
      file, device.stepP2pEvents, device.stepP2pDropped));
  }
  return inspectorSuccess;
}


/*
 * Description:
 *
 *   Dumps the state of all communicators in a commList to Prometheus format.
 *
 * Thread Safety:
 *   Thread-safe - acquires necessary locks to iterate through communicators.
 *
 * Input:
 *   struct inspectorCommInfoList* commList - list of communicators.
 *   const char* output_root - base output directory.
 *
 * Output:
 *   Prometheus metrics are written to UUID-named file.
 *
 * Return:
 *   inspectorResult_t - success or error code.
 */
inspectorResult_t inspectorPromCommInfoListDump(struct inspectorCommInfoList* commList,
                                                const char* output_root,
                                                struct inspectorDumpThread* dumpThread) {
  return inspectorPromCommInfoListsDump(commList, nullptr, output_root, dumpThread);
}

static void inspectorPromMergeStepProxyAgg(
    inspectorPromStepProxyAgg& target,
    const inspectorPromStepProxyAgg& source) {
  target.count += source.count;
  target.transferBytes += source.transferBytes;
  target.unknownTransferSizes += source.unknownTransferSizes;
  target.missingTransitions += source.missingTransitions;
  for (int phase = 0; phase < inspectorProxyWaitPhaseCount; phase++) {
    target.phaseCount[phase] += source.phaseCount[phase];
    target.phaseSumUsecs[phase] += source.phaseSumUsecs[phase];
    target.phaseMaxUsecs[phase] = std::max(
      target.phaseMaxUsecs[phase], source.phaseMaxUsecs[phase]);
  }
}

inspectorResult_t inspectorPromCommInfoListsDump(struct inspectorCommInfoList* first,
                                                 struct inspectorCommInfoList* second,
                                                 const char* output_root,
                                                 struct inspectorDumpThread* dumpThread) {
  inspectorResult_t res = inspectorSuccess;
  uint32_t processed = 0;
  uint32_t totalComms = 0;
  uint64_t currentTime = inspectorGetTime();
  std::map<std::string, inspectorPromDevice> devices;
  std::map<inspectorPromStepProxyKey,
           inspectorPromStepProxyAgg> detachedProxy;
  inspectorPromTakeStreamingDevices(devices, detachedProxy);

  struct inspectorCommInfoList* lists[] = {first, second};
  for (size_t i = 0; i < 2; i++) {
    struct inspectorCommInfoList* commList = lists[i];
    if (commList == nullptr) continue;
    INS_CHK_GOTO(inspectorLockRd(&commList->guard), res, exit);
    if (commList->ncomms > 0) {
      uint32_t listProcessed = 0;
      totalComms += commList->ncomms;
      inspectorResult_t fillResult
        = inspectorPromFillDeviceBuckets(commList, devices, &listProcessed);
      inspectorUnlockRWLock(&commList->guard);
      if (fillResult != inspectorSuccess) {
        res = fillResult;
        goto exit;
      }
      processed += listProcessed;
    } else {
      INS_CHK_GOTO(inspectorUnlockRWLock(&commList->guard), res, exit);
    }
  }

  if (!detachedProxy.empty()) {
    if (devices.empty()) {
      const char* worldRank = getenv("SLURM_PROCID");
      char deviceKey[128];
      snprintf(deviceKey, sizeof(deviceKey), "pxn-rank%s-pid%d",
               worldRank ? worldRank : "unknown", getpid());
      inspectorPromDevice& detachedDevice = devices[deviceKey];
      detachedDevice.deviceUuidStr = deviceKey;
      char nodeName[256];
      inspectorPromGetNodeName(nodeName, sizeof(nodeName));
      detachedDevice.nodeName = nodeName;
      detachedDevice.gpuName = "PXN";
    }
    inspectorPromDevice& device = devices.begin()->second;
    for (const auto& entry : detachedProxy) {
      inspectorPromMergeStepProxyAgg(device.stepProxy[entry.first], entry.second);
    }
    device.hasData = true;
  }

  if (!devices.empty()) {
    INS_CHK_GOTO(inspectorPromWriteDeviceBuckets(devices,
                                                 output_root,
                                                 currentTime,
                                                 dumpThread),
                 res, exit);
    TRACE_INSPECTOR(
      "NCCL Inspector: Completed dump across devices, flushed %u/%u communicators",
      processed, totalComms);
  }

exit:
  return res;
}

/*
 * Description:
 *
 *   Validates and adjusts the dump interval for Prometheus-specific requirements.
 *   Prometheus requires a minimum 30-second interval to match node exporter poll interval.
 *
 * Thread Safety:
 *   Thread-safe.
 *
 * Input:
 *   int64_t interval - raw interval in microseconds from environment variable (-1 = disabled, 0 = continuous, >0 = periodic).
 *
 * Output:
 *   None.
 *
 * Return:
 *   int64_t - validated interval in microseconds.
 */
int64_t inspectorPromValidateInterval(int64_t interval) {
  const int64_t MIN_PROM_INTERVAL = 30000000;

  if (interval < 0) {
    return interval;
  } else if (interval >= 0 && interval < MIN_PROM_INTERVAL) {
    INFO_INSPECTOR(
      "NCCL Inspector: Prometheus dump requires minimum interval of %ld microseconds "
      "to match node exporter poll interval, but got %ld. Setting to minimum.",
      MIN_PROM_INTERVAL, interval);
    return MIN_PROM_INTERVAL;
  }

  return interval;
}
