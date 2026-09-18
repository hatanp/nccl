/*************************************************************************
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * See LICENSE.txt for more license information
 *************************************************************************/

#include "inspector_prom.h"
#include "inspector.h"
#include "inspector_cudawrap.h"
#include "inspector_ring.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <inttypes.h>
#include <algorithm>
#include <map>
#include <limits>
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

using inspectorPromCollBucketMap = std::map<inspectorPromCollBucketKey, inspectorPromBucketAgg>;
using inspectorPromP2pBucketMap = std::map<inspectorPromP2pBucketKey, inspectorPromBucketAgg>;

struct inspectorPromDevice {
  std::string deviceUuidStr;
  std::string nodeName;
  std::string gpuName;
  FILE* file = nullptr;
  inspectorPromCollBucketMap collBuckets;
  inspectorPromP2pBucketMap p2pBuckets;
  uint64_t collOverwritten = 0;
  uint64_t p2pOverwritten = 0;
  bool hasData = false;
};
static const int kInspectorPromFormatMinor = 2;

static void inspectorPromAggUpdate(inspectorPromBucketAgg& agg,
                                   const inspectorCompletedOpInfo& op) {
  agg.count++;
  agg.execTimeSum += static_cast<double>(op.execTimeUsecs);
  agg.execTimeMin = std::min(agg.execTimeMin, op.execTimeUsecs);
  agg.execTimeMax = std::max(agg.execTimeMax, op.execTimeUsecs);
  agg.execTimes.push_back(op.execTimeUsecs);

  inspectorPromSlowEvent event;
  event.sequence = op.sn;
  event.execTimeUsecs = op.execTimeUsecs;
  event.startTimestampUsecs = op.evtTrk.evntTrace[NCCL_INSP_EVT_TRK_OP_START].ts;
  event.stopTimestampUsecs = op.evtTrk.evntTrace[NCCL_INSP_EVT_TRK_OP_STOP].ts;
  event.peer = op.peer;
  agg.slowest.push_back(event);
  std::sort(agg.slowest.begin(), agg.slowest.end(),
            [](const inspectorPromSlowEvent& lhs, const inspectorPromSlowEvent& rhs) {
              return lhs.execTimeUsecs > rhs.execTimeUsecs;
            });
  if (agg.slowest.size() > 4) agg.slowest.resize(4);

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
  snprintf(output, outputSize,
           "%s/nccl_inspector_metrics_%s.prom",
           baseFilename, deviceUuidStr);

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
  int written = snprintf(buffer, sizeof(buffer),
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

  if (written < 0 || (size_t)written >= sizeof(buffer)) {
    return inspectorMemoryError;
  }

  if (fwrite(buffer, 1, written, file) != (size_t)written) {
    return inspectorFileOpenError;
  }

  for (size_t i = 0; i < agg.slowest.size(); i++) {
    const inspectorPromSlowEvent& event = agg.slowest[i];
    written = snprintf(buffer, sizeof(buffer),
                       "nccl_collective_slow_event_exec_time_microseconds{%s,"
                       "top_index=\"%zu\",sequence=\"%" PRIu64 "\","
                       "start_timestamp_us=\"%" PRIu64 "\","
                       "stop_timestamp_us=\"%" PRIu64 "\"} %" PRIu64 "\n",
                       labels, i, event.sequence, event.startTimestampUsecs,
                       event.stopTimestampUsecs, event.execTimeUsecs);
    if (written < 0 || (size_t)written >= sizeof(buffer)) return inspectorMemoryError;
    if (fwrite(buffer, 1, written, file) != (size_t)written) return inspectorFileOpenError;
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
  int written = snprintf(buffer, sizeof(buffer),
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

  if (written < 0 || (size_t)written >= sizeof(buffer)) {
    return inspectorMemoryError;
  }

  if (fwrite(buffer, 1, written, file) != (size_t)written) {
    return inspectorFileOpenError;
  }

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
    if (written < 0 || (size_t)written >= sizeof(buffer)) return inspectorMemoryError;
    if (fwrite(buffer, 1, written, file) != (size_t)written) return inspectorFileOpenError;
  }

  fflush(file);
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
                                                       inspectorPromCollBucketMap& buckets,
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
      inspectorPromAggUpdate(buckets[key], collInfo);
    }
  }

  return inspectorSuccess;
}

static inspectorResult_t inspectorPromCommInfoDumpP2p(struct inspectorCommInfo* commInfo,
                                                      inspectorPromP2pBucketMap& buckets,
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
      inspectorPromAggUpdate(buckets[key], p2pInfo);
    }
  }

  return inspectorSuccess;
}

static inspectorResult_t inspectorPromCommInfoDump(struct inspectorCommInfo* commInfo,
                                                   inspectorPromCollBucketMap& collBuckets,
                                                   inspectorPromP2pBucketMap& p2pBuckets,
                                                   uint64_t* collOverwritten,
                                                   uint64_t* p2pOverwritten,
                                                   bool* needs_writing) {
  *needs_writing = false;

  INS_CHK(inspectorPromCommInfoDumpColl(commInfo, collBuckets,
                                        collOverwritten, needs_writing));
  INS_CHK(inspectorPromCommInfoDumpP2p(commInfo, p2pBuckets,
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
    if (device.deviceUuidStr.empty()) {
      device.deviceUuidStr = deviceKey;
    }
    if (device.nodeName.empty()) {
      char nodeName[256];
      inspectorPromGetNodeName(nodeName, sizeof(nodeName));
      device.nodeName = nodeName;
    }
    if (device.gpuName.empty()) {
      char gpuName[16];
      snprintf(gpuName, sizeof(gpuName), "GPU%d", itr->cudaDeviceId);
      device.gpuName = gpuName;
    }

    INS_CHK(inspectorPromCommInfoDump(itr,
                                      device.collBuckets,
                                      device.p2pBuckets,
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

inspectorResult_t inspectorPromCommInfoListsDump(struct inspectorCommInfoList* first,
                                                 struct inspectorCommInfoList* second,
                                                 const char* output_root,
                                                 struct inspectorDumpThread* dumpThread) {
  inspectorResult_t res = inspectorSuccess;
  uint32_t processed = 0;
  uint32_t totalComms = 0;
  uint64_t currentTime = inspectorGetTime();
  std::map<std::string, inspectorPromDevice> devices;

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
