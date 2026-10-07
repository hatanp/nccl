#include "inspector_prom.h"
#include "nccl/profiler.h"

#include <assert.h>
#include <dirent.h>
#include <stdarg.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#include <fstream>
#include <map>
#include <sstream>
#include <string>

extern ncclProfiler_t ncclProfiler_v5;

static void testLogger(ncclDebugLogLevel, unsigned long, const char*, int,
                       const char*, ...) {}

static std::string findPromFile(const char* directory) {
  DIR* handle = opendir(directory);
  assert(handle != nullptr);
  std::string result;
  while (dirent* entry = readdir(handle)) {
    std::string name(entry->d_name);
    if (name.size() >= 5 && name.substr(name.size() - 5) == ".prom") {
      result = std::string(directory) + "/" + name;
      break;
    }
  }
  closedir(handle);
  return result;
}

static uint64_t field(const std::string& line, const char* name) {
  const std::string key = std::string("\"") + name + "\":";
  const size_t offset = line.find(key);
  assert(offset != std::string::npos);
  return strtoull(line.c_str() + offset + key.size(), nullptr, 10);
}

int main(int argc, char** argv) {
  assert(argc == 1 || (argc == 2 && (strcmp(argv[1], "off") == 0
    || strcmp(argv[1], "cap0") == 0 || strcmp(argv[1], "cap1") == 0
    || strcmp(argv[1], "missing") == 0)));
  const bool missingDescriptor = argc == 2 && strcmp(argv[1], "missing") == 0;
  const bool parentEnabled = argc == 1 || strcmp(argv[1], "off") != 0;
  const size_t parentCap = argc == 2 && strcmp(argv[1], "cap0") == 0 ? 0
    : argc == 2 && strcmp(argv[1], "cap1") == 0 ? 1 : 2;
  char outputTemplate[] = "/tmp/inspector-proxy-lifecycle-XXXXXX";
  char* outputDirectory = mkdtemp(outputTemplate);
  assert(outputDirectory != nullptr);
  setenv("NCCL_INSPECTOR_ENABLE", "1", 1);
  setenv("NCCL_INSPECTOR_ENABLE_P2P", "0", 1);
  setenv("NCCL_INSPECTOR_PROXY_STEP_ENABLE", "1", 1);
  setenv("NCCL_INSPECTOR_PROXY_PEAK_ENABLE", "1", 1);
  setenv("NCCL_INSPECTOR_PARENT_IDENTITY_ENABLE", parentEnabled ? "1" : "0", 1);
  setenv("NCCL_INSPECTOR_PARENT_DICTIONARY_CAPACITY", std::to_string(parentCap).c_str(), 1);
  setenv("NCCL_INSPECTOR_PROXY_OP_POOL_SIZE", "8", 1);
  setenv("NCCL_INSPECTOR_PROXY_STEP_POOL_SIZE", "8", 1);
  setenv("NCCL_INSPECTOR_DUMP_THREAD_ENABLE", "0", 1);
  setenv("NCCL_INSPECTOR_PROM_DUMP", "1", 1);
  setenv("NCCL_INSPECTOR_PROM_RETAIN", "1", 1);
  setenv("NCCL_INSPECTOR_DUMP_MIN_SIZE_BYTES", "0", 1);
  setenv("NCCL_INSPECTOR_REQUIRE_KERNEL_TIMING", "0", 1);
  setenv("NCCL_INSPECTOR_DUMP_DIR", outputDirectory, 1);
  setenv("NCCL_INSPECTOR_STEP_ENABLE", "1", 1);
  setenv("NCCL_INSPECTOR_WORLD_SIZE", "256", 1);
  setenv("NCCL_INSPECTOR_DP_SIZE", "64", 1);
  setenv("NCCL_INSPECTOR_EDP_SIZE", "2", 1);
  setenv("NCCL_INSPECTOR_EP_SIZE", "32", 1);
  setenv("NCCL_INSPECTOR_PP_SIZE", "4", 1);

  void* context = nullptr;
  int activationMask = 0;
  assert(ncclProfiler_v5.init(&context, 1234, &activationMask, "test", 2, 64,
                              0, testLogger) == ncclSuccess);
  assert(context != nullptr);
  assert((activationMask & ncclProfileProxyOp) != 0);
  assert((activationMask & ncclProfileProxyStep) != 0);
  assert(ncclInspectorStepBegin(7, 0) == 0);

  ncclProfilerEventDescr_t coll;
  memset(&coll, 0, sizeof(coll));
  coll.type = ncclProfileColl;
  coll.rank = 0;
  coll.coll.seqNumber = 1;
  coll.coll.func = "ReduceScatter";
  coll.coll.count = 1024;
  char originalDatatype[] = "ncclFloat32";
  coll.coll.datatype = originalDatatype;
  coll.coll.sendBuff = missingDescriptor ? nullptr : reinterpret_cast<void*>(0x1000);
  coll.coll.recvBuff = reinterpret_cast<void*>(0x2000);
  coll.coll.nChannels = 1;
  coll.coll.algo = "RING";
  coll.coll.proto = "SIMPLE";
  void* collHandle = nullptr;
  assert(ncclProfiler_v5.startEvent(context, &collHandle, &coll) == ncclSuccess);
  assert(collHandle != nullptr);
  const uint64_t firstParentId = static_cast<inspectorCollInfo*>(collHandle)->parentIdentity.id;
  assert((firstParentId != 0) == parentEnabled);
  // Descriptor values/string may be changed after start; retained metadata must
  // still describe the original callback, never these replacements.
  strcpy(originalDatatype, "ncclInt8");
  coll.coll.sendBuff = reinterpret_cast<void*>(0xdead);
  coll.coll.recvBuff = reinterpret_cast<void*>(0xbeef);
  coll.coll.count = 2048;

  ncclProfilerEventDescr_t proxyOp;
  memset(&proxyOp, 0, sizeof(proxyOp));
  proxyOp.type = ncclProfileProxyOp;
  proxyOp.parentObj = collHandle;
  proxyOp.rank = 0;
  proxyOp.proxyOp.pid = getpid();
  proxyOp.proxyOp.channelId = 0;
  proxyOp.proxyOp.peer = 1;
  proxyOp.proxyOp.nSteps = 1;
  proxyOp.proxyOp.chunkSize = 4096;
  proxyOp.proxyOp.isSend = 1;
  void* proxyOpHandle = nullptr;
  assert(ncclProfiler_v5.startEvent(context, &proxyOpHandle, &proxyOp)
         == ncclSuccess);
  assert(proxyOpHandle != nullptr);

  ncclProfilerEventDescr_t kernel;
  memset(&kernel, 0, sizeof(kernel));
  kernel.type = ncclProfileKernelCh;
  kernel.parentObj = collHandle;
  kernel.rank = 0;
  kernel.kernelCh.channelId = 0;
  kernel.kernelCh.pTimer = 1000000;
  void* kernelHandle = nullptr;
  assert(ncclProfiler_v5.startEvent(context, &kernelHandle, &kernel)
         == ncclSuccess);
  assert(kernelHandle != nullptr);
  ncclProfilerEventStateArgs_t kernelStateArgs;
  memset(&kernelStateArgs, 0, sizeof(kernelStateArgs));
  kernelStateArgs.kernelCh.pTimer = 2000000;
  assert(ncclProfiler_v5.recordEventState(
           kernelHandle, ncclProfilerKernelChStop, &kernelStateArgs)
         == ncclSuccess);
  assert(ncclProfiler_v5.stopEvent(kernelHandle) == ncclSuccess);

  ncclProfilerEventDescr_t detachedProxyOp = proxyOp;
  detachedProxyOp.parentObj = reinterpret_cast<void*>(1);
  detachedProxyOp.proxyOp.pid = getpid() + 1;
  void* detachedProxyOpHandle = reinterpret_cast<void*>(1);
  assert(ncclProfiler_v5.startEvent(
           reinterpret_cast<void*>(1), &detachedProxyOpHandle, &detachedProxyOp) == ncclSuccess);
  assert(detachedProxyOpHandle != nullptr);
  ncclProfilerEventDescr_t detachedProxyStep;
  memset(&detachedProxyStep, 0, sizeof(detachedProxyStep));
  detachedProxyStep.type = ncclProfileProxyStep;
  detachedProxyStep.parentObj = detachedProxyOpHandle;
  detachedProxyStep.rank = 0;
  detachedProxyStep.proxyStep.step = 0;
  void* detachedProxyStepHandle = nullptr;
  assert(ncclProfiler_v5.startEvent(
           context, &detachedProxyStepHandle, &detachedProxyStep) == ncclSuccess);
  assert(detachedProxyStepHandle != nullptr);
  ncclProfilerEventStateArgs_t detachedStateArgs;
  memset(&detachedStateArgs, 0, sizeof(detachedStateArgs));
  assert(ncclProfiler_v5.recordEventState(
           detachedProxyStepHandle, ncclProfilerProxyStepSendGPUWait,
           &detachedStateArgs) == ncclSuccess);
  assert(ncclProfiler_v5.recordEventState(
           detachedProxyStepHandle, ncclProfilerProxyStepSendPeerWait_v4,
           &detachedStateArgs) == ncclSuccess);
  detachedStateArgs.proxyStep.transSize = 4096;
  assert(ncclProfiler_v5.recordEventState(
           detachedProxyStepHandle, ncclProfilerProxyStepSendWait,
           &detachedStateArgs) == ncclSuccess);
  assert(ncclProfiler_v5.stopEvent(detachedProxyStepHandle) == ncclSuccess);
  assert(ncclProfiler_v5.stopEvent(detachedProxyOpHandle) == ncclSuccess);

  // The proxy operation must retain its local collective parent even after
  // NCCL stops the task event. Force a pool allocation between parent stop
  // and proxy stop so a missing proxy reference deterministically reuses it.
  assert(ncclProfiler_v5.stopEvent(collHandle) == ncclSuccess);
  ncclProfilerEventDescr_t replacementColl = coll;
  replacementColl.coll.seqNumber = 1; // Same function/sequence/count, different buffers.
  replacementColl.coll.func = "ReduceScatter";
  replacementColl.coll.count = 1024;
  replacementColl.coll.datatype = "ncclFloat32";
  replacementColl.coll.sendBuff = reinterpret_cast<void*>(0x3000);
  replacementColl.coll.recvBuff = reinterpret_cast<void*>(0x4000);
  replacementColl.coll.nChannels = 0;
  void* replacementCollHandle = nullptr;
  assert(ncclProfiler_v5.startEvent(
           context, &replacementCollHandle, &replacementColl) == ncclSuccess);
  assert(replacementCollHandle != nullptr);
  assert(replacementCollHandle != collHandle);
  const uint64_t secondParentId = static_cast<inspectorCollInfo*>(replacementCollHandle)->parentIdentity.id;
  if (parentEnabled) assert(secondParentId > firstParentId);
  ncclProfilerEventDescr_t secondProxyOp = proxyOp;
  secondProxyOp.parentObj = replacementCollHandle;
  secondProxyOp.proxyOp.isSend = 0; // Distinct existing direction bin retains both witnesses.
  void* secondProxyHandle = nullptr;
  assert(ncclProfiler_v5.startEvent(context, &secondProxyHandle, &secondProxyOp) == ncclSuccess);
  assert(secondProxyHandle != nullptr);
  ncclProfilerEventDescr_t secondStep{};
  secondStep.type = ncclProfileProxyStep;
  secondStep.parentObj = secondProxyHandle;
  void* secondStepHandle = nullptr;
  assert(ncclProfiler_v5.startEvent(context, &secondStepHandle, &secondStep) == ncclSuccess);
  assert(secondStepHandle != nullptr);
  ncclProfilerEventStateArgs_t secondArgs{};
  secondArgs.proxyStep.transSize = 4096;
  assert(ncclProfiler_v5.recordEventState(secondStepHandle, ncclProfilerProxyStepRecvWait, &secondArgs) == ncclSuccess);
  usleep(1000);
  assert(ncclProfiler_v5.recordEventState(secondStepHandle, ncclProfilerProxyStepRecvFlushWait, &secondArgs) == ncclSuccess);
  usleep(1000);
  assert(ncclProfiler_v5.recordEventState(secondStepHandle, ncclProfilerProxyStepRecvGPUWait, &secondArgs) == ncclSuccess);
  usleep(1000);
  assert(ncclProfiler_v5.stopEvent(secondStepHandle) == ncclSuccess);
  assert(ncclProfiler_v5.stopEvent(secondProxyHandle) == ncclSuccess);

  ncclProfilerEventDescr_t proxyStep;
  memset(&proxyStep, 0, sizeof(proxyStep));
  proxyStep.type = ncclProfileProxyStep;
  proxyStep.parentObj = proxyOpHandle;
  proxyStep.rank = 0;
  proxyStep.proxyStep.step = 0;
  void* proxyStepHandle = nullptr;
  assert(ncclProfiler_v5.startEvent(context, &proxyStepHandle, &proxyStep)
         == ncclSuccess);
  assert(proxyStepHandle != nullptr);

  ncclProfilerEventStateArgs_t stateArgs;
  memset(&stateArgs, 0, sizeof(stateArgs));
  usleep(1000);
  assert(ncclProfiler_v5.recordEventState(
           proxyStepHandle, ncclProfilerProxyStepSendGPUWait, &stateArgs)
         == ncclSuccess);
  usleep(1000);
  assert(ncclProfiler_v5.recordEventState(
           proxyStepHandle, ncclProfilerProxyStepSendPeerWait_v4, &stateArgs)
         == ncclSuccess);
  usleep(1000);
  stateArgs.proxyStep.transSize = 4096;
  assert(ncclProfiler_v5.recordEventState(
           proxyStepHandle, ncclProfilerProxyStepSendWait, &stateArgs)
         == ncclSuccess);
  stateArgs.proxyStep.transSize = SIZE_MAX;
  assert(ncclProfiler_v5.recordEventState(
           proxyStepHandle, ncclProfilerProxyStepSendWait, &stateArgs)
         == ncclSuccess);
  usleep(1000);
  assert(ncclProfiler_v5.stopEvent(proxyStepHandle) == ncclSuccess);
  assert(ncclProfiler_v5.stopEvent(proxyOpHandle) == ncclSuccess);
  assert(ncclProfiler_v5.stopEvent(replacementCollHandle) == ncclSuccess);
  // A DP-sized communicator does not establish AllReduce semantics. Use a
  // separate fixed descriptor and GPU interval to verify the emitted step row,
  // while keeping the existing ReduceScatter and detached-PXN fixture intact.
  ncclProfilerEventDescr_t ambiguousColl = coll;
  ambiguousColl.coll.seqNumber = 3;
  ambiguousColl.coll.func = "AllReduce";
  ambiguousColl.coll.datatype = "ncclFloat32";
  ambiguousColl.coll.count = 463339520; // 1,853,358,080 bytes of float32.
  void* ambiguousCollHandle = nullptr;
  assert(ncclProfiler_v5.startEvent(
           context, &ambiguousCollHandle, &ambiguousColl) == ncclSuccess);
  assert(ambiguousCollHandle != nullptr);
  assert(ambiguousCollHandle == replacementCollHandle); // Free-list address reused.
  if (parentEnabled) {
    assert(static_cast<inspectorCollInfo*>(ambiguousCollHandle)->parentIdentity.id > secondParentId);
  }
  ncclProfilerEventDescr_t ambiguousKernel = kernel;
  ambiguousKernel.parentObj = ambiguousCollHandle;
  ambiguousKernel.kernelCh.pTimer = 3000000;
  void* ambiguousKernelHandle = nullptr;
  assert(ncclProfiler_v5.startEvent(
           context, &ambiguousKernelHandle, &ambiguousKernel) == ncclSuccess);
  assert(ambiguousKernelHandle != nullptr);
  kernelStateArgs.kernelCh.pTimer = 4000000;
  assert(ncclProfiler_v5.recordEventState(
           ambiguousKernelHandle, ncclProfilerKernelChStop, &kernelStateArgs)
         == ncclSuccess);
  assert(ncclProfiler_v5.stopEvent(ambiguousKernelHandle) == ncclSuccess);
  assert(ncclProfiler_v5.stopEvent(ambiguousCollHandle) == ncclSuccess);
  assert(ncclInspectorStepEnd(7, 0) == 0);
  assert(ncclProfiler_v5.finalize(context) == ncclSuccess);

  std::string path = findPromFile(outputDirectory);
  assert(!path.empty());
  std::ifstream input(path);
  std::stringstream buffer;
  buffer << input.rdbuf();
  std::string output = buffer.str();
  assert(output.find("# nccl_inspector_step_proxy_info") != std::string::npos);
  assert(output.find("# nccl_inspector_step_proxy_peak {") != std::string::npos);
  assert(output.find("\"identity_known\":true") != std::string::npos);
  assert(output.find("\"identity_known\":false") != std::string::npos);
  assert(output.find("\"sequence\":1,\"channel\":0,\"peer\":1") != std::string::npos);
  assert(output.find("\"clock\":\"host_gettimeofday_us\"") != std::string::npos);
  assert(output.find("\"retention\":\"max_per_phase_per_aggregate\"") != std::string::npos);

  assert(output.find("\"dropped_ops\":0") != std::string::npos);
  assert(output.find("\"dropped_steps\":0") != std::string::npos);
  assert(output.find("\"detached_ops\":1") != std::string::npos);
  assert(output.find("\"world_size\":256") != std::string::npos);
  assert(output.find("\"dp_size\":64") != std::string::npos);
  assert(output.find("\"edp_size\":2") != std::string::npos);
  assert(output.find("\"ep_size\":32") != std::string::npos);
  assert(output.find("\"pp_size\":4") != std::string::npos);
  assert(output.find("\"family\":\"pxn\"") != std::string::npos);
  assert(output.find("\"step\":7") != std::string::npos);
  assert(output.find("\"family\":\"dp\"") != std::string::npos);
  assert(output.find("\"operation\":\"ReduceScatter\"") != std::string::npos);
  assert(output.find("\"direction\":\"send\"") != std::string::npos);
  assert(output.find("\"comm_id\":\"0x4d2\"") != std::string::npos);
  assert(output.find("\"comm_name\":\"test\"") != std::string::npos);
  assert(output.find("\"comm_rank\":0") != std::string::npos);
  assert(output.find("\"nranks\":64") != std::string::npos);
  assert(output.find("\"n_nodes\":2") != std::string::npos);
  assert(output.find("\"count\":1") != std::string::npos);
  assert(output.find("\"transfer_bytes\":4096") != std::string::npos);
  assert(output.find("\"unknown_transfer_sizes\":1") != std::string::npos);
  assert(output.find("\"missing_transitions\":0") != std::string::npos);
  assert(output.find("\"phase_count\":[1,1,1,0,0,0]") != std::string::npos);
  assert(output.find("# nccl_inspector_step {") != std::string::npos);
  assert(output.find("\"gpu_interval_count\":1") != std::string::npos);
  assert(output.find("\"gpu_first_start_ns\":1000000") != std::string::npos);
  assert(output.find("\"gpu_last_stop_ns\":2000000") != std::string::npos);
  assert(output.find("\"gpu_union_us\":1000") != std::string::npos);
  assert(output.find("\"gpu_envelope_us\":1000") != std::string::npos);
  assert(output.find("\"gpu_merged_interval_count\":1") != std::string::npos);
  assert(output.find("\"gpu_intervals_ns\":[[1000000,2000000]]")
         != std::string::npos);

  assert(output.find("version=\"v5.9\"") != std::string::npos);
  const std::string ambiguousPrefix =
    "# nccl_inspector_step {\"step\":7,\"family\":\"unknown\","
    "\"size_family_hint\":\"dp\",\"operation\":\"AllReduce\"";
  const size_t ambiguousStart = output.find(ambiguousPrefix);
  assert(ambiguousStart != std::string::npos);
  const size_t ambiguousEnd = output.find('\n', ambiguousStart);
  assert(ambiguousEnd != std::string::npos);
  const std::string ambiguousRow = output.substr(
    ambiguousStart, ambiguousEnd - ambiguousStart);
  // Restrict assertions to this row: the original RS row must not satisfy them.
  assert(ambiguousRow.find("\"count\":1,") != std::string::npos);
  assert(ambiguousRow.find("\"sum_us\":") != std::string::npos);
  assert(ambiguousRow.find("\"max_us\":") != std::string::npos);
  assert(ambiguousRow.find("\"first_start_us\":") != std::string::npos);
  assert(ambiguousRow.find("\"last_stop_us\":") != std::string::npos);
  assert(ambiguousRow.find("\"gpu_interval_count\":1,") != std::string::npos);
  assert(ambiguousRow.find("\"gpu_first_start_ns\":3000000,")
         != std::string::npos);
  assert(ambiguousRow.find("\"gpu_last_stop_ns\":4000000,")
         != std::string::npos);
  assert(ambiguousRow.find("\"gpu_union_us\":1000,") != std::string::npos);
  assert(ambiguousRow.find("\"gpu_envelope_us\":1000,") != std::string::npos);
  assert(ambiguousRow.find("\"gpu_merged_interval_count\":1,")
         != std::string::npos);
  assert(ambiguousRow.find("\"gpu_intervals_ns\":[[3000000,4000000]]")
         != std::string::npos);

  const size_t expectedDictionaryEntries = parentEnabled
    ? parentCap - (missingDescriptor && parentCap > 0 ? 1 : 0) : 0;
  std::istringstream lines(output);
  std::map<uint64_t, std::string> parentRows;
  std::string line;
  unsigned localPeakRecords = 0;
  unsigned unavailableLocalPeaks = 0;
  uint64_t reportedUnavailable = 0;
  while (std::getline(lines, line)) {
    if (line.find("# nccl_inspector_proxy_parent ") == 0) {
      const uint64_t id = field(line, "parent_id");
      assert(parentRows.emplace(id, line).second);
      assert(line.find("\"native_count\":1024") != std::string::npos);
      assert(line.find("\"native_datatype\":\"ncclFloat32\"") != std::string::npos);
    } else if (line.find("# nccl_inspector_step_proxy_info ") == 0) {
      reportedUnavailable = field(line, "parent_metadata_unavailable_peak_records");
      assert(field(line, "parent_dictionary_entries") == expectedDictionaryEntries);
    } else if (line.find("# nccl_inspector_step_proxy_peak ") == 0) {
      const uint64_t id = field(line, "parent_id");
      const bool metadataKnown = line.find("\"parent_metadata_known\":true") != std::string::npos;
      if (line.find("\"identity_known\":false") != std::string::npos) {
        assert(id == 0 && !metadataKnown); // Invalid foreign pointers never produce metadata.
      } else {
        localPeakRecords++;
        assert(id == firstParentId || id == secondParentId);
        const bool expectedKnown = parentEnabled && parentCap > 0
          && (parentCap > 1 || id == firstParentId)
          && !(missingDescriptor && id == firstParentId);
        assert(metadataKnown == expectedKnown);
        if (!metadataKnown) unavailableLocalPeaks++;
      }
    }
  }
  assert(localPeakRecords == 6);
  assert(parentRows.size() == expectedDictionaryEntries);
  assert(reportedUnavailable == (parentEnabled ? unavailableLocalPeaks : 0));
  if (parentEnabled && parentCap > 0 && !missingDescriptor) {
    assert(parentRows.at(firstParentId).find("\"send_buffer\":\"0x1000\"") != std::string::npos);
    assert(parentRows.at(firstParentId).find("\"recv_buffer\":\"0x2000\"") != std::string::npos);
  }
  if (parentEnabled && parentCap > 1) {
    assert(parentRows.at(secondParentId).find("\"send_buffer\":\"0x3000\"") != std::string::npos);
    assert(parentRows.at(secondParentId).find("\"recv_buffer\":\"0x4000\"") != std::string::npos);
  }

  unlink(path.c_str());
  rmdir(outputDirectory);
  return 0;
}
