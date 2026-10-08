// NCCL 2.31 order: the profiler thread completes a collective's kernel (and
// the Inspector releases its record) before the proxy thread appends the
// collective's network operation. Modes:
//   late    - the proxy starts inside the quarantine window: it keeps the
//             parent's identity and step, and nothing is released twice;
//   expired - the proxy starts after the window: its phases are kept without
//             parent identity, in the step current at its start.
#include "inspector_event_pool.h"
#include "inspector_prom.h"
#include "nccl/profiler.h"

#include <assert.h>
#include <dirent.h>
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#include <fstream>
#include <sstream>
#include <string>

extern ncclProfiler_t ncclProfiler_v5;

static int gDoubleReleases = 0;

static void testLogger(ncclDebugLogLevel, unsigned long, const char*, int,
                       const char* format, ...) {
  char message[512];
  va_list args;
  va_start(args, format);
  vsnprintf(message, sizeof(message), format, args);
  va_end(args);
  if (strstr(message, "Double release") != nullptr) gDoubleReleases++;
}

static std::string readPromFile(const char* directory) {
  DIR* handle = opendir(directory);
  assert(handle != nullptr);
  std::string path;
  while (dirent* entry = readdir(handle)) {
    std::string name(entry->d_name);
    if (name.size() >= 5 && name.substr(name.size() - 5) == ".prom") {
      path = std::string(directory) + "/" + name;
      break;
    }
  }
  closedir(handle);
  assert(!path.empty());
  std::ifstream input(path);
  std::stringstream buffer;
  buffer << input.rdbuf();
  return buffer.str();
}

static std::string sendWaitPeak(const std::string& output) {
  std::istringstream lines(output);
  std::string line;
  while (std::getline(lines, line)) {
    if (line.find("# nccl_inspector_step_proxy_peak ") == 0
        && line.find("\"phase\":\"send_wait\"") != std::string::npos) return line;
  }
  return "";
}

int main(int argc, char** argv) {
  assert(argc == 2 && (strcmp(argv[1], "late") == 0 || strcmp(argv[1], "expired") == 0));
  const bool expired = strcmp(argv[1], "expired") == 0;
  char outputTemplate[] = "/tmp/inspector-proxy-late-parent-XXXXXX";
  char* outputDirectory = mkdtemp(outputTemplate);
  assert(outputDirectory != nullptr);
  setenv("NCCL_INSPECTOR_ENABLE", "1", 1);
  setenv("NCCL_INSPECTOR_ENABLE_P2P", "0", 1);
  setenv("NCCL_INSPECTOR_PROXY_STEP_ENABLE", "1", 1);
  setenv("NCCL_INSPECTOR_PROXY_PEAK_ENABLE", "1", 1);
  setenv("NCCL_INSPECTOR_PARENT_IDENTITY_ENABLE", "1", 1);
  setenv("NCCL_INSPECTOR_PARENT_DICTIONARY_CAPACITY", "2", 1);
  setenv("NCCL_INSPECTOR_POOL_QUARANTINE_MS", expired ? "1" : "1000", 1);
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
  assert(ncclInspectorStepBegin(5, 0) == 0);

  ncclProfilerEventDescr_t coll;
  memset(&coll, 0, sizeof(coll));
  coll.type = ncclProfileColl;
  coll.coll.seqNumber = 11;
  coll.coll.func = "ReduceScatter";
  coll.coll.count = 1024;
  coll.coll.datatype = "ncclFloat32";
  coll.coll.sendBuff = reinterpret_cast<void*>(0x1000);
  coll.coll.recvBuff = reinterpret_cast<void*>(0x2000);
  coll.coll.nChannels = 1;
  coll.coll.algo = "RING";
  coll.coll.proto = "SIMPLE";
  void* collHandle = nullptr;
  assert(ncclProfiler_v5.startEvent(context, &collHandle, &coll) == ncclSuccess);
  assert(collHandle != nullptr);

  // The host callback stops the task event; the profiler thread then sees the
  // kernel finish, which releases the record.
  assert(ncclProfiler_v5.stopEvent(collHandle) == ncclSuccess);
  ncclProfilerEventDescr_t kernel;
  memset(&kernel, 0, sizeof(kernel));
  kernel.type = ncclProfileKernelCh;
  kernel.parentObj = collHandle;
  kernel.kernelCh.channelId = 0;
  kernel.kernelCh.pTimer = 1000000;
  void* kernelHandle = nullptr;
  assert(ncclProfiler_v5.startEvent(context, &kernelHandle, &kernel) == ncclSuccess);
  ncclProfilerEventStateArgs_t kernelState;
  memset(&kernelState, 0, sizeof(kernelState));
  kernelState.kernelCh.pTimer = 2000000;
  assert(ncclProfiler_v5.recordEventState(kernelHandle, ncclProfilerKernelChStop, &kernelState)
         == ncclSuccess);
  assert(ncclProfiler_v5.stopEvent(kernelHandle) == ncclSuccess);
  assert(g_eventPool.collAllocCount == 0);
  assert(ncclInspectorStepEnd(5, 0) == 0);
  assert(ncclInspectorStepBegin(6, 0) == 0);
  if (expired) usleep(20000);

  // The proxy thread appends the send only now.
  ncclProfilerEventDescr_t proxyOp;
  memset(&proxyOp, 0, sizeof(proxyOp));
  proxyOp.type = ncclProfileProxyOp;
  proxyOp.parentObj = collHandle;
  proxyOp.proxyOp.pid = getpid();
  proxyOp.proxyOp.channelId = 0;
  proxyOp.proxyOp.peer = 1;
  proxyOp.proxyOp.nSteps = 1;
  proxyOp.proxyOp.chunkSize = 4096;
  proxyOp.proxyOp.isSend = 1;
  void* proxyOpHandle = nullptr;
  assert(ncclProfiler_v5.startEvent(context, &proxyOpHandle, &proxyOp) == ncclSuccess);
  assert(proxyOpHandle != nullptr);
  ncclProfilerEventDescr_t proxyStep;
  memset(&proxyStep, 0, sizeof(proxyStep));
  proxyStep.type = ncclProfileProxyStep;
  proxyStep.parentObj = proxyOpHandle;
  void* proxyStepHandle = nullptr;
  assert(ncclProfiler_v5.startEvent(context, &proxyStepHandle, &proxyStep) == ncclSuccess);
  assert(proxyStepHandle != nullptr);
  ncclProfilerEventStateArgs_t stepState;
  memset(&stepState, 0, sizeof(stepState));
  assert(ncclProfiler_v5.recordEventState(proxyStepHandle, ncclProfilerProxyStepSendGPUWait,
                                          &stepState) == ncclSuccess);
  usleep(1000);
  assert(ncclProfiler_v5.recordEventState(proxyStepHandle, ncclProfilerProxyStepSendPeerWait_v4,
                                          &stepState) == ncclSuccess);
  usleep(1000);
  stepState.proxyStep.transSize = 4096;
  assert(ncclProfiler_v5.recordEventState(proxyStepHandle, ncclProfilerProxyStepSendWait,
                                          &stepState) == ncclSuccess);
  usleep(1000);
  assert(ncclProfiler_v5.stopEvent(proxyStepHandle) == ncclSuccess);
  assert(ncclProfiler_v5.stopEvent(proxyOpHandle) == ncclSuccess);
  assert(g_eventPool.collAllocCount == 0);
  assert(ncclInspectorStepEnd(6, 0) == 0);
  assert(ncclProfiler_v5.finalize(context) == ncclSuccess);
  assert(gDoubleReleases == 0);

  const std::string output = readPromFile(outputDirectory);
  const std::string peak = sendWaitPeak(output);
  assert(!peak.empty());
  if (expired) {
    assert(output.find("\"parent_late_ops\":0,") != std::string::npos);
    assert(output.find("\"parent_expired_ops\":1,") != std::string::npos);
    assert(output.find("\"pool_quarantine_ms\":1,") != std::string::npos);
    assert(peak.find("\"step\":6,") != std::string::npos);
    assert(peak.find("\"identity_known\":false") != std::string::npos);
  } else {
    assert(output.find("\"parent_late_ops\":1,") != std::string::npos);
    assert(output.find("\"parent_expired_ops\":0,") != std::string::npos);
    assert(output.find("\"pool_quarantine_ms\":1000,") != std::string::npos);
    assert(peak.find("\"step\":5,") != std::string::npos);
    assert(peak.find("\"identity_known\":true") != std::string::npos);
    assert(peak.find("\"sequence\":11,\"channel\":0,\"peer\":1") != std::string::npos);
    assert(peak.find("\"parent_metadata_known\":true") != std::string::npos);
  }
  printf("proxy late-parent test passed (%s)\n", argv[1]);
  return 0;
}
