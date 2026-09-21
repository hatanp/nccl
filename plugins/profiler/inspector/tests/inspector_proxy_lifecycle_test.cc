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

int main() {
  char outputTemplate[] = "/tmp/inspector-proxy-lifecycle-XXXXXX";
  char* outputDirectory = mkdtemp(outputTemplate);
  assert(outputDirectory != nullptr);
  setenv("NCCL_INSPECTOR_ENABLE", "1", 1);
  setenv("NCCL_INSPECTOR_ENABLE_P2P", "0", 1);
  setenv("NCCL_INSPECTOR_PROXY_STEP_ENABLE", "1", 1);
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
  coll.coll.datatype = "float32";
  coll.coll.nChannels = 0;
  coll.coll.algo = "RING";
  coll.coll.proto = "SIMPLE";
  void* collHandle = nullptr;
  assert(ncclProfiler_v5.startEvent(context, &collHandle, &coll) == ncclSuccess);
  assert(collHandle != nullptr);

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
  usleep(1000);
  assert(ncclProfiler_v5.stopEvent(proxyStepHandle) == ncclSuccess);
  assert(ncclProfiler_v5.stopEvent(proxyOpHandle) == ncclSuccess);
  assert(ncclProfiler_v5.stopEvent(collHandle) == ncclSuccess);
  assert(ncclInspectorStepEnd(7, 0) == 0);
  assert(ncclProfiler_v5.finalize(context) == ncclSuccess);

  std::string path = findPromFile(outputDirectory);
  assert(!path.empty());
  std::ifstream input(path);
  std::stringstream buffer;
  buffer << input.rdbuf();
  std::string output = buffer.str();
  assert(output.find("# nccl_inspector_step_proxy_info") != std::string::npos);
  assert(output.find("\"dropped_ops\":0") != std::string::npos);
  assert(output.find("\"dropped_steps\":0") != std::string::npos);
  assert(output.find("\"step\":7") != std::string::npos);
  assert(output.find("\"family\":\"dp\"") != std::string::npos);
  assert(output.find("\"operation\":\"ReduceScatter\"") != std::string::npos);
  assert(output.find("\"direction\":\"send\"") != std::string::npos);
  assert(output.find("\"count\":1") != std::string::npos);
  assert(output.find("\"transfer_bytes\":4096") != std::string::npos);
  assert(output.find("\"missing_transitions\":0") != std::string::npos);
  assert(output.find("\"phase_count\":[1,1,1,0,0,0]") != std::string::npos);

  unlink(path.c_str());
  rmdir(outputDirectory);
  return 0;
}
