#ifndef NCCL_INSPECTOR_THREAD_NAME_H_
#define NCCL_INSPECTOR_THREAD_NAME_H_

#include <pthread.h>
#include <stdlib.h>
#include <string.h>

// NCCL 2.31 delivers KernelCh events from a per-communicator profiler thread
// (ncclProfilerThreadFunc) that it leaves unnamed, unlike its proxy threads.
// The first KernelCh start on a thread names it, as NCCL_SET_THREAD_NAME=1
// does for NCCL's own threads, so per-thread samplers can attribute its CPU
// time. A thread that already carries an NCCL name keeps it (NCCL 2.30
// delivered KernelCh from the proxy progress thread).
#define INSPECTOR_PROFILER_THREAD_NAME "NCCL Profiler"

static inline bool inspectorShouldNameThread(const char* setThreadNameEnv,
                                             const char* currentName) {
  if (setThreadNameEnv == nullptr || strtol(setThreadNameEnv, nullptr, 0) != 1) return false;
  return strncmp(currentName, "NCCL", 4) != 0;
}

#ifdef __linux__
static inline void inspectorNameProfilerThreadOnce() {
  static thread_local bool checked = false;
  if (checked) return;
  checked = true;
  char current[16] = {0};
  if (pthread_getname_np(pthread_self(), current, sizeof(current)) != 0) return;
  if (inspectorShouldNameThread(getenv("NCCL_SET_THREAD_NAME"), current)) {
    pthread_setname_np(pthread_self(), INSPECTOR_PROFILER_THREAD_NAME);
  }
}
#else
static inline void inspectorNameProfilerThreadOnce() {}
#endif

#endif // NCCL_INSPECTOR_THREAD_NAME_H_
