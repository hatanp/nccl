#include "inspector_thread_name.h"

#include <assert.h>
#include <stdio.h>
#include <string>
#include <thread>

#ifdef __linux__
static std::string nameAfterStart(const char* initial) {
  std::string result;
  std::thread([&] {
    pthread_setname_np(pthread_self(), initial);
    inspectorNameProfilerThreadOnce();
    char name[16] = {0};
    pthread_getname_np(pthread_self(), name, sizeof(name));
    result = name;
    // Only the first KernelCh start on a thread checks the name.
    pthread_setname_np(pthread_self(), "renamed");
    inspectorNameProfilerThreadOnce();
    pthread_getname_np(pthread_self(), name, sizeof(name));
    assert(std::string(name) == "renamed");
  }).join();
  return result;
}
#endif

int main() {
  assert(!inspectorShouldNameThread(nullptr, "pt_main_thread"));
  assert(!inspectorShouldNameThread("0", "pt_main_thread"));
  assert(inspectorShouldNameThread("1", "pt_main_thread"));
  assert(inspectorShouldNameThread("1", ""));
  assert(!inspectorShouldNameThread("1", "NCCL Progress 3"));
  assert(!inspectorShouldNameThread("1", INSPECTOR_PROFILER_THREAD_NAME));
  static_assert(sizeof(INSPECTOR_PROFILER_THREAD_NAME) <= 16, "Linux thread names hold 15 characters");
#ifdef __linux__
  setenv("NCCL_SET_THREAD_NAME", "1", 1);
  assert(nameAfterStart("pt_main_thread") == INSPECTOR_PROFILER_THREAD_NAME);
  assert(nameAfterStart("NCCL Progress 1") == "NCCL Progress 1");
  setenv("NCCL_SET_THREAD_NAME", "0", 1);
  assert(nameAfterStart("pt_main_thread") == "pt_main_thread");
  printf("thread-name tests passed (naming checked)\n");
#else
  printf("thread-name tests passed (naming needs Linux)\n");
#endif
  return 0;
}
