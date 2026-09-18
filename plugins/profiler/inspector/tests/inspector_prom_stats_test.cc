#include "inspector_prom_stats.h"

#include <assert.h>
#include <stdint.h>
#include <vector>

struct TestEvent {
  uint64_t duration;
};

int main() {
  std::vector<uint64_t> samples;
  for (uint64_t seen = 1; seen <= 100000; seen++) {
    inspectorPromBoundedSampleUpdate(samples, 256, seen, seen, 4096,
                                     seen);
  }
  assert(samples.size() == 256);
  bool retainedLateSample = false;
  for (uint64_t sample : samples) {
    assert(sample >= 1 && sample <= 100000);
    retainedLateSample |= sample > 256;
  }
  assert(retainedLateSample);

  std::vector<TestEvent> slowest;
  for (uint64_t duration = 1; duration <= 100; duration++) {
    inspectorPromBoundedTopK(
      slowest, TestEvent {duration}, 4,
      [](const TestEvent& lhs, const TestEvent& rhs) {
        return lhs.duration > rhs.duration;
      });
  }
  assert(slowest.size() == 4);
  assert(slowest[0].duration == 100);
  assert(slowest[1].duration == 99);
  assert(slowest[2].duration == 98);
  assert(slowest[3].duration == 97);
  return 0;
}
