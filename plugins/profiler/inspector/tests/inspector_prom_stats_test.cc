#include "inspector_prom_stats.h"

#include <assert.h>
#include <stdint.h>
#include <map>
#include <string>
#include <vector>

struct TestEvent {
  uint64_t duration;
};

int main() {
  inspectorPromTopologySizes sizes;
  sizes.world = 2048;
  sizes.dp = 512;
  sizes.edp = 16;
  sizes.ep = 32;
  sizes.pp = 4;
  assert(inspectorPromClassifyFamily(false, false, 2048, 32, sizes)
         == inspectorPromFamilyGlobal);
  assert(inspectorPromClassifyFamily(false, false, 512, 8, sizes)
         == inspectorPromFamilyDp);
  assert(inspectorPromClassifyFamily(false, false, 16, 8, sizes)
         == inspectorPromFamilyEdp);
  assert(inspectorPromClassifyFamily(false, false, 32, 1, sizes)
         == inspectorPromFamilyEp);
  assert(inspectorPromClassifyFamily(false, false, 4, 4, sizes)
         == inspectorPromFamilyUnknown);
  assert(inspectorPromClassifyFamily(true, false, 2, 1, sizes)
         == inspectorPromFamilyPpLocal);
  assert(inspectorPromClassifyFamily(true, false, 2, 2, sizes)
         == inspectorPromFamilyPpCrossNode);
  assert(std::string(inspectorPromSemanticFamilyName(inspectorPromFamilyDp))
         == "dp");

  // The helper needs only the AllReduce discriminator: both AllGather and
  // ReduceScatter take the non-P2P, non-AllReduce path tested above.
  // Regression: a 1,853,358,080-byte AllReduce on {0,192} or {32,224}
  // matches EDP2 in TP1/PP4/DP64, but these are pipeline endpoint groups.
  // Membership and payload are not classifier inputs; do not invent them.
  sizes.world = 256;
  sizes.dp = 64;
  sizes.edp = 2;
  assert(inspectorPromClassifyFamily(false, true, 2, 2, sizes)
         == inspectorPromFamilyUnknown);
  assert(inspectorPromClassifyFamily(false, false, 2, 2, sizes)
         == inspectorPromFamilyEdp); // AG/RS heuristic preserved.

  // At 128 nodes EDP4 and PP4 collide. Even a full PP-size match is not
  // evidence of replica semantics for AllReduce.
  sizes.world = 512;
  sizes.dp = 128;
  sizes.edp = 4;
  assert(inspectorPromClassifyFamily(false, true, 4, 4, sizes)
         == inspectorPromFamilyUnknown);
  assert(inspectorPromClassifyFamily(false, false, 4, 4, sizes)
         == inspectorPromFamilyEdp);
  assert(inspectorPromClassifyFamily(false, true, 128, 32, sizes)
         == inspectorPromFamilyUnknown); // DP size is not semantic proof either.
  assert(inspectorPromClassifyFamily(false, false, 128, 32, sizes)
         == inspectorPromFamilyDp);
  assert(inspectorPromClassifyFamily(false, true, 32, 8, sizes)
         == inspectorPromFamilyUnknown); // EP size has the same limitation.
  assert(inspectorPromClassifyFamily(false, true, 512, 128, sizes)
         == inspectorPromFamilyGlobal);
  assert(inspectorPromClassifyFamily(true, false, 4, 1, sizes)
         == inspectorPromFamilyPpLocal);
  assert(inspectorPromClassifyFamily(true, false, 4, 4, sizes)
         == inspectorPromFamilyPpCrossNode);
  assert(std::string(inspectorPromSemanticFamilyName(
           inspectorPromClassifyFamily(false, true, 4, 4, sizes)))
         == "unknown");

  // Distinct legacy bins must survive even when their emitted semantics agree.
  // Production keys use the size classifier, not the conservative label.
  std::map<inspectorPromSemanticFamily, uint64_t> binCounts;
  binCounts[inspectorPromClassifySizeFamily(false, 128, 32, sizes)] += 2;
  binCounts[inspectorPromClassifySizeFamily(false, 4, 4, sizes)] += 3;
  binCounts[inspectorPromClassifySizeFamily(false, 32, 8, sizes)] += 5;
  assert(binCounts.size() == 3);
  assert(binCounts[inspectorPromFamilyDp] == 2);
  assert(binCounts[inspectorPromFamilyEdp] == 3);
  assert(binCounts[inspectorPromFamilyEp] == 5);
  for (const auto& bin : binCounts) {
    const auto policy = inspectorPromFamilyPolicy(bin.first, true);
    assert(policy.family == inspectorPromFamilyUnknown);
    assert(policy.emitGpuIntervals == (bin.first != inspectorPromFamilyEp));
    const auto nonAllReduce = inspectorPromFamilyPolicy(bin.first, false);
    assert(nonAllReduce.family == bin.first);
    assert(nonAllReduce.emitGpuIntervals == policy.emitGpuIntervals);
  }
  for (const auto family : {inspectorPromFamilyGlobal, inspectorPromFamilyPpLocal,
                            inspectorPromFamilyPpCrossNode,
                            inspectorPromFamilyUnknown}) {
    const auto policy = inspectorPromFamilyPolicy(family, false);
    assert(policy.family == family);
    assert(!policy.emitGpuIntervals);
  }
  assert(inspectorPromFamilyPolicy(inspectorPromFamilyGlobal, true).family
         == inspectorPromFamilyGlobal);

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
