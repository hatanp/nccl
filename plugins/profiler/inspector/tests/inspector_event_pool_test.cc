#include "inspector_event_pool.h"
#include "nccl/profiler.h"

#include <assert.h>
#include <stdio.h>
#include <stdlib.h>

static void testLogger(ncclDebugLogLevel, unsigned long, const char*, int, const char*, ...) {}
ncclDebugLogger_t logFn = testLogger;

static uint64_t gNowUsecs = 1000000;
static uint64_t fakeClock() { return gNowUsecs; }

static void init(const char* quarantineMs, const char* grow, uint32_t collSize) {
  inspectorEventPoolFinalize();
  setenv("NCCL_INSPECTOR_POOL_QUARANTINE_MS", quarantineMs, 1);
  setenv("NCCL_INSPECTOR_POOL_GROW", grow, 1);
  assert(inspectorEventPoolInit(collSize, collSize, 4) == inspectorSuccess);
  inspectorEventPoolSetClock(fakeClock);
}

static inspectorCollInfo* startColl(uint64_t sn, int64_t step) {
  inspectorCollInfo* coll = inspectorEventPoolAllocColl();
  assert(coll != nullptr);
  coll->type = ncclProfileColl;
  coll->func = "ReduceScatter";
  coll->algo = "RING";
  coll->proto = "SIMPLE";
  coll->sn = sn;
  coll->msgSizeBytes = 4096;
  coll->parentIdentity.id = sn + 100;
  coll->applicationStep = step;
  return coll;
}

int main() {
  // Default window: a released record keeps its contents and is not reused.
  init("1000", "1", 2);
  assert(inspectorEventPoolQuarantineUsecs() == 1000000);
  inspectorCollInfo* first = startColl(1, 5);
  inspectorProxyParentView view{};
  uint64_t age = 0;
  assert(inspectorEventPoolViewColl(first, &view, &age) == inspectorPoolRecordLive);
  assert(view.sn == 1 && view.applicationStep == 5 && view.identity.id == 101 && age == 0);
  inspectorEventPoolReleaseColl(first);
  gNowUsecs += 250000;
  view = inspectorProxyParentView{};
  assert(inspectorEventPoolViewColl(first, &view, &age) == inspectorPoolRecordQuarantined);
  assert(age == 250000 && view.sn == 1 && view.applicationStep == 5 && view.msgSizeBytes == 4096);
  inspectorCollInfo* second = startColl(2, 6);
  assert(second != first);
  // A second release is still caught and leaves the counts intact.
  inspectorEventPoolReleaseColl(first);
  assert(g_eventPool.collAllocCount == 1);
  // Just inside the window the record is still held; at the window it is reusable.
  gNowUsecs += 749999;
  assert(inspectorEventPoolViewColl(first, &view, &age) == inspectorPoolRecordQuarantined);
  inspectorCollInfo* third = startColl(3, 6);
  assert(third != first);
  gNowUsecs += 1;
  assert(inspectorEventPoolViewColl(first, &view, &age) == inspectorPoolRecordExpired);
  inspectorCollInfo* fourth = startColl(4, 7);
  assert(fourth == first);
  assert(inspectorEventPoolViewColl(fourth, &view, &age) == inspectorPoolRecordLive && view.sn == 4);
  inspectorEventPoolReleaseColl(second);
  inspectorEventPoolReleaseColl(third);
  inspectorEventPoolReleaseColl(fourth);
  assert(g_eventPool.collAllocCount == 0);
  assert(inspectorEventPoolQuarantineEvictions() == 0);

  // Growth disabled and the pool full of quarantined records: reuse the oldest.
  init("1000", "0", 2);
  inspectorCollInfo* a = startColl(1, 1);
  inspectorCollInfo* b = startColl(2, 1);
  inspectorEventPoolReleaseColl(a);
  inspectorEventPoolReleaseColl(b);
  inspectorCollInfo* c = startColl(3, 1);
  assert(c == a && inspectorEventPoolQuarantineEvictions() == 1);
  inspectorCollInfo* d = startColl(4, 1);
  assert(d == b && inspectorEventPoolQuarantineEvictions() == 2);
  assert(inspectorEventPoolAllocColl() == nullptr);
  inspectorEventPoolReleaseColl(c);
  inspectorEventPoolReleaseColl(d);

  // Window 0 restores immediate reuse.
  init("0", "1", 2);
  inspectorCollInfo* e = startColl(1, 1);
  inspectorEventPoolReleaseColl(e);
  assert(inspectorEventPoolViewColl(e, &view, &age) == inspectorPoolRecordExpired);
  assert(startColl(2, 1) == e);

  // P2P records follow the same rules.
  init("1000", "1", 2);
  inspectorP2pInfo* p2p = inspectorEventPoolAllocP2p();
  p2p->type = ncclProfileP2p;
  p2p->func = "Send";
  p2p->sn = 9;
  p2p->applicationStep = 3;
  inspectorEventPoolReleaseP2p(p2p);
  view = inspectorProxyParentView{};
  assert(inspectorEventPoolViewP2p(p2p, &view, &age) == inspectorPoolRecordQuarantined);
  assert(view.sn == 9 && view.applicationStep == 3);
  assert(inspectorEventPoolAllocP2p() != p2p);

  inspectorEventPoolFinalize();
  printf("event-pool tests passed\n");
  return 0;
}
