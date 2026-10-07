#include "inspector_parent_identity.h"
#include "inspector_proxy_stats.h"
#include "inspector.h"

#include <assert.h>
#include <algorithm>
#include <stdio.h>
#include <string>
#include <thread>
#include <vector>

static inspectorParentIdentity snapshot(uint64_t id, uintptr_t address = 0x1000) {
  return inspectorParentSnapshot(id, reinterpret_cast<void*>(address),
    reinterpret_cast<void*>(address + 0x1000), 1024, "ncclBfloat16");
}

int main() {
  static_assert(offsetof(inspectorCollInfo, type) == 0, "callback type must be first");
  static_assert(offsetof(inspectorProxyOpInfo, type) == 0, "callback type must be first");
  char datatype[] = "ncclBfloat16";
  auto copied = inspectorParentSnapshot(7, reinterpret_cast<void*>(0x1234),
    reinterpret_cast<void*>(0x5678), 4096, datatype);
  datatype[0] = 'x';
  assert(inspectorParentIdentityKnown(copied));
  assert(std::string(inspectorParentDatatypeName(copied.datatype)) == "ncclBfloat16");
  assert(copied.sendBuffer == 0x1234 && copied.recvBuffer == 0x5678 && copied.count == 4096);
  assert(!inspectorParentIdentityKnown(inspectorParentSnapshot(1, nullptr,
    reinterpret_cast<void*>(1), 1, "ncclInt8")));
  assert(!inspectorParentIdentityKnown(inspectorParentSnapshot(1,
    reinterpret_cast<void*>(1), reinterpret_cast<void*>(2), 0, "ncclInt8")));
  assert(!inspectorParentIdentityKnown(inspectorParentSnapshot(1,
    reinterpret_cast<void*>(1), reinterpret_cast<void*>(2), 1, nullptr)));
  assert(!inspectorParentIdentityKnown(inspectorParentSnapshot(1,
    reinterpret_cast<void*>(1), reinterpret_cast<void*>(2), 1, "float32")));
  assert(!inspectorParentIdentityKnown(snapshot(0)));

  std::atomic<uint64_t> counter{0};
  std::vector<std::vector<uint64_t>> ids(4);
  std::vector<std::thread> threads;
  for (size_t i = 0; i < ids.size(); i++) {
    threads.emplace_back([&counter, &ids, i]() {
      for (int j = 0; j < 1000; j++) ids[i].push_back(inspectorParentAllocateId(counter));
    });
  }
  for (auto& thread : threads) thread.join();
  std::vector<uint64_t> all;
  for (auto& values : ids) all.insert(all.end(), values.begin(), values.end());
  std::sort(all.begin(), all.end());
  for (size_t i = 0; i < all.size(); i++) assert(all[i] == i + 1);
  counter.store(UINT64_MAX - 1);
  assert(inspectorParentAllocateId(counter) == UINT64_MAX);
  assert(inspectorParentAllocateId(counter) == 0);
  assert(inspectorParentAllocateId(counter) == 0);
  const std::string process = inspectorParentProcessInstance();
  assert(process.size() == 32);
  assert(process.find_first_not_of("0123456789abcdef") == std::string::npos);
  assert(process == inspectorParentProcessInstance());
  assert(inspectorParentNextId() == 1);
  assert(inspectorParentNextId() == 2);

  inspectorProxyPhasePeak first{};
  first.startUsecs = 100;
  first.stopUsecs = 150;
  first.sequence = 12; // Deliberately equal across different parents/buffers.
  first.parentIdentityId = 7;
  auto second = first;
  second.parentIdentityId = 8;
  second.startUsecs = 200;
  second.stopUsecs = 300;
  inspectorProxyPhasePeak selected{}, numericOnly{};
  inspectorParentIdentity selectedParent{};
  assert(inspectorProxySelectPeakWithParent(selected, selectedParent, first, snapshot(7)));
  assert(inspectorProxySelectPeak(numericOnly, first));
  assert(inspectorProxySelectPeakWithParent(selected, selectedParent, second, snapshot(8, 0x3000)));
  assert(inspectorProxySelectPeak(numericOnly, second));
  assert(selected.parentIdentityId == 8 && selectedParent.id == 8 && selectedParent.sendBuffer == 0x3000);
  auto equal = second;
  equal.parentIdentityId = 9;
  // Exact ties keep the original policy: no new ID-driven tie breaker.
  assert(!inspectorProxySelectPeakWithParent(selected, selectedParent, equal, snapshot(9, 0x9000)));
  assert(!inspectorProxySelectPeak(numericOnly, equal));
  assert(selectedParent.id == 8);
  equal.startUsecs = 150;
  equal.stopUsecs = 250;
  assert(inspectorProxySelectPeakWithParent(selected, selectedParent, equal, snapshot(9, 0x9000)));
  assert(inspectorProxySelectPeak(numericOnly, equal));
  assert(selectedParent.id == 9 && selectedParent.sendBuffer == 0x9000);
  assert(selected.startUsecs == numericOnly.startUsecs && selected.stopUsecs == numericOnly.stopUsecs);
  equal.stopUsecs = 400;
  assert(inspectorProxySelectPeakWithParent(selected, selectedParent, equal, snapshot(123)));
  assert(!inspectorParentIdentityKnown(selectedParent)); // ID/snapshot mismatch fails closed.

  inspectorParentDictionary forward(2), reverse(2), zero(0);
  for (uint64_t id : {4, 1, 3, 2, 1}) forward.consider(snapshot(id));
  for (uint64_t id : {1, 2, 3, 1, 4}) reverse.consider(snapshot(id));
  zero.consider(snapshot(1));
  assert(zero.entries().empty() && !zero.contains(1));
  assert(forward.entries().size() == 2 && forward.available() == 2);
  assert(reverse.entries().size() == 2 && reverse.available() == 2);
  assert(forward.contains(1) && forward.contains(2) && !forward.contains(3));
  for (const auto& item : forward.entries()) {
    assert(reverse.contains(item.first));
    assert(inspectorParentIdentityEqual(item.second.identity, reverse.entries().at(item.first).identity));
  }
  forward.consider(snapshot(1, 0x9999));
  assert(!forward.contains(1) && forward.contains(2) && forward.available() == 1);
  forward.consider(snapshot(1)); // A later agreeing copy cannot erase conflict.
  assert(!forward.contains(1));
  inspectorParentDictionary missing(2);
  auto invalid = snapshot(1);
  invalid.datatype = -1;
  missing.consider(invalid);
  missing.consider(snapshot(2));
  missing.consider(snapshot(3));
  assert(missing.entries().size() == 2 && !missing.contains(1) && missing.contains(2));

  printf("parent_identity=%zu phase_peak=%zu coll_info=%zu proxy_op=%zu proxy_step=%zu dictionary_entry=%zu\n",
    sizeof(inspectorParentIdentity), sizeof(inspectorProxyPhasePeak), sizeof(inspectorCollInfo),
    sizeof(inspectorProxyOpInfo), sizeof(inspectorProxyStepInfo), sizeof(inspectorParentDictionary::Entry));
  return 0;
}
