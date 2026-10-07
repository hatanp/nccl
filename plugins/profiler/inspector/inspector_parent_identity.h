#ifndef NCCL_INSPECTOR_PARENT_IDENTITY_H_
#define NCCL_INSPECTOR_PARENT_IDENTITY_H_

#include <atomic>
#include <map>
#include <stddef.h>
#include <stdint.h>
#include <string.h>

// Immutable values copied from one local ncclProfileColl callback. No GPU
// address or descriptor-owned string is dereferenced after initialization.
struct inspectorParentIdentity {
  uint64_t id;
  uintptr_t sendBuffer;
  uintptr_t recvBuffer;
  uint64_t count;
  int datatype;
};

static inline const char* inspectorParentDatatypeName(int datatype) {
  static const char* const names[] = {
    "ncclInt8", "ncclUint8", "ncclInt32", "ncclUint32", "ncclInt64", "ncclUint64",
    "ncclFloat16", "ncclFloat32", "ncclFloat64", "ncclBfloat16",
    "ncclFloat8e4m3", "ncclFloat8e5m2"
  };
  return datatype >= 0 && datatype < 12 ? names[datatype] : "unknown";
}

static inline inspectorParentIdentity inspectorParentSnapshot(
    uint64_t id, const void* send, const void* recv, size_t count,
    const char* datatype) {
  inspectorParentIdentity result{};
  result.id = id;
  result.sendBuffer = reinterpret_cast<uintptr_t>(send);
  result.recvBuffer = reinterpret_cast<uintptr_t>(recv);
  result.count = count;
  result.datatype = -1;
  if (datatype != nullptr) {
    for (int i = 0; i < 12; i++) {
      if (strcmp(datatype, inspectorParentDatatypeName(i)) == 0) {
        result.datatype = i;
        break;
      }
    }
  }
  return result;
}

static inline bool inspectorParentIdentityKnown(const inspectorParentIdentity& value) {
  return value.id != 0 && value.sendBuffer != 0 && value.recvBuffer != 0
    && value.count != 0 && value.datatype >= 0 && value.datatype < 12;
}

static inline bool inspectorParentIdentityEqual(
    const inspectorParentIdentity& a, const inspectorParentIdentity& b) {
  return a.id == b.id && a.sendBuffer == b.sendBuffer && a.recvBuffer == b.recvBuffer
    && a.count == b.count && a.datatype == b.datatype;
}

// Saturate instead of wrapping/reusing an ID. Zero always means unavailable.
static inline uint64_t inspectorParentAllocateId(std::atomic<uint64_t>& counter) {
  uint64_t previous = counter.load(std::memory_order_relaxed);
  while (previous != UINT64_MAX) {
    if (counter.compare_exchange_weak(previous, previous + 1,
                                     std::memory_order_relaxed)) return previous + 1;
  }
  return 0;
}

const char* inspectorParentProcessInstance();
uint64_t inspectorParentNextId();

// Dump-time only: retain the lowest K IDs, independent of visitation order.
// Invalid entries occupy their ID slot and cannot masquerade as usable data.
// Conflicting copies fail closed. No callback-time all-parent registry exists.
class inspectorParentDictionary {
 public:
  struct Entry {
    inspectorParentIdentity identity;
    bool conflict;
  };
  explicit inspectorParentDictionary(size_t capacity) : capacity_(capacity) {}
  void consider(const inspectorParentIdentity& identity) {
    if (identity.id == 0 || capacity_ == 0) return;
    auto found = entries_.find(identity.id);
    if (found != entries_.end()) {
      found->second.conflict |= !inspectorParentIdentityEqual(found->second.identity, identity);
      return;
    }
    if (entries_.size() == capacity_) {
      auto largest = --entries_.end();
      if (identity.id >= largest->first) return;
      entries_.erase(largest);
    }
    entries_.emplace(identity.id, Entry{identity, false});
  }
  bool contains(uint64_t id) const {
    auto found = entries_.find(id);
    return found != entries_.end() && !found->second.conflict
      && inspectorParentIdentityKnown(found->second.identity);
  }
  size_t available() const {
    size_t result = 0;
    for (const auto& item : entries_) result += contains(item.first) ? 1 : 0;
    return result;
  }
  const std::map<uint64_t, Entry>& entries() const { return entries_; }
 private:
  size_t capacity_;
  std::map<uint64_t, Entry> entries_;
};

#endif
