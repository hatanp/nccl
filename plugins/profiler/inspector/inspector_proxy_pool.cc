#include "inspector_proxy_pool.h"

#include <stddef.h>
#include <atomic>
#include <stdlib.h>
#include <string.h>
#include <pthread.h>

template <typename T>
struct inspectorProxyPoolEntry {
  T object;
  inspectorProxyPoolEntry<T>* next;
  bool inUse;
};

template <typename T>
struct inspectorFixedProxyPool {
  inspectorProxyPoolEntry<T>* entries = nullptr;
  inspectorProxyPoolEntry<T>* freeList = nullptr;
  uint32_t capacity = 0;
  uint64_t dropped = 0;
  pthread_mutex_t lock = PTHREAD_MUTEX_INITIALIZER;
};

static inspectorFixedProxyPool<inspectorProxyOpInfo> gOpPool;
static inspectorFixedProxyPool<inspectorProxyStepInfo> gStepPool;
static std::atomic<uint64_t> gDetachedOps {0};

template <typename T>
static bool initializePool(inspectorFixedProxyPool<T>& pool, uint32_t capacity) {
  pool.entries = static_cast<inspectorProxyPoolEntry<T>*>(
    calloc(capacity, sizeof(inspectorProxyPoolEntry<T>)));
  if (pool.entries == nullptr) return false;
  pool.capacity = capacity;
  pool.dropped = 0;
  pool.freeList = nullptr;
  for (uint32_t index = 0; index < capacity; index++) {
    pool.entries[index].next = pool.freeList;
    pool.freeList = &pool.entries[index];
  }
  return true;
}

template <typename T>
static T* allocateEntry(inspectorFixedProxyPool<T>& pool) {
  pthread_mutex_lock(&pool.lock);
  inspectorProxyPoolEntry<T>* entry = pool.freeList;
  if (entry == nullptr) {
    pool.dropped++;
    pthread_mutex_unlock(&pool.lock);
    return nullptr;
  }
  pool.freeList = entry->next;
  entry->next = nullptr;
  entry->inUse = true;
  pthread_mutex_unlock(&pool.lock);
  memset(&entry->object, 0, sizeof(T));
  return &entry->object;
}

template <typename T>
static void releaseEntry(inspectorFixedProxyPool<T>& pool, T* object) {
  if (object == nullptr || pool.entries == nullptr) return;
  auto* entry = reinterpret_cast<inspectorProxyPoolEntry<T>*>(
    reinterpret_cast<char*>(object) - offsetof(inspectorProxyPoolEntry<T>, object));
  if (entry < pool.entries || entry >= pool.entries + pool.capacity) return;
  pthread_mutex_lock(&pool.lock);
  if (entry->inUse) {
    entry->inUse = false;
    entry->next = pool.freeList;
    pool.freeList = entry;
  }
  pthread_mutex_unlock(&pool.lock);
}

inspectorResult_t inspectorProxyPoolInit(uint32_t opCapacity,
                                         uint32_t stepCapacity) {
  if (opCapacity == 0 || stepCapacity == 0) return inspectorMemoryError;
  if (!initializePool(gOpPool, opCapacity)) return inspectorMemoryError;
  gDetachedOps.store(0, std::memory_order_relaxed);
  if (!initializePool(gStepPool, stepCapacity)) {
    free(gOpPool.entries);
    gOpPool.entries = nullptr;
    gOpPool.freeList = nullptr;
    gOpPool.capacity = 0;
    return inspectorMemoryError;
  }
  return inspectorSuccess;
}

void inspectorProxyPoolFinalize() {
  free(gOpPool.entries);
  free(gStepPool.entries);
  gOpPool.entries = nullptr;
  gStepPool.entries = nullptr;
  gOpPool.freeList = nullptr;
  gStepPool.freeList = nullptr;
  gOpPool.capacity = 0;
  gStepPool.capacity = 0;
}

struct inspectorProxyOpInfo* inspectorProxyPoolAllocOp() {
  return allocateEntry(gOpPool);
}

struct inspectorProxyStepInfo* inspectorProxyPoolAllocStep() {
  return allocateEntry(gStepPool);
}

void inspectorProxyPoolReleaseOp(struct inspectorProxyOpInfo* op) {
  releaseEntry(gOpPool, op);
}

void inspectorProxyPoolReleaseStep(struct inspectorProxyStepInfo* step) {
  releaseEntry(gStepPool, step);
}

uint64_t inspectorProxyPoolDroppedOps() { return gOpPool.dropped; }
uint64_t inspectorProxyPoolDroppedSteps() { return gStepPool.dropped; }
void inspectorProxyPoolRecordDetachedOp() {
  gDetachedOps.fetch_add(1, std::memory_order_relaxed);
}
uint64_t inspectorProxyPoolDetachedOps() {
  return gDetachedOps.load(std::memory_order_relaxed);
}
