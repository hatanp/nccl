#include "inspector_event_pool.h"
#include <stdlib.h>
#include <string.h>
#include <time.h>

// Global event pool
struct inspectorEventPool g_eventPool;

static uint64_t inspectorPoolMonotonicUsecs() {
  struct timespec ts;
  clock_gettime(CLOCK_MONOTONIC, &ts);
  return static_cast<uint64_t>(ts.tv_sec) * 1000000ull
    + static_cast<uint64_t>(ts.tv_nsec) / 1000ull;
}

static uint64_t (*gPoolNowUsecs)() = inspectorPoolMonotonicUsecs;

void inspectorEventPoolSetClock(uint64_t (*nowUsecs)()) {
  gPoolNowUsecs = nowUsecs ? nowUsecs : inspectorPoolMonotonicUsecs;
}

// Quarantine helpers, shared by the collective and P2P pools. Callers hold the
// pool lock. Released entries keep their contents until they leave the FIFO.
template <typename Entry>
static void poolReleaseExpired(Entry*& head, Entry*& tail, Entry*& freeList) {
  if (head == nullptr) return;
  const uint64_t now = gPoolNowUsecs();
  while (head != nullptr && now >= head->releasedUsecs
         && now - head->releasedUsecs >= g_eventPool.quarantineUsecs) {
    Entry* entry = head;
    head = entry->next;
    if (head == nullptr) tail = nullptr;
    entry->quarantined = false;
    entry->next = freeList;
    freeList = entry;
  }
}

// Pool exhausted with growth disabled: reuse the oldest quarantined entry.
template <typename Entry>
static bool poolEvictOldest(Entry*& head, Entry*& tail, Entry*& freeList,
                            uint64_t& evictions) {
  if (head == nullptr) return false;
  Entry* entry = head;
  head = entry->next;
  if (head == nullptr) tail = nullptr;
  entry->quarantined = false;
  entry->next = freeList;
  freeList = entry;
  evictions++;
  return true;
}

template <typename Entry>
static void poolReleaseEntry(Entry* entry, Entry*& head, Entry*& tail, Entry*& freeList) {
  entry->inUse = false;
  if (g_eventPool.quarantineUsecs == 0) {
    entry->next = freeList;
    freeList = entry;
    return;
  }
  entry->quarantined = true;
  entry->releasedUsecs = gPoolNowUsecs();
  entry->next = nullptr;
  if (tail != nullptr) tail->next = entry;
  else head = entry;
  tail = entry;
}

template <typename Entry>
static inspectorPoolRecordState poolRecordState(const Entry* entry, uint64_t* releasedAgeUsecs) {
  if (entry->inUse) return inspectorPoolRecordLive;
  if (entry->quarantined) {
    const uint64_t now = gPoolNowUsecs();
    const uint64_t age = now >= entry->releasedUsecs ? now - entry->releasedUsecs : 0;
    if (age < g_eventPool.quarantineUsecs) {
      *releasedAgeUsecs = age;
      return inspectorPoolRecordQuarantined;
    }
  }
  return inspectorPoolRecordExpired;
}

/*
 * Description:
 *   Helper function to allocate a new chunk for a pool.
 *
 * Parameters:
 *
 *   entrySize - Size of each entry in bytes.
 *   chunkSize - Number of entries to allocate in the chunk.
 *
 * Return:
 *
 *   struct inspectorPoolChunk* - Newly allocated chunk, or nullptr on
 *   failure.
 *
 */
static struct inspectorPoolChunk* allocatePoolChunk(size_t entrySize,
                                                    uint32_t chunkSize) {
  struct inspectorPoolChunk* chunk
    = (struct inspectorPoolChunk*)calloc(1, sizeof(struct inspectorPoolChunk));
  if (chunk == nullptr) {
    return nullptr;
  }

  chunk->entries = calloc(chunkSize, entrySize);
  if (chunk->entries == nullptr) {
    free(chunk);
    return nullptr;
  }

  chunk->chunkSize = chunkSize;
  chunk->next = nullptr;
  return chunk;
}


/*
 * Description:
 *   Initialize collective info pool with first chunk.
 *
 * Parameters:
 *
 *   strideSize - Number of entries to allocate in the initial chunk.
 *
 * Return:
 *
 *   inspectorSuccess    - Success.
 *   inspectorLockError  - Failed to initialize mutex.
 *   inspectorMemoryError - Failed to allocate initial chunk.
 *
 */
static inspectorResult_t initCollectivePool(uint32_t strideSize) {
  g_eventPool.collStrideSize = strideSize;
  g_eventPool.collTotalSize = 0;
  g_eventPool.collAllocCount = 0;
  g_eventPool.collChunkCount = 0;
  g_eventPool.collChunkList = nullptr;
  g_eventPool.collFreeList = nullptr;

  if (pthread_mutex_init(&g_eventPool.collPoolLock, nullptr) != 0) {
    return inspectorLockError;
  }

  struct inspectorPoolChunk* chunk
    = allocatePoolChunk(sizeof(struct inspectorCollInfoPoolEntry),
                        strideSize);
  if (chunk == nullptr) {
    INFO_INSPECTOR(
      "NCCL Inspector: Failed to allocate initial collective info pool chunk");
    pthread_mutex_destroy(&g_eventPool.collPoolLock);
    return inspectorMemoryError;
  }

  g_eventPool.collChunkList = chunk;
  g_eventPool.collChunkCount = 1;
  g_eventPool.collTotalSize = strideSize;

  struct inspectorCollInfoPoolEntry* entries
    = (struct inspectorCollInfoPoolEntry*)chunk->entries;
  g_eventPool.collFreeList = &entries[0];
  for (uint32_t i = 0; i < strideSize - 1; i++) {
    entries[i].next = &entries[i + 1];
    entries[i].inUse = false;
  }
  entries[strideSize - 1].next = nullptr;
  entries[strideSize - 1].inUse = false;

  INFO_INSPECTOR(
    "NCCL Inspector: Initialized collective pool with stride size %u",
    strideSize);
  return inspectorSuccess;
}

/*
 * Description:
 *   Grow collective pool by allocating a new chunk.
 *
 * Return:
 *
 *   inspectorSuccess    - Success.
 *   inspectorMemoryError - Failed to allocate new chunk.
 *
 */
static inspectorResult_t growCollectivePool() {
  struct inspectorPoolChunk* newChunk
    = allocatePoolChunk(sizeof(struct inspectorCollInfoPoolEntry),
                        g_eventPool.collStrideSize);

  if (newChunk == nullptr) {
    WARN_INSPECTOR("NCCL Inspector: Failed to grow collective info pool (current chunks: %u, total entries: %u)",
                   g_eventPool.collChunkCount, g_eventPool.collTotalSize);
    return inspectorMemoryError;
  }

  newChunk->next = g_eventPool.collChunkList;
  g_eventPool.collChunkList = newChunk;
  g_eventPool.collChunkCount++;
  g_eventPool.collTotalSize += g_eventPool.collStrideSize;

  struct inspectorCollInfoPoolEntry* entries
    = (struct inspectorCollInfoPoolEntry*)newChunk->entries;
  for (uint32_t i = 0; i < g_eventPool.collStrideSize - 1; i++) {
    entries[i].next = &entries[i + 1];
    entries[i].inUse = false;
  }
  entries[g_eventPool.collStrideSize - 1].next = g_eventPool.collFreeList;
  entries[g_eventPool.collStrideSize - 1].inUse = false;
  g_eventPool.collFreeList = &entries[0];

  INFO_INSPECTOR(
    "NCCL Inspector: Grew collective pool to %u chunks (%u total entries)",
    g_eventPool.collChunkCount, g_eventPool.collTotalSize);
  return inspectorSuccess;
}

/*
 * Description:
 *   Initialize P2P info pool with first chunk.
 *
 * Parameters:
 *
 *   strideSize - Number of entries to allocate in the initial chunk.
 *
 * Return:
 *
 *   inspectorSuccess    - Success.
 *   inspectorLockError  - Failed to initialize mutex.
 *   inspectorMemoryError - Failed to allocate initial chunk.
 *
 */
static inspectorResult_t initP2pPool(uint32_t strideSize) {
  g_eventPool.p2pStrideSize = strideSize;
  g_eventPool.p2pTotalSize = 0;
  g_eventPool.p2pAllocCount = 0;
  g_eventPool.p2pChunkCount = 0;
  g_eventPool.p2pChunkList = nullptr;
  g_eventPool.p2pFreeList = nullptr;

  if (pthread_mutex_init(&g_eventPool.p2pPoolLock, nullptr) != 0) {
    return inspectorLockError;
  }

  struct inspectorPoolChunk* chunk
    = allocatePoolChunk(sizeof(struct inspectorP2pInfoPoolEntry),
                        strideSize);
  if (chunk == nullptr) {
    INFO_INSPECTOR(
      "NCCL Inspector: Failed to allocate initial P2P info pool chunk");
    pthread_mutex_destroy(&g_eventPool.p2pPoolLock);
    return inspectorMemoryError;
  }

  g_eventPool.p2pChunkList = chunk;
  g_eventPool.p2pChunkCount = 1;
  g_eventPool.p2pTotalSize = strideSize;

  struct inspectorP2pInfoPoolEntry* entries
    = (struct inspectorP2pInfoPoolEntry*)chunk->entries;
  g_eventPool.p2pFreeList = &entries[0];
  for (uint32_t i = 0; i < strideSize - 1; i++) {
    entries[i].next = &entries[i + 1];
    entries[i].inUse = false;
  }
  entries[strideSize - 1].next = nullptr;
  entries[strideSize - 1].inUse = false;

  INFO_INSPECTOR(
    "NCCL Inspector: Initialized P2P pool with stride size %u",
    strideSize);
  return inspectorSuccess;
}

/*
 * Description:
 *   Grow P2P pool by allocating a new chunk.
 *
 * Return:
 *
 *   inspectorSuccess    - Success.
 *   inspectorMemoryError - Failed to allocate new chunk.
 *
 */
static inspectorResult_t growP2pPool() {
  struct inspectorPoolChunk* newChunk
    = allocatePoolChunk(sizeof(struct inspectorP2pInfoPoolEntry),
                        g_eventPool.p2pStrideSize);

  if (newChunk == nullptr) {
    WARN_INSPECTOR("NCCL Inspector: Failed to grow P2P info pool (current chunks: %u, total entries: %u)",
                   g_eventPool.p2pChunkCount, g_eventPool.p2pTotalSize);
    return inspectorMemoryError;
  }

  newChunk->next = g_eventPool.p2pChunkList;
  g_eventPool.p2pChunkList = newChunk;
  g_eventPool.p2pChunkCount++;
  g_eventPool.p2pTotalSize += g_eventPool.p2pStrideSize;

  struct inspectorP2pInfoPoolEntry* entries
    = (struct inspectorP2pInfoPoolEntry*)newChunk->entries;
  for (uint32_t i = 0; i < g_eventPool.p2pStrideSize - 1; i++) {
    entries[i].next = &entries[i + 1];
    entries[i].inUse = false;
  }
  entries[g_eventPool.p2pStrideSize - 1].next = g_eventPool.p2pFreeList;
  entries[g_eventPool.p2pStrideSize - 1].inUse = false;
  g_eventPool.p2pFreeList = &entries[0];

  INFO_INSPECTOR( "NCCL Inspector: Grew P2P pool to %u chunks (%u total entries)",
                  g_eventPool.p2pChunkCount, g_eventPool.p2pTotalSize);
  return inspectorSuccess;
}

/*
 * Description:
 *   Initialize comm info pool with first chunk.
 *
 * Parameters:
 *
 *   strideSize - Number of entries to allocate in the initial chunk.
 *
 * Return:
 *
 *   inspectorSuccess    - Success.
 *   inspectorLockError  - Failed to initialize mutex.
 *   inspectorMemoryError - Failed to allocate initial chunk.
 *
 */
static inspectorResult_t initCommPool(uint32_t strideSize) {
  g_eventPool.commStrideSize = strideSize;
  g_eventPool.commTotalSize = 0;
  g_eventPool.commAllocCount = 0;
  g_eventPool.commChunkCount = 0;
  g_eventPool.commChunkList = nullptr;
  g_eventPool.commFreeList = nullptr;

  if (pthread_mutex_init(&g_eventPool.commPoolLock, nullptr) != 0) {
    return inspectorLockError;
  }

  // Allocate first chunk
  struct inspectorPoolChunk* chunk = allocatePoolChunk(sizeof(struct inspectorCommInfoPoolEntry), strideSize);
  if (chunk == nullptr) {
    INFO_INSPECTOR( "NCCL Inspector: Failed to allocate initial comm info pool chunk");
    pthread_mutex_destroy(&g_eventPool.commPoolLock);
    return inspectorMemoryError;
  }

  g_eventPool.commChunkList = chunk;
  g_eventPool.commChunkCount = 1;
  g_eventPool.commTotalSize = strideSize;

  // Initialize free list for this chunk
  struct inspectorCommInfoPoolEntry* entries = (struct inspectorCommInfoPoolEntry*)chunk->entries;
  g_eventPool.commFreeList = &entries[0];
  for (uint32_t i = 0; i < strideSize - 1; i++) {
    entries[i].next = &entries[i + 1];
    entries[i].inUse = false;
  }
  entries[strideSize - 1].next = nullptr;
  entries[strideSize - 1].inUse = false;

  INFO_INSPECTOR( "NCCL Inspector: Initialized comm pool with stride size %u", strideSize);
  return inspectorSuccess;
}

/*
 * Description:
 *   Grow comm pool by allocating a new chunk.
 *
 * Return:
 *
 *   inspectorSuccess    - Success.
 *   inspectorMemoryError - Failed to allocate new chunk.
 *
 */
static inspectorResult_t growCommPool() {
  struct inspectorPoolChunk* newChunk = allocatePoolChunk(
    sizeof(struct inspectorCommInfoPoolEntry),
    g_eventPool.commStrideSize);

  if (newChunk == nullptr) {
    WARN_INSPECTOR("NCCL Inspector: Failed to grow comm info pool (current chunks: %u, total entries: %u)",
                   g_eventPool.commChunkCount, g_eventPool.commTotalSize);
    return inspectorMemoryError;
  }

  // Add chunk to the list
  newChunk->next = g_eventPool.commChunkList;
  g_eventPool.commChunkList = newChunk;
  g_eventPool.commChunkCount++;
  g_eventPool.commTotalSize += g_eventPool.commStrideSize;

  // Add entries from new chunk to free list
  struct inspectorCommInfoPoolEntry* entries = (struct inspectorCommInfoPoolEntry*)newChunk->entries;
  for (uint32_t i = 0; i < g_eventPool.commStrideSize - 1; i++) {
    entries[i].next = &entries[i + 1];
    entries[i].inUse = false;
  }
  entries[g_eventPool.commStrideSize - 1].next = g_eventPool.commFreeList;
  entries[g_eventPool.commStrideSize - 1].inUse = false;
  g_eventPool.commFreeList = &entries[0];

  INFO_INSPECTOR( "NCCL Inspector: Grew comm pool to %u chunks (%u total entries)",
                  g_eventPool.commChunkCount, g_eventPool.commTotalSize);
  return inspectorSuccess;
}

/*
 * Description:
 *   Free all chunks in a chunk list.
 *
 * Parameters:
 *
 *   chunkList - Head of the chunk list to free.
 *
 * Return:
 *
 *   None.
 *
 */
static void freeChunkList(struct inspectorPoolChunk* chunkList) {
  struct inspectorPoolChunk* chunk = chunkList;
  while (chunk != nullptr) {
    struct inspectorPoolChunk* next = chunk->next;
    free(chunk->entries);
    free(chunk);
    chunk = next;
  }
}

/*
 * Description:
 *   Cleanup all pools and destroy their associated locks.
 *
 * Return:
 *
 *   None.
 *
 */
static void cleanupPartialPoolInit() {
  if (g_eventPool.collChunkList != nullptr) {
    pthread_mutex_destroy(&g_eventPool.collPoolLock);
    freeChunkList(g_eventPool.collChunkList);
    g_eventPool.collChunkList = nullptr;
  }
  if (g_eventPool.p2pChunkList != nullptr) {
    pthread_mutex_destroy(&g_eventPool.p2pPoolLock);
    freeChunkList(g_eventPool.p2pChunkList);
    g_eventPool.p2pChunkList = nullptr;
  }
  if (g_eventPool.commChunkList != nullptr) {
    pthread_mutex_destroy(&g_eventPool.commPoolLock);
    freeChunkList(g_eventPool.commChunkList);
    g_eventPool.commChunkList = nullptr;
  }
}

/*
 * Description:
 *   Initialize all event pools with specified sizes.
 *
 * Parameters:
 *
 *   collPoolSize     - Initial size for collective info pool.
 *   p2pPoolSize      - Initial size for P2P info pool.
 *   commPoolSize     - Initial size for comm info pool.
 * Return:
 *
 *   inspectorSuccess    - Success.
 *   inspectorLockError  - Failed to initialize mutex.
 *   inspectorMemoryError - Failed to allocate initial chunk.
 *
 */
inspectorResult_t inspectorEventPoolInit(uint32_t collPoolSize,
                                         uint32_t p2pPoolSize,
                                         uint32_t commPoolSize) {
  inspectorResult_t res;

  memset(&g_eventPool, 0, sizeof(struct inspectorEventPool));

  const char* growStr = getenv("NCCL_INSPECTOR_POOL_GROW");
  g_eventPool.growEnabled = growStr ? (atoi(growStr) != 0) : true;
  const char* quarantineStr = getenv("NCCL_INSPECTOR_POOL_QUARANTINE_MS");
  long quarantineMs = quarantineStr ? strtol(quarantineStr, nullptr, 10) : 1000;
  if (quarantineMs < 0) quarantineMs = 0;
  g_eventPool.quarantineUsecs = static_cast<uint64_t>(quarantineMs) * 1000ull;

  res = initCollectivePool(collPoolSize);
  if (res != inspectorSuccess) {
    cleanupPartialPoolInit();
    return res;
  }

  res = initP2pPool(p2pPoolSize);
  if (res != inspectorSuccess) {
    cleanupPartialPoolInit();
    return res;
  }

  res = initCommPool(commPoolSize);
  if (res != inspectorSuccess) {
    cleanupPartialPoolInit();
    return res;
  }

  INFO_INSPECTOR(
    "NCCL Inspector: Memory pools initialized (stride-based) - Coll: %u, P2P: %u, Comms: %u, pool grow: %s, quarantine: %ld ms",
    collPoolSize, p2pPoolSize, commPoolSize, g_eventPool.growEnabled ? "enabled" : "disabled",
    quarantineMs);

  return inspectorSuccess;
}

/*
 * Description:
 *   Finalize and cleanup all event pools.
 *
 * Return:
 *
 *   inspectorSuccess - Success.
 *
 */
inspectorResult_t inspectorEventPoolFinalize() {
  cleanupPartialPoolInit();
  return inspectorSuccess;
}

/*
 * Description:
 *   Allocate a collective info object from the pool. Grows the pool if
 *   necessary.
 *
 * Thread Safety:
 *
 *   Thread-safe.
 *
 * Return:
 *
 *   struct inspectorCollInfo* - Allocated collective info, or nullptr on
 *                                failure.
 *
 */
struct inspectorCollInfo* inspectorEventPoolAllocColl() {
  pthread_mutex_lock(&g_eventPool.collPoolLock);
  poolReleaseExpired(g_eventPool.collQuarantineHead, g_eventPool.collQuarantineTail,
                     g_eventPool.collFreeList);

  // If free list is empty, try to grow the pool
  if (g_eventPool.collFreeList == nullptr && !g_eventPool.growEnabled) {
    if (!poolEvictOldest(g_eventPool.collQuarantineHead, g_eventPool.collQuarantineTail,
                         g_eventPool.collFreeList, g_eventPool.collQuarantineEvictions)) {
      pthread_mutex_unlock(&g_eventPool.collPoolLock);
      WARN_INSPECTOR("NCCL Inspector: Collective pool exhausted and pool grow is disabled (NCCL_INSPECTOR_POOL_GROW=0) - allocation failed!");
      return nullptr;
    }
  }
  if (g_eventPool.collFreeList == nullptr) {
    INFO_INSPECTOR(
      "NCCL Inspector: Collective pool exhausted, growing pool (current: %u chunks, %u entries)",
      g_eventPool.collChunkCount, g_eventPool.collTotalSize);

    inspectorResult_t res = growCollectivePool();
    if (res != inspectorSuccess) {
      pthread_mutex_unlock(&g_eventPool.collPoolLock);
      WARN_INSPECTOR("NCCL Inspector: Failed to grow collective pool - allocation failed!");
      return nullptr;
    }
  }

  struct inspectorCollInfoPoolEntry* entry = g_eventPool.collFreeList;
  g_eventPool.collFreeList = entry->next;
  entry->inUse = true;
  entry->quarantined = false;
  entry->next = nullptr;
  g_eventPool.collAllocCount++;

  pthread_mutex_unlock(&g_eventPool.collPoolLock);

  memset(&entry->obj, 0, sizeof(struct inspectorCollInfo));

  return &entry->obj;
}

/*
 * Description:
 *   Allocate a P2P info object from the pool. Grows the pool if
 *   necessary.
 *
 * Thread Safety:
 *
 *   Thread-safe.
 *
 * Return:
 *
 *   struct inspectorP2pInfo* - Allocated P2P info, or nullptr on failure.
 *
 */
struct inspectorP2pInfo* inspectorEventPoolAllocP2p() {
  pthread_mutex_lock(&g_eventPool.p2pPoolLock);
  poolReleaseExpired(g_eventPool.p2pQuarantineHead, g_eventPool.p2pQuarantineTail,
                     g_eventPool.p2pFreeList);

  // If free list is empty, try to grow the pool
  if (g_eventPool.p2pFreeList == nullptr && !g_eventPool.growEnabled) {
    if (!poolEvictOldest(g_eventPool.p2pQuarantineHead, g_eventPool.p2pQuarantineTail,
                         g_eventPool.p2pFreeList, g_eventPool.p2pQuarantineEvictions)) {
      pthread_mutex_unlock(&g_eventPool.p2pPoolLock);
      WARN_INSPECTOR("NCCL Inspector: P2P pool exhausted and pool grow is disabled (NCCL_INSPECTOR_POOL_GROW=0) - allocation failed!");
      return nullptr;
    }
  }
  if (g_eventPool.p2pFreeList == nullptr) {
    INFO_INSPECTOR( "NCCL Inspector: P2P pool exhausted, growing pool (current: %u chunks, %u entries)",
                    g_eventPool.p2pChunkCount, g_eventPool.p2pTotalSize);

    inspectorResult_t res = growP2pPool();
    if (res != inspectorSuccess) {
      pthread_mutex_unlock(&g_eventPool.p2pPoolLock);
      WARN_INSPECTOR("NCCL Inspector: Failed to grow P2P pool - allocation failed!");
      return nullptr;
    }
  }

  struct inspectorP2pInfoPoolEntry* entry = g_eventPool.p2pFreeList;
  g_eventPool.p2pFreeList = entry->next;
  entry->inUse = true;
  entry->quarantined = false;
  entry->next = nullptr;
  g_eventPool.p2pAllocCount++;

  pthread_mutex_unlock(&g_eventPool.p2pPoolLock);

  // Clear the structure before returning
  memset(&entry->obj, 0, sizeof(struct inspectorP2pInfo));

  return &entry->obj;
}

/*
 * Description:
 *   Allocate a comm info object from the pool. Grows the pool if
 *   necessary.
 *
 * Thread Safety:
 *
 *   Thread-safe.
 *
 * Return:
 *
 *   struct inspectorCommInfo* - Allocated comm info, or nullptr on
 *                                failure.
 *
 */
struct inspectorCommInfo* inspectorEventPoolAllocComm() {
  pthread_mutex_lock(&g_eventPool.commPoolLock);

  // If free list is empty, try to grow the pool
  if (g_eventPool.commFreeList == nullptr) {
    if (!g_eventPool.growEnabled) {
      pthread_mutex_unlock(&g_eventPool.commPoolLock);
      WARN_INSPECTOR("NCCL Inspector: Comm pool exhausted and pool grow is disabled (NCCL_INSPECTOR_POOL_GROW=0) - allocation failed!");
      return nullptr;
    }
    INFO_INSPECTOR( "NCCL Inspector: Comm pool exhausted, growing pool (current: %u chunks, %u entries)",
                    g_eventPool.commChunkCount, g_eventPool.commTotalSize);

    inspectorResult_t res = growCommPool();
    if (res != inspectorSuccess) {
      pthread_mutex_unlock(&g_eventPool.commPoolLock);
      WARN_INSPECTOR("NCCL Inspector: Failed to grow comm pool - allocation failed!");
      return nullptr;
    }
  }

  struct inspectorCommInfoPoolEntry* entry = g_eventPool.commFreeList;
  g_eventPool.commFreeList = entry->next;
  entry->inUse = true;
  entry->next = nullptr;
  g_eventPool.commAllocCount++;

  pthread_mutex_unlock(&g_eventPool.commPoolLock);

  // Clear the structure before returning
  entry->obj = inspectorCommInfo{};

  return &entry->obj;
}

/*
 * Description:
 *   Release a collective info object back to the pool.
 *
 * Thread Safety:
 *
 *   Thread-safe.
 *
 * Parameters:
 *
 *   collInfo - Collective info object to release.
 *
 * Return:
 *
 *   None.
 *
 */
void inspectorEventPoolReleaseColl(struct inspectorCollInfo* collInfo) {
  if (collInfo == nullptr) {
    return;
  }

  // Calculate the pool entry address from the object address
  struct inspectorCollInfoPoolEntry* entry =
    (struct inspectorCollInfoPoolEntry*)((char*)collInfo -
                                         offsetof(struct inspectorCollInfoPoolEntry, obj));

  pthread_mutex_lock(&g_eventPool.collPoolLock);

  if (!entry->inUse) {
    pthread_mutex_unlock(&g_eventPool.collPoolLock);
    WARN_INSPECTOR("NCCL Inspector: Double release detected for collective info!");
    return;
  }

  poolReleaseEntry(entry, g_eventPool.collQuarantineHead, g_eventPool.collQuarantineTail,
                   g_eventPool.collFreeList);
  g_eventPool.collAllocCount--;

  pthread_mutex_unlock(&g_eventPool.collPoolLock);
}

/*
 * Description:
 *   Release a P2P info object back to the pool.
 *
 * Thread Safety:
 *
 *   Thread-safe.
 *
 * Parameters:
 *
 *   p2pInfo - P2P info object to release.
 *
 * Return:
 *
 *   None.
 *
 */
void inspectorEventPoolReleaseP2p(struct inspectorP2pInfo* p2pInfo) {
  if (p2pInfo == nullptr) {
    return;
  }

  // Calculate the pool entry address from the object address
  struct inspectorP2pInfoPoolEntry* entry =
    (struct inspectorP2pInfoPoolEntry*)((char*)p2pInfo -
                                        offsetof(struct inspectorP2pInfoPoolEntry, obj));

  pthread_mutex_lock(&g_eventPool.p2pPoolLock);

  if (!entry->inUse) {
    pthread_mutex_unlock(&g_eventPool.p2pPoolLock);
    WARN_INSPECTOR("NCCL Inspector: Double release detected for P2P info!");
    return;
  }

  poolReleaseEntry(entry, g_eventPool.p2pQuarantineHead, g_eventPool.p2pQuarantineTail,
                   g_eventPool.p2pFreeList);
  g_eventPool.p2pAllocCount--;

  pthread_mutex_unlock(&g_eventPool.p2pPoolLock);
}

/*
 * Description:
 *   Release a comm info object back to the pool.
 *
 * Thread Safety:
 *
 *   Thread-safe.
 *
 * Parameters:
 *
 *   commInfo - Comm info object to release.
 *
 * Return:
 *
 *   None.
 *
 */
void inspectorEventPoolReleaseComm(struct inspectorCommInfo* commInfo) {
  if (commInfo == nullptr) {
    return;
  }

  // Calculate the pool entry address from the object address
  struct inspectorCommInfoPoolEntry* entry =
    (struct inspectorCommInfoPoolEntry*)((char*)commInfo -
                                         offsetof(struct inspectorCommInfoPoolEntry, obj));

  pthread_mutex_lock(&g_eventPool.commPoolLock);

  if (!entry->inUse) {
    pthread_mutex_unlock(&g_eventPool.commPoolLock);
    WARN_INSPECTOR("NCCL Inspector: Double release detected for comm info!");
    return;
  }

  entry->inUse = false;
  entry->next = g_eventPool.commFreeList;
  g_eventPool.commFreeList = entry;
  g_eventPool.commAllocCount--;

  pthread_mutex_unlock(&g_eventPool.commPoolLock);
}

/*
 * Description:
 *   Copy what a proxy operation needs from its collective parent. The record
 *   may already be released (NCCL 2.31 can complete the kernel before the
 *   proxy appends the operation); inside the quarantine window it is unchanged.
 *
 * Thread Safety:
 *
 *   Thread-safe.
 *
 * Return:
 *
 *   inspectorPoolRecordState - Live/Quarantined (view filled) or Expired.
 *
 */
inspectorPoolRecordState inspectorEventPoolViewColl(
    const struct inspectorCollInfo* collInfo, struct inspectorProxyParentView* view,
    uint64_t* releasedAgeUsecs) {
  const struct inspectorCollInfoPoolEntry* entry =
    (const struct inspectorCollInfoPoolEntry*)((const char*)collInfo -
                                               offsetof(struct inspectorCollInfoPoolEntry, obj));
  *releasedAgeUsecs = 0;
  pthread_mutex_lock(&g_eventPool.collPoolLock);
  inspectorPoolRecordState state = poolRecordState(entry, releasedAgeUsecs);
  if (state != inspectorPoolRecordExpired) {
    view->func = collInfo->func;
    view->sn = collInfo->sn;
    view->msgSizeBytes = collInfo->msgSizeBytes;
    view->algo = collInfo->algo;
    view->proto = collInfo->proto;
    view->identity = collInfo->parentIdentity;
    view->applicationStep = collInfo->applicationStep;
  }
  pthread_mutex_unlock(&g_eventPool.collPoolLock);
  return state;
}

/*
 * Description:
 *   P2P counterpart of inspectorEventPoolViewColl (no algo/proto/identity).
 *
 * Thread Safety:
 *
 *   Thread-safe.
 *
 */
inspectorPoolRecordState inspectorEventPoolViewP2p(
    const struct inspectorP2pInfo* p2pInfo, struct inspectorProxyParentView* view,
    uint64_t* releasedAgeUsecs) {
  const struct inspectorP2pInfoPoolEntry* entry =
    (const struct inspectorP2pInfoPoolEntry*)((const char*)p2pInfo -
                                              offsetof(struct inspectorP2pInfoPoolEntry, obj));
  *releasedAgeUsecs = 0;
  pthread_mutex_lock(&g_eventPool.p2pPoolLock);
  inspectorPoolRecordState state = poolRecordState(entry, releasedAgeUsecs);
  if (state != inspectorPoolRecordExpired) {
    view->func = p2pInfo->func;
    view->sn = p2pInfo->sn;
    view->msgSizeBytes = p2pInfo->msgSizeBytes;
    view->applicationStep = p2pInfo->applicationStep;
  }
  pthread_mutex_unlock(&g_eventPool.p2pPoolLock);
  return state;
}

uint64_t inspectorEventPoolQuarantineEvictions() {
  pthread_mutex_lock(&g_eventPool.collPoolLock);
  uint64_t evictions = g_eventPool.collQuarantineEvictions;
  pthread_mutex_unlock(&g_eventPool.collPoolLock);
  pthread_mutex_lock(&g_eventPool.p2pPoolLock);
  evictions += g_eventPool.p2pQuarantineEvictions;
  pthread_mutex_unlock(&g_eventPool.p2pPoolLock);
  return evictions;
}

uint64_t inspectorEventPoolQuarantineUsecs() {
  return g_eventPool.quarantineUsecs;
}
