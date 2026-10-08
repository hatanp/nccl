#ifndef NCCL_INSPECTOR_EVENT_POOL_H_
#define NCCL_INSPECTOR_EVENT_POOL_H_

#include <pthread.h>
#include <stdint.h>

#include "inspector.h"

// Memory pool entry structures
// Released collective and P2P records stay unchanged in a FIFO quarantine for
// NCCL_INSPECTOR_POOL_QUARANTINE_MS before reuse: on NCCL 2.31 a proxy
// operation can start after its parent's kernel completion released the record.
struct inspectorCollInfoPoolEntry {
  struct inspectorCollInfo obj;
  struct inspectorCollInfoPoolEntry* next;
  bool inUse;
  bool quarantined;
  uint64_t releasedUsecs;
};

struct inspectorP2pInfoPoolEntry {
  struct inspectorP2pInfo obj;
  struct inspectorP2pInfoPoolEntry* next;
  bool inUse;
  bool quarantined;
  uint64_t releasedUsecs;
};

struct inspectorCommInfoPoolEntry {
  struct inspectorCommInfo obj;
  struct inspectorCommInfoPoolEntry* next;
  bool inUse;
};

// Chunk structure for stride-based pool growth
struct inspectorPoolChunk {
  void* entries;                    // Pointer to the array of entries in this chunk
  uint32_t chunkSize;               // Number of entries in this chunk
  struct inspectorPoolChunk* next;  // Next chunk in the list
};

struct inspectorEventPool {
  // Collective info pool
  struct inspectorPoolChunk* collChunkList;
  struct inspectorCollInfoPoolEntry* collFreeList;
  struct inspectorCollInfoPoolEntry* collQuarantineHead;
  struct inspectorCollInfoPoolEntry* collQuarantineTail;
  uint64_t collQuarantineEvictions;
  uint32_t collStrideSize;
  uint32_t collTotalSize;
  uint32_t collAllocCount;
  uint32_t collChunkCount;
  pthread_mutex_t collPoolLock;

  // P2P info pool
  struct inspectorPoolChunk* p2pChunkList;
  struct inspectorP2pInfoPoolEntry* p2pFreeList;
  struct inspectorP2pInfoPoolEntry* p2pQuarantineHead;
  struct inspectorP2pInfoPoolEntry* p2pQuarantineTail;
  uint64_t p2pQuarantineEvictions;
  uint32_t p2pStrideSize;
  uint32_t p2pTotalSize;
  uint32_t p2pAllocCount;
  uint32_t p2pChunkCount;
  pthread_mutex_t p2pPoolLock;

  // Comm info pool (keeping for future extensibility)
  struct inspectorPoolChunk* commChunkList;
  struct inspectorCommInfoPoolEntry* commFreeList;
  uint32_t commStrideSize;
  uint32_t commTotalSize;
  uint32_t commAllocCount;
  uint32_t commChunkCount;
  pthread_mutex_t commPoolLock;

  // Controls whether pools are allowed to grow beyond their initial size.
  // Disabled via NCCL_INSPECTOR_POOL_GROW=0.
  bool growEnabled;

  // Quarantine window for released collective/P2P records (0: reuse at once).
  uint64_t quarantineUsecs;
};

// What a proxy operation copies from its collective or P2P parent when it
// starts. All fields are set when the parent starts and never change.
struct inspectorProxyParentView {
  const char* func;
  uint64_t sn;
  size_t msgSizeBytes;
  const char* algo;                  // collective only
  const char* proto;                 // collective only
  inspectorParentIdentity identity;  // collective only
  int64_t applicationStep;
};

enum inspectorPoolRecordState {
  inspectorPoolRecordLive,         // in use: the parent is still running
  inspectorPoolRecordQuarantined,  // released, unchanged inside the quarantine window
  inspectorPoolRecordExpired       // released longer ago than the window (or never quarantined)
};

extern struct inspectorEventPool g_eventPool;

// Memory pool functions
inspectorResult_t inspectorEventPoolInit(uint32_t collPoolSize,
                                        uint32_t p2pPoolSize,
                                        uint32_t commPoolSize);
inspectorResult_t inspectorEventPoolFinalize();

struct inspectorCollInfo* inspectorEventPoolAllocColl();
struct inspectorP2pInfo* inspectorEventPoolAllocP2p();
struct inspectorCommInfo* inspectorEventPoolAllocComm();
void inspectorEventPoolReleaseColl(struct inspectorCollInfo* collInfo);
void inspectorEventPoolReleaseP2p(struct inspectorP2pInfo* p2pInfo);
void inspectorEventPoolReleaseComm(struct inspectorCommInfo* commInfo);

// Copy a parent's fields under the pool lock unless the record has expired.
// releasedAgeUsecs receives the time since release for a quarantined record.
inspectorPoolRecordState inspectorEventPoolViewColl(
  const struct inspectorCollInfo* collInfo, struct inspectorProxyParentView* view,
  uint64_t* releasedAgeUsecs);
inspectorPoolRecordState inspectorEventPoolViewP2p(
  const struct inspectorP2pInfo* p2pInfo, struct inspectorProxyParentView* view,
  uint64_t* releasedAgeUsecs);
uint64_t inspectorEventPoolQuarantineEvictions();
uint64_t inspectorEventPoolQuarantineUsecs();
// Monotonic microsecond clock of the quarantine; tests may replace it.
void inspectorEventPoolSetClock(uint64_t (*nowUsecs)());

#endif // NCCL_INSPECTOR_EVENT_POOL_H_
