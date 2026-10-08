#ifndef NCCL_INSPECTOR_PROXY_POOL_H_
#define NCCL_INSPECTOR_PROXY_POOL_H_

#include <stdint.h>

#include "inspector.h"

inspectorResult_t inspectorProxyPoolInit(uint32_t opCapacity,
                                         uint32_t stepCapacity);
void inspectorProxyPoolFinalize();
struct inspectorProxyOpInfo* inspectorProxyPoolAllocOp();
struct inspectorProxyStepInfo* inspectorProxyPoolAllocStep();
void inspectorProxyPoolReleaseOp(struct inspectorProxyOpInfo* op);
void inspectorProxyPoolReleaseStep(struct inspectorProxyStepInfo* step);
uint64_t inspectorProxyPoolDroppedOps();
uint64_t inspectorProxyPoolDroppedSteps();
void inspectorProxyPoolRecordDetachedOp();
uint64_t inspectorProxyPoolDetachedOps();
// Local proxy operations that started after their parent record was released
// (late: inside the quarantine window, identity kept; expired: identity lost).
void inspectorProxyPoolRecordParentLate(uint64_t releasedAgeUsecs);
uint64_t inspectorProxyPoolParentLateOps();
uint64_t inspectorProxyPoolParentLateMaxUsecs();
void inspectorProxyPoolRecordParentExpired();
uint64_t inspectorProxyPoolParentExpiredOps();

#endif
