#include "inspector_proxy_pool.h"

#include <assert.h>

int main() {
  assert(inspectorProxyPoolInit(2, 1) == inspectorSuccess);
  assert(inspectorProxyPoolDetachedOps() == 0);
  inspectorProxyPoolRecordDetachedOp();
  assert(inspectorProxyPoolDetachedOps() == 1);
  inspectorProxyOpInfo* first = inspectorProxyPoolAllocOp();
  inspectorProxyOpInfo* second = inspectorProxyPoolAllocOp();
  assert(first != nullptr);
  assert(second != nullptr);
  assert(inspectorProxyPoolAllocOp() == nullptr);
  assert(inspectorProxyPoolDroppedOps() == 1);
  inspectorProxyPoolReleaseOp(first);
  inspectorProxyOpInfo* reused = inspectorProxyPoolAllocOp();
  assert(reused == first);

  inspectorProxyStepInfo* step = inspectorProxyPoolAllocStep();
  assert(step != nullptr);
  assert(inspectorProxyPoolAllocStep() == nullptr);
  assert(inspectorProxyPoolDroppedSteps() == 1);
  inspectorProxyPoolReleaseStep(step);
  inspectorProxyPoolReleaseStep(step);  // Double release is a no-op.
  assert(inspectorProxyPoolAllocStep() == step);

  inspectorProxyPoolReleaseOp(second);
  inspectorProxyPoolReleaseOp(reused);
  inspectorProxyPoolFinalize();
  return 0;
}
