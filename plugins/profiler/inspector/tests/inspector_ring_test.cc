#include "inspector.h"

#include <assert.h>
#include <stdint.h>
#include <vector>

int main() {
  inspectorCompletedRing ring;
  assert(inspectorRingInit(&ring, 2, sizeof(uint64_t)) == inspectorSuccess);

  uint64_t one = 1;
  uint64_t two = 2;
  uint64_t three = 3;
  assert(inspectorRingEnqueue(&ring, &one) == inspectorSuccess);
  assert(inspectorRingEnqueue(&ring, &two) == inspectorSuccess);
  assert(ring.overwritten == 0);
  assert(inspectorRingEnqueue(&ring, &three) == inspectorSuccess);
  assert(ring.overwritten == 1);

  std::vector<uint64_t> drained;
  assert(inspectorRingDrain<uint64_t>(&ring, drained) == inspectorSuccess);
  assert(drained.size() == 2);
  assert(drained[0] == two);
  assert(drained[1] == three);

  inspectorRingFinalize(&ring);
  assert(ring.overwritten == 0);
  return 0;
}
